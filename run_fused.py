"""Single-pass glass-camera runner: one decode, three concurrent networks, two counters.

Produces the same two branches as run_accuracy.py (full frame and the x=40-80% crop)
and reconciles them with merge_indexed.py, but:
  * the video is decoded once instead of once per branch,
  * the umbrella network runs once per frame and is shared by both branches
    (run_accuracy.py runs the identical full-frame umbrella pass in both children),
  * letterboxing runs in a CPU thread pool while the GPU works on the previous batch,
  * the person networks for both branches and the umbrella network are enqueued on
    separate CUDA streams so they overlap on the GPU.

Detection parameters, letterboxing, models, input shapes, batch size, profiles,
tracking and reconciliation are unchanged. No reference counts or annotations are read.
"""
import argparse,csv,json,queue,threading,time,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from ultralytics.data.augment import LetterBox
from ultralytics.trackers.basetrack import BaseTrack

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from engine import StreamingCounter
from gpu_backend import GpuModel
from run import produce
from runtime_control import validate_sources,observations_between,put_buffer,peak_rss_mb
from run_accuracy import FULL_PROFILE,ROI_PROFILE

BRANCHES=[('full',(736,1280),(0,0,1,1),FULL_PROFILE),('roi',(736,512),(.4,0,.8,1),ROI_PROFILE)]
UMBRELLA_SHAPE=(384,640)


def load_setup(path):
    """Camera setup JSON: {"profile": {...geometry overrides...}, "roi_crop": [x1,y1,x2,y2]} (fractions),
    optionally "roi_shape": [h, w] (network input of the roi branch, multiples of 32; default 736x512)."""
    if not path:return dict(profile={},roi_crop=None)
    setup=json.loads(Path(path).read_text());setup.setdefault('profile',{});setup.setdefault('roi_crop',None);return setup


def scheduled_overrides(setup,clock):
    """Profile values in force at a local time: setup['schedule'] = [{"from": "HH:MM", "profile": {...}}, ...].
    Entries apply in 'from' order up to the clock time (a later entry overrides an earlier one's keys)."""
    out={};hm=time.strftime('%H:%M',time.localtime(clock))
    for e in sorted(setup.get('schedule') or [],key=lambda e:e['from']):
        if e['from']<=hm:out.update(e['profile'])
    return out


def schedule_keys(setup):return {k for e in setup.get('schedule') or [] for k in e['profile']}


def apply_overrides(profile,base,values):
    """Set the scheduled keys in place (the tracker and counters hold this same dict, so they switch at once):
    the active value, else the branch's own starting value from base (None: absent)."""
    for k,v0 in base.items():
        v=values.get(k,v0)
        if v is None:profile.pop(k,None)
        else:profile[k]=v


def branch_specs(setup):
    def shape(n,s):return tuple(setup['roi_shape']) if n=='roi' and setup.get('roi_shape') else s
    return [(n,shape(n,s),tuple(setup['roi_crop']) if n=='roi' and setup['roi_crop'] else c,dict(o,**setup['profile'])) for n,s,c,o in BRANCHES]


def line_band(profile,width):
    """Pixel band [line-margin, line+margin] used by the reconciliation rules (1692..1764 for the glass preset)."""
    line=profile['line_x']*width;margin=profile['margin']*width
    return (round(line-margin,6),round(line+margin,6))


_CLAHE=threading.local()
_LIFT=np.clip(255*(np.arange(256)/255)**0.6,0,255).astype(np.uint8)


def enhance(image,mode):
    """Detector-input contrast fix for deep shadows (the picture shown/recorded is not changed).
    clahe: local contrast on the L channel (LAB), clip 2, 8x8 tiles; lift: global gamma 0.6 (darks up)."""
    if not mode:return image
    if mode=='lift':return cv2.LUT(image,_LIFT)
    if mode=='clahe':
        c=getattr(_CLAHE,'c',None) or cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8));_CLAHE.c=c
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB);lab[...,0]=c.apply(lab[...,0])
        return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    raise ValueError(f'Unknown enhance mode {mode}')


def letterbox(image,shape,mode=None):
    out=enhance(LetterBox(new_shape=shape,auto=False,stride=32)(image=image),mode)
    return np.ascontiguousarray(out[...,::-1].transpose(2,0,1))


class Branch:
    """One counting branch with its own engine, ByteTrack ID allocator and outputs."""
    def __init__(self,name,shape,crop,overrides,meta,output,model,device,batch,half,stride,dump=False):
        self.name=name;self.shape=shape
        w,h=meta['width'],meta['height']
        self.crop=[round(v*s) for v,s in zip(crop,[w,h,w,h])]
        profile=json.loads((ROOT/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())
        profile.update(overrides);profile['vectorized_tracking']=True
        self.profile=profile
        self.engine=StreamingCounter(profile,w,h,meta['fps'],stride)
        self.track_counter=BaseTrack._count
        self.net=GpuModel(model,device,shape,'graph',batch,half) if model else None  # None: replay of dumped detections
        self.folder=output/name;self.folder.mkdir(parents=True,exist_ok=True)
        self.dump=(self.folder/'detections.jsonl').open('w') if dump else None
        self.directions=set(profile.get('count_directions',['ltr','rtl']))
        self.tracks=(self.folder/'tracks.jsonl').open('w')
        self.events=[];self.observations=0;self.tracking_seconds=0.

    def update(self,item):
        # ByteTrack allocates IDs from a class variable; keep one allocator per
        # branch so umbrella track IDs match a separate-process run.
        if self.dump:self.dump.write(json.dumps(item)+'\n')
        BaseTrack._count=self.track_counter
        t=time.perf_counter();record,events,_=self.engine.update(item);self.tracking_seconds+=time.perf_counter()-t
        self.track_counter=BaseTrack._count
        events=[e for e in events if e['direction'] in self.directions]
        self.tracks.write(json.dumps(record)+'\n');self.events+=events;self.observations+=1
        return record,events

    def close(self,summary):
        self.tracks.close()
        if self.dump:self.dump.close()
        with (self.folder/'events.csv').open('w') as f:
            # 'rule' (e.g. late_birth) only when some event has one, so plain runs keep the old format.
            rule=any(e.get('rule') for e in self.events)
            writer=csv.writer(f);writer.writerow(['event_id','direction','frame','seconds','track_id','x','y']+['rule']*rule)
            writer.writerows([e['event_id'],e['direction'],e['frame'],e['seconds'],e['track_id'],e['x'],e['y']]+[e.get('rule','')]*rule for e in self.events)
        (self.folder/'summary.json').write_text(json.dumps(self.engine.summary()|summary|dict(
            observations=self.observations,tracking_seconds=self.tracking_seconds,profile=self.profile),indent=2)+'\n')


def to_items(batch,people,umbrellas,crop,w,h):
    x1,y1=crop[:2];items=[]
    for record,p,u in zip(batch,people,umbrellas):
        det=p.boxes.data.cpu().numpy().copy();points=p.keypoints.data.cpu().numpy().copy()
        if len(det):det[:,[0,2]]+=x1;det[:,[1,3]]+=y1
        if len(points):points[:,:,0]+=x1;points[:,:,1]+=y1
        det=det.tolist();points=points.tolist()
        udets=u.boxes.data.cpu().numpy().copy().tolist()
        det.extend(udets);points.extend([[[0,0,0] for _ in range(17)] for _ in udets])
        items.append(dict(index=record['index'],detections=det,umbrella_observed=True,keypoints=points))
    return items


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float);p.add_argument('--device',default='0')
    p.add_argument('--models-dir',type=Path,default=ROOT/'models')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--decode-threads',type=int,default=12)
    p.add_argument('--prep-threads',type=int,default=6)
    p.add_argument('--setup',type=Path,help='Camera setup JSON (line, ROI polygon, roi crop) for a changed view')
    p.add_argument('--dump-detections',action='store_true',help='Also write each branch\'s per-frame detections for replay_fused.py')
    p.add_argument('--half',action='store_true',help='FP16 networks (changes detections slightly; validate before use)')
    p.add_argument('--enhance',choices=['clahe','lift'],help='Contrast fix of the detector input for deep shadows (experimental)')
    p.add_argument('--conf',type=float,help='Person detection confidence floor (default: the setup\'s detect_conf, else 0.1). With --dump-detections, a lower floor lets replay_fused.py --min-conf try any threshold above it')
    a=p.parse_args();a.source=[s.resolve() for s in a.source];a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an unused output directory')
    stride=2
    meta=validate_sources(a.source,max_seconds=a.max_seconds,threads=a.decode_threads)
    w,h,fps=meta['width'],meta['height'],meta['fps']
    if (w,h)!=(2880,1620) or abs(fps-25)>.001:raise ValueError('This calibrated preset requires the 2880x1620, 25 fps glass camera')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    a.output.mkdir(parents=True,exist_ok=True)
    setup=load_setup(a.setup)
    if a.conf is None:a.conf=setup.get('detect_conf',.1)
    branches=[Branch(n,s,c,o,meta,a.output,a.models_dir/'yolo11m-pose.pt',a.device,a.batch,a.half,stride,a.dump_detections) for n,s,c,o in branch_specs(setup)]
    umbrella=GpuModel(a.models_dir/'yolo11n.pt',a.device,UMBRELLA_SHAPE,'graph',a.batch,a.half)
    if any(b.net.task!='pose' for b in branches):raise ValueError('Glass requires a pose model')
    startup=time.perf_counter()-started

    reader=SimpleNamespace(decoder='cv2',source=a.source,start_seconds=0,decode_threads=a.decode_threads,
                           max_seconds=a.max_seconds,stride=stride,pace=False,batch=a.batch)
    decoded=queue.Queue(maxsize=3);prepared=queue.Queue(maxsize=3);stop=threading.Event()
    wall=time.perf_counter()
    producer=threading.Thread(target=produce,args=(reader,decoded,stop,dict(width=w,height=h,fps=fps),wall,0),daemon=True)
    producer.start()
    pool=ThreadPoolExecutor(a.prep_threads)
    jobs=[(b.shape,b.crop) for b in branches]+[(UMBRELLA_SHAPE,[0,0,w,h])]

    def prep_one(image,shape,crop):
        x1,y1,x2,y2=crop
        return letterbox(image[y1:y2,x1:x2],shape,a.enhance)

    def prepare():
        try:
            while True:
                batch=decoded.get()
                if batch is None or isinstance(batch,BaseException):put_buffer(prepared,batch,stop);return
                futures=[[pool.submit(prep_one,r['image'],shape,crop) for r in batch] for shape,crop in jobs]
                arrays=[np.stack([f.result() for f in fs]) for fs in futures]
                shapes=[[(c[3]-c[1],c[2]-c[0],3)]*len(batch) for _,c in jobs]
                for r in batch:r.pop('image')  # free full frames early
                if not put_buffer(prepared,(batch,arrays,shapes),stop):return
        except BaseException as exc:put_buffer(prepared,exc,stop)
    preparer=threading.Thread(target=prepare,daemon=True);preparer.start()

    nets=[b.net for b in branches]+[umbrella]
    inference=0.;observations=0
    try:
        while True:
            item=prepared.get()
            if isinstance(item,BaseException):raise item
            if item is None:break
            batch,arrays,shapes=item;t=time.perf_counter()
            for net,array,shape in zip(nets,arrays,shapes):
                if len(array)<a.batch:array=np.concatenate([array,np.zeros((a.batch-len(array),*array.shape[1:]),np.uint8)])
                net.enqueue_prepared(array,shape)
            people=[net.collect(a.conf,.7,[0]) for net in nets[:2]]
            umbrellas=umbrella.collect(.1,.7,[25])
            inference+=time.perf_counter()-t
            for b,ps in zip(branches,people):
                for item_ in to_items(batch,ps,umbrellas,b.crop,w,h):b.update(item_)
            observations+=len(batch)
            if observations%400<len(batch):
                print(json.dumps(dict(observations=observations,seconds=round(time.perf_counter()-wall,1),
                                      counts={b.name:b.engine.summary()['events'] for b in branches})),flush=True)
    finally:
        stop.set();producer.join(timeout=5);preparer.join(timeout=5);pool.shutdown(cancel_futures=True)
    processing=time.perf_counter()-wall
    expected=observations_between(0,meta['end_frame'],stride)
    complete=observations==expected
    for b in branches:
        b.close(dict(all_observations_processed=complete,expected_observations=expected,
                     processing_seconds=processing,startup_seconds=startup))
    if not complete:raise RuntimeError(f'Processed {observations} of {expected} planned observations')
    from merge_indexed import merge
    t=time.perf_counter();result=merge(a.output/'full',a.output/'roi',a.output/'result',line_band(branches[0].profile,w));merge_seconds=time.perf_counter()-t
    video_seconds=meta['end_frame']/fps
    result.update(source=list(map(str,a.source)),setup=str(a.setup) if a.setup else None,mode='fused single-pass video inference',half=a.half,enhance=a.enhance,conf=a.conf,
                  startup_seconds=startup,processing_seconds=processing,merge_seconds=merge_seconds,
                  elapsed_seconds=time.perf_counter()-started,inference_wall_seconds=inference,
                  observations=observations,all_observations_processed=complete,video_seconds=video_seconds,
                  realtime_factor=video_seconds/processing,peak_process_rss_mb=peak_rss_mb(),
                  gpu_name=torch.cuda.get_device_name(int(a.device)),
                  peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(int(a.device))/1048576)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='added'}))


if __name__=='__main__':main()
