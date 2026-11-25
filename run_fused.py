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
from views import load_view,detector_settings,detector_at

GLASS=load_view('glass')
BRANCHES=[(b['name'],tuple(b['shape']),tuple(b['crop']),b['profile']) for b in GLASS['branches']]


def pose_model_paths(models_dir,det):
    """One weight path, or a list for an ensemble: pose_model as a list, and/or tta_flip
    (each model also run on the mirrored input, given as (path, True))."""
    m=det['pose_model'];paths=[Path(models_dir)/x for x in m] if isinstance(m,(list,tuple)) else [Path(models_dir)/m]
    if det.get('tta_flip'):paths=paths+[(p,True) for p in paths]
    return paths if len(paths)>1 else paths[0]


def view_branches(view):
    return [(b['name'],tuple(b['shape']),tuple(b['crop']),b['profile']) for b in view['branches']]
UMBRELLA_SHAPE=(384,640)


def enhance_image(image,mode):
    """Optional contrast enhancement of a BGR network input (detector setting ``enhance``)."""
    if not mode:return image
    if mode=='clahe':
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        lab[...,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[...,0])
        return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    if mode.startswith('gamma:'):
        g=float(mode.split(':',1)[1]);table=np.clip(((np.arange(256)/255.)**g)*255+.5,0,255).astype(np.uint8)
        return cv2.LUT(image,table)
    raise ValueError(f'Unknown enhance mode {mode!r}')


def letterbox(image,shape,enhance=None):
    out=LetterBox(new_shape=shape,auto=False,stride=32)(image=image)
    if enhance:out=enhance_image(out,enhance)
    return np.ascontiguousarray(out[...,::-1].transpose(2,0,1))


class Branch:
    """One counting branch with its own engine, ByteTrack ID allocator and outputs."""
    def __init__(self,name,shape,crop,branch_profile,meta,output,model,device,batch,half,stride,record_detections=False):
        self.name=name;self.shape=shape
        w,h=meta['width'],meta['height']
        self.crop=[round(v*s) for v,s in zip(crop,[w,h,w,h])]
        profile=dict(branch_profile);profile['vectorized_tracking']=True
        self.profile=profile
        self.engine=StreamingCounter(profile,w,h,meta['fps'],stride)
        self.track_counter=BaseTrack._count
        if isinstance(model,(list,tuple)):
            from gpu_backend import EnsembleModel
            paths=[m[0] if isinstance(m,tuple) else m for m in model];flips=[isinstance(m,tuple) and m[1] for m in model]
            self.net=EnsembleModel(paths,device,shape,'graph',batch,half,flips)
        else:self.net=GpuModel(model,device,shape,'graph',batch,half)
        self.folder=output/name;self.folder.mkdir(parents=True,exist_ok=True)
        self.tracks=(self.folder/'tracks.jsonl').open('w')
        self.detections=(self.folder/'detections.jsonl').open('w') if record_detections else None
        self.events=[];self.observations=0;self.tracking_seconds=0.

    def update(self,item):
        # ByteTrack allocates IDs from a class variable; keep one allocator per
        # branch so umbrella track IDs match a separate-process run.
        if self.detections:self.detections.write(json.dumps(item)+'\n')
        BaseTrack._count=self.track_counter
        t=time.perf_counter();record,events,_=self.engine.update(item);self.tracking_seconds+=time.perf_counter()-t
        self.track_counter=BaseTrack._count
        self.tracks.write(json.dumps(record)+'\n');self.events+=events;self.observations+=1
        return record,events

    def close(self,summary):
        self.tracks.close()
        if self.detections:self.detections.close()
        with (self.folder/'events.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['event_id','direction','frame','seconds','track_id','x','y'])
            writer.writerows([e['event_id'],e['direction'],e['frame'],e['seconds'],e['track_id'],e['x'],e['y']] for e in self.events)
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
    p.add_argument('--half',action='store_true',help='FP16 networks (changes detections slightly; validate before use)')
    p.add_argument('--record-detections',action='store_true',help='Save per-branch detections for tools/replay_view.py')
    p.add_argument('--stride',type=int,default=2,help='Process every Nth frame (default 2 = 12.5 observations/s, as calibrated)')
    p.add_argument('--view',default='glass',help='Camera view preset: name in views/ or a JSON path (default: glass)')
    p.add_argument('--assume-fps',type=float,help='Count as if the source had this frame rate, as the live pipeline does when the camera lowers its rate in the dark (e.g. 25 for a 20 fps night recording)')
    p.add_argument('--clock-start',help="Local time of frame 0 ('YYYY-MM-DD HH:MM:SS'): applies the view's detector schedule by time of day, as the live pipeline does")
    p.add_argument('--detector',default='{}',help='JSON overrides of the view\'s detector settings (views.detector_settings), e.g. \'{"pose_model":"yolo11x-pose.pt"}\'')
    a=p.parse_args();a.source=[s.resolve() for s in a.source];a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an unused output directory')
    stride=a.stride
    meta=validate_sources(a.source,max_seconds=a.max_seconds,threads=a.decode_threads)
    w,h,fps=meta['width'],meta['height'],meta['fps']
    view=load_view(a.view)
    count_meta=dict(meta)
    if a.assume_fps:count_meta['fps']=fps=a.assume_fps   # decoding still checks the real rate (meta)
    if (w,h)!=(view['width'],view['height']) or abs(fps-view['fps'])>.01:   # camera recordings average 24.998
        raise ValueError(f"View {view['name']} requires {view['width']}x{view['height']} at {view['fps']} fps")
    det=detector_settings(view,json.loads(a.detector));half=a.half or det['half']
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    a.output.mkdir(parents=True,exist_ok=True)
    branches=[Branch(n,s,c,o,count_meta,a.output,pose_model_paths(a.models_dir,det),a.device,a.batch,half,stride,a.record_detections) for n,s,c,o in view_branches(view)]
    umbrella=GpuModel(a.models_dir/'yolo11n.pt',a.device,UMBRELLA_SHAPE,'graph',a.batch,a.half)
    if any(b.net.task!='pose' for b in branches):raise ValueError('Glass requires a pose model')
    startup=time.perf_counter()-started

    reader=SimpleNamespace(decoder='cv2',source=a.source,start_seconds=0,decode_threads=a.decode_threads,
                           max_seconds=a.max_seconds,stride=stride,pace=False,batch=a.batch)
    decoded=queue.Queue(maxsize=3);prepared=queue.Queue(maxsize=3);stop=threading.Event()
    wall=time.perf_counter()
    producer=threading.Thread(target=produce,args=(reader,decoded,stop,dict(width=w,height=h,fps=meta['fps']),wall,0),daemon=True)
    producer.start()
    pool=ThreadPoolExecutor(a.prep_threads)
    jobs=[(b.shape,b.crop,det['enhance']) for b in branches]+[(UMBRELLA_SHAPE,[0,0,w,h],None)]

    def prep_one(image,shape,crop,enhance):
        x1,y1,x2,y2=crop
        return letterbox(image[y1:y2,x1:x2],shape,enhance)

    def prepare():
        try:
            while True:
                batch=decoded.get()
                if batch is None or isinstance(batch,BaseException):put_buffer(prepared,batch,stop);return
                futures=[[pool.submit(prep_one,r['image'],shape,crop,enhance) for r in batch] for shape,crop,enhance in jobs]
                arrays=[np.stack([f.result() for f in fs]) for fs in futures]
                shapes=[[(c[3]-c[1],c[2]-c[0],3)]*len(batch) for _,c,_ in jobs]
                for r in batch:r.pop('image')  # free full frames early
                if not put_buffer(prepared,(batch,arrays,shapes),stop):return
        except BaseException as exc:put_buffer(prepared,exc,stop)
    preparer=threading.Thread(target=prepare,daemon=True);preparer.start()

    nets=[b.net for b in branches]+[umbrella]
    inference=0.;observations=0
    if a.clock_start:
        from datetime import datetime,timedelta
        clock0=datetime.fromisoformat(a.clock_start);switches=[(None,None,None)]
    try:
        while True:
            item=prepared.get()
            if isinstance(item,BaseException):raise item
            if item is None:break
            batch,arrays,shapes=item;t=time.perf_counter()
            for net,array,shape in zip(nets,arrays,shapes):
                if len(array)<a.batch:array=np.concatenate([array,np.zeros((a.batch-len(array),*array.shape[1:]),np.uint8)])
                net.enqueue_prepared(array,shape)
            now=det
            if a.clock_start:
                now=detector_at(view,clock0+timedelta(seconds=batch[0]['index']/meta['fps']),json.loads(a.detector))
                if (now['conf'],now['iou'])!=switches[-1][1:]:switches.append((batch[0]['index'],now['conf'],now['iou']))
            people=[net.collect(now['conf'],now['iou'],[0]) for net in nets[:2]]
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
    t=time.perf_counter();result=merge(a.output/'full',a.output/'roi',a.output/'result',profile=view['profile'],size=(w,h));merge_seconds=time.perf_counter()-t
    video_seconds=meta['end_frame']/meta['fps']
    result.update(source=list(map(str,a.source)),view=view['name'],mode='fused single-pass video inference',half=half,detector=det,
                  detector_switches=switches[1:] if a.clock_start else None,
                  startup_seconds=startup,processing_seconds=processing,merge_seconds=merge_seconds,
                  elapsed_seconds=time.perf_counter()-started,inference_wall_seconds=inference,
                  observations=observations,all_observations_processed=complete,video_seconds=video_seconds,
                  realtime_factor=video_seconds/processing,peak_process_rss_mb=peak_rss_mb(),
                  gpu_name=torch.cuda.get_device_name(int(a.device)),
                  peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(int(a.device))/1048576)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='added'}))


if __name__=='__main__':main()
