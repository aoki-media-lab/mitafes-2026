"""Live RTSP version of run_fused.py: camera -> decode -> 3 networks -> 2 counters -> streaming merge.

Purpose: verify the data flow for real-time operation (keeping up with 25 fps input,
bounded latency, no unbounded buffering, reconnection). Counting parameters are the
calibrated glass preset, so counts are meaningless for a camera with another view.

The RTSP URL is read from ~/.mitafes_rtsp (mode 600) unless --url is given; it is never
printed or written to outputs.
"""
import argparse,json,os,queue,subprocess,threading,time,sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_fused import Branch,UMBRELLA_SHAPE,letterbox,to_items,load_setup,branch_specs,line_band,scheduled_overrides,schedule_keys,apply_overrides
from live_web import geometry,day_history
from gpu_backend import GpuModel
from stream_merge import StreamMerger
from runtime_control import peak_rss_mb

W,H,FPS,STRIDE=2880,1620,25.0,2


def redact(text):
    text=str(text)
    return text.split('@',1)[-1] if '@' in text else text


class LiveReader(threading.Thread):
    """Reads every frame (HEVC needs it), keeps every STRIDE-th, never blocks on the consumer.

    A full queue means processing fell behind: that batch is dropped and counted,
    instead of letting the RTSP/TCP backlog and latency grow without bound.
    """
    def __init__(self,url,out,batch,stop,disconnect_at=None):
        super().__init__(daemon=True)
        self.url=url;self.out=out;self.batch=batch;self.stop=stop;self.disconnect_at=disconnect_at
        self.stats=dict(received=0,kept=0,dropped_frames=0,reconnects=0,read_failures=0,open_failures=0)
        self.t0=None;self.pts0=None;self.last_drift=None;self.error=None

    def open(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp|stimeout;5000000'
        cap=cv2.VideoCapture(self.url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,8])
        if not cap.isOpened():return None
        size=(int(cap.get(3)),int(cap.get(4)))
        if size!=(W,H):cap.release();raise ValueError(f'Stream is {size}, preset needs {(W,H)}')
        return cap

    def run(self):
        if isinstance(self.url,Path):return self.run_file()
        index=0;batch=[];simulated=False
        try:
            while not self.stop.is_set():
                cap=self.open()
                if cap is None:
                    self.stats['open_failures']+=1;self.stop.wait(2);continue
                if self.t0 is None:self.t0=time.perf_counter()
                else:
                    # Continue the time axis across the outage so the counters see a gap.
                    if batch:
                        try:self.out.put_nowait(batch)
                        except queue.Full:self.stats['dropped_frames']+=len(batch)
                        batch=[]
                    index=max(index,round((time.perf_counter()-self.t0)*FPS));index+=(-index)%STRIDE;self.pts0=None
                while not self.stop.is_set():
                    if self.disconnect_at and not simulated and time.perf_counter()-self.t0>=self.disconnect_at:
                        simulated=True;self.stats['reconnects']+=1;print(json.dumps(dict(event='simulated_disconnect')),flush=True);break
                    keep=index%STRIDE==0
                    if keep:ok,im=cap.read()
                    else:ok=cap.grab()
                    if not ok:self.stats['read_failures']+=1;self.stats['reconnects']+=1;break
                    now=time.perf_counter();self.stats['received']+=1
                    pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if self.pts0 is None:self.pts0=(pts,now)
                    # Wall time minus stream time since connect: grows if decode falls behind the camera.
                    self.last_drift=(now-self.pts0[1])-(pts-self.pts0[0])
                    if keep:
                        self.stats['kept']+=1;batch.append(dict(index=index,image=im,arrival=now))
                        if len(batch)==self.batch:
                            try:self.out.put_nowait(batch)
                            except queue.Full:self.stats['dropped_frames']+=len(batch)
                            batch=[]
                    index+=1
                cap.release()
        except BaseException as exc:self.error=exc
        finally:
            if batch:self.out.put(batch)
            self.out.put(None)

    def run_file(self):
        """Virtual camera: frame i of a recording 'arrives' at t0 + i/25 s, like a live stream.

        Latency is measured from that scheduled arrival, so a reader that falls behind
        real time shows up as growing latency / drift, exactly as with a camera backlog.
        """
        batch=[]
        try:
            cap=cv2.VideoCapture(str(self.url),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,8])
            if not cap.isOpened():raise RuntimeError('Cannot open recording')
            size=(int(cap.get(3)),int(cap.get(4)))
            if size!=(W,H):raise ValueError(f'Recording is {size}, preset needs {(W,H)}')
            self.t0=time.perf_counter();index=0
            while not self.stop.is_set():
                scheduled=self.t0+index/FPS;delay=scheduled-time.perf_counter()
                if delay>0 and self.stop.wait(delay):break
                keep=index%STRIDE==0
                if keep:ok,im=cap.read()
                else:ok=cap.grab()
                if not ok:break
                now=time.perf_counter();self.stats['received']+=1;self.last_drift=now-scheduled
                if keep:
                    self.stats['kept']+=1;batch.append(dict(index=index,image=im,arrival=scheduled))
                    if len(batch)==self.batch:
                        try:self.out.put_nowait(batch)
                        except queue.Full:self.stats['dropped_frames']+=len(batch)
                        batch=[]
                index+=1
            cap.release()
        except BaseException as exc:self.error=exc
        finally:
            if batch:self.out.put(batch)
            self.out.put(None)


def snapshot(path,image,profile,roi_crop,people,records,totals,highlight=(),trails=None):
    """Live frame with the count line, ROI, raw detections and tracked anchors."""
    img=image.copy()
    for pts,kind in geometry(profile,W,H):
        color,t={'line':((0,0,255),4),'margin':((0,200,255),2),'roi':((0,255,0),2)}[kind]
        cv2.polylines(img,[pts.astype(np.int32)],kind=='roi',color,t)
    x1,y1,x2,y2=roi_crop;cv2.rectangle(img,(x1,y1),(x2-1,y2-1),(255,0,0),2)
    for b in people.boxes.data.cpu().numpy():
        cv2.rectangle(img,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,255,255),2)
    for color,record in zip([(0,255,255),(255,0,255)],records):
        for person in record['people']:
            if 'anchor' not in person:continue
            x,y=person['anchor']
            cv2.circle(img,(int(x),int(y)),10,color,-1);cv2.putText(img,str(person['id']),(int(x)+12,int(y)),0,1.2,color,3)
    for tid,direction in highlight:
        trail=np.asarray(list(trails.get(tid,[]))) if trails else []
        if len(trail):
            cv2.polylines(img,[trail.astype(np.int32)],False,(0,0,255),5)
            cv2.putText(img,f'{direction} #{tid}',(int(trail[-1][0])+15,int(trail[-1][1])-20),0,2,(0,0,255),5)
    cv2.putText(img,json.dumps(totals),(40,H-120),0,1.6,(0,0,255),4)
    cv2.imwrite(str(path),cv2.resize(img,(1440,810)),[cv2.IMWRITE_JPEG_QUALITY,85])


def gpu_power():
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=2).stdout.strip().split(',')
        return dict(power_w=float(out[0]),temp_c=int(out[1]),util=int(out[2]))
    except Exception:return None


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',help='RTSP URL (default: contents of ~/.mitafes_rtsp)')
    p.add_argument('--file',type=Path,help='Replay a recording at real-time pace instead of a camera')
    p.add_argument('--duration',type=float,default=600,help='Seconds to run')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='0')
    p.add_argument('--models-dir',type=Path,default=ROOT/'models')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--prep-threads',type=int,default=6)
    p.add_argument('--queue-batches',type=int,default=4)
    p.add_argument('--simulate-disconnect-at',type=float,help='Drop and reopen the stream after N seconds')
    p.add_argument('--setup',type=Path,help='Camera setup JSON (line, ROI polygon, roi crop) for a changed view')
    p.add_argument('--snapshot-every',type=float,default=0,help='Save an annotated live frame every N seconds (0: off)')
    p.add_argument('--web-port',type=int,default=0,help='Serve a live annotated view on this port (0: off)')
    p.add_argument('--web-host',default='127.0.0.1',help='Interface for the web view (no authentication; keep it private)')
    p.add_argument('--web-width',type=int,default=960,help='Width of the streamed view (bandwidth: ~8 Mbps at 960/65)')
    p.add_argument('--web-quality',type=int,default=65,help='JPEG quality of the streamed view')
    p.add_argument('--web-fps',type=float,default=0,help='Cap on streamed frames per second (0: every observation, 12.5)')
    p.add_argument('--web-token-file',type=Path,help='Serve the web view only under /<token>/; the token is read from this file, created (mode 600) if missing')
    p.add_argument('--web-soften',type=float,default=0,help='Gaussian smoothing (sigma, px) of the streamed picture before overlays; 0.7 cuts bandwidth ~1/3')
    p.add_argument('--web-max-mbps',type=float,default=0,help='Cap on the stream bitrate per viewer; lowers the frame rate when frames are large (0: off)')
    p.add_argument('--record',action='store_true',help='Also record the annotated view (12.5 fps H.264, hourly files) to <output>/video; needs --web-port')
    p.add_argument('--record-segment',type=int,default=3600,help='Seconds per recorded file (cut at clock times)')
    p.add_argument('--event-snapshots',action='store_true',help='Save an annotated frame with the trajectory for every full-branch count')
    a=p.parse_args()
    url=a.file.resolve() if a.file else (a.url or (Path.home()/'.mitafes_rtsp').read_text().strip())
    a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an unused output directory')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    meta=dict(width=W,height=H,fps=FPS)
    setup=load_setup(a.setup);specs=branch_specs(setup)
    (a.output/'setup.json').write_text(json.dumps(setup|dict(source=str(a.setup) if a.setup else None),indent=1)+'\n')
    if a.snapshot_every:(a.output/'snapshots').mkdir()
    if a.event_snapshots:(a.output/'event_snapshots').mkdir()
    branches=[Branch(n,s,c,o,meta,a.output,a.models_dir/'yolo11m-pose.pt',a.device,a.batch,False,STRIDE) for n,s,c,o in specs]
    # Detection floor and time-of-day profile changes (setup 'detect_conf', 'schedule'); checked every 10 s.
    conf=setup.get('detect_conf',.1);active=None
    bases=[{k:b.profile.get(k) for k in schedule_keys(setup)} for b in branches]
    def reschedule():
        nonlocal active
        values=scheduled_overrides(setup,time.time())
        if values!=active:
            for b,base in zip(branches,bases):apply_overrides(b.profile,base,values)
            if active is not None or values:print(json.dumps(dict(event='profile_schedule',clock=time.strftime('%H:%M:%S'),values=values)),flush=True)
            active=values
    reschedule()
    umbrella=GpuModel(a.models_dir/'yolo11n.pt',a.device,UMBRELLA_SHAPE,'graph',a.batch,False)
    nets=[b.net for b in branches]+[umbrella]
    jobs=[(b.shape,b.crop) for b in branches]+[(UMBRELLA_SHAPE,[0,0,W,H])]
    band=line_band(branches[0].profile,W);merger=StreamMerger(band);startup=time.perf_counter()-started
    web=None;token=None
    if a.web_port and a.web_token_file:
        f=a.web_token_file.expanduser()
        if not f.exists():
            import secrets
            f.touch(mode=0o600);f.write_text(secrets.token_urlsafe(18)+'\n')
        token=f.read_text().strip()
    if a.web_port:
        from live_web import LiveWeb
        # Today's earlier runs seed the day's totals; 'covered' keeps the directions all of them counted.
        covered=set(branches[0].profile.get('count_directions',['ltr','rtl']))
        history=day_history(a.output.parent,set(covered),exclude=a.output,covered=covered)
        web=LiveWeb(a.web_host,a.web_port,W,H,branches[0].profile,branches[1].crop,a.web_width,a.web_quality,a.web_fps,
                   history,a.web_max_mbps,token,a.web_soften,day_label=time.strftime('%-m/%-d'),runs=a.output.parent,
                   record_dir=a.output/'video' if a.record else None,record_segment=a.record_segment,covered=covered)
        # The token is not written to logs; it is in the token file.
        print(json.dumps(dict(event='web',url=f'http://{a.web_host}:{a.web_port}/'+('<token>/' if token else ''))),flush=True)
    print(json.dumps(dict(event='ready',startup_seconds=round(startup,1),source=redact(url))),flush=True)

    frames=queue.Queue(maxsize=a.queue_batches);stop=threading.Event()
    reader=LiveReader(url,frames,a.batch,stop,a.simulate_disconnect_at);reader.start()
    pool=ThreadPoolExecutor(a.prep_threads)
    log=(a.output/'live_status.jsonl').open('w');events_log=(a.output/'live_events.jsonl').open('w')
    window=deque();all_latency=[];processed=0;last_report=time.perf_counter();last_index=-1;gaps=0
    last_snapshot=-1e9;trails={}
    stage=dict(prep=0.,gpu=0.,items=0.,track=0.,merge=0.,web=0.);stage_obs=0   # seconds per pipeline stage (status rows)
    deadline=time.perf_counter()+a.duration
    def prep(image,shape,crop):
        x1,y1,x2,y2=crop;return letterbox(image[y1:y2,x1:x2],shape)
    try:
        while True:
            if time.perf_counter()>=deadline:stop.set()
            try:batch=frames.get(timeout=1)
            except queue.Empty:
                if stop.is_set() and not reader.is_alive():break
                continue
            if batch is None:break
            t0=time.perf_counter()
            arrays=[np.stack([f.result() for f in [pool.submit(prep,r['image'],s,c) for r in batch]]) for s,c in jobs]
            shapes=[[(c[3]-c[1],c[2]-c[0],3)]*len(batch) for _,c in jobs]
            shot=False
            if a.snapshot_every and time.perf_counter()-last_snapshot>=a.snapshot_every:
                shot=True;last_snapshot=time.perf_counter()
            images=[r.pop('image') for r in batch] if (shot or a.event_snapshots or web) else None
            for r in batch:r.pop('image',None)
            t1=time.perf_counter();stage['prep']+=t1-t0
            for net,array,shape in zip(nets,arrays,shapes):
                if len(array)<a.batch:array=np.concatenate([array,np.zeros((a.batch-len(array),*array.shape[1:]),np.uint8)])
                net.enqueue_prepared(array,shape)
            people=[net.collect(conf,.7,[0]) for net in nets[:2]];umbrellas=umbrella.collect(.1,.7,[25])
            t2=time.perf_counter();stage['gpu']+=t2-t1
            per_branch=[to_items(batch,ps,umbrellas,b.crop,W,H) for b,ps in zip(branches,people)]
            stage['items']+=time.perf_counter()-t2;stage_obs+=len(batch)
            for i,source in enumerate(batch):
                if last_index>=0 and source['index']!=last_index+STRIDE:gaps+=1
                last_index=source['index']
                t3=time.perf_counter()
                (rf,ef),(rr,er)=[b.update(items[i]) for b,items in zip(branches,per_branch)]
                t4=time.perf_counter();stage['track']+=t4-t3
                added=merger.update(source['index'],rf,rr,ef,er)
                t5=time.perf_counter();stage['merge']+=t5-t4
                now=time.perf_counter()
                clock=round(time.time(),2)
                for e in ef:events_log.write(json.dumps(dict(e,source='full',wall=now,clock=clock))+'\n')
                for e in added:events_log.write(json.dumps(dict(e,source='roi_added',wall=now,clock=clock))+'\n')
                latency=now-source['arrival'];window.append(latency);all_latency.append(latency);processed+=1
                if web:
                    t6=time.perf_counter()
                    web.submit(source['index'],images[i],people[0][i].boxes.data[:,:4].cpu().numpy(),umbrellas[i].boxes.data[:,:4].cpu().numpy(),
                               (rf,rr),[dict(e,source='full') for e in ef]+[dict(e,source='roi_added') for e in added],
                               merger.totals(),latency,now-reader.t0 if reader.t0 else 0)
                    stage['web']+=time.perf_counter()-t6
                if a.event_snapshots:
                    for person in rf['people']:
                        if 'anchor' in person:trails.setdefault(person['id'],deque(maxlen=40)).append(person['anchor'])
                    if source['index']%250==0:
                        alive={person['id'] for person in rf['people']};trails={k:v for k,v in trails.items() if k in alive}
                elapsed=now-reader.t0 if reader.t0 else 0
                if shot and i==len(batch)-1:
                    snapshot(a.output/'snapshots'/f'{elapsed:07.1f}s_f{source["index"]}.jpg',images[i],branches[0].profile,
                             branches[1].crop,people[0][i],[rf,rr],merger.totals())
                if a.event_snapshots and ef:
                    snapshot(a.output/'event_snapshots'/f'{elapsed:07.1f}s_f{source["index"]}_e{ef[0]["event_id"]}.jpg',images[i],
                             branches[0].profile,branches[1].crop,people[0][i],[rf,rr],merger.totals(),
                             [(e['track_id'],e['direction']) for e in ef],trails)
            if time.perf_counter()-last_report>=10:
                reschedule()
                lat=sorted(window);window.clear();last_report=time.perf_counter()
                elapsed=last_report-reader.t0 if reader.t0 else 0
                row=dict(t=round(elapsed,1),received_fps=round(reader.stats['received']/elapsed,2) if elapsed else None,
                         processed_obs=processed,queue_batches=frames.qsize(),dropped_frames=reader.stats['dropped_frames'],
                         reconnects=reader.stats['reconnects'],gaps=gaps,
                         decode_drift_s=round(reader.last_drift,3) if reader.last_drift is not None else None,
                         latency_p50=round(lat[len(lat)//2],3) if lat else None,latency_p95=round(lat[int(len(lat)*.95)],3) if lat else None,
                         counts=merger.totals(),gpu=gpu_power(),rss_mb=round(peak_rss_mb()),
                         stage_ms_per_obs={k:round(v/max(stage_obs,1)*1000,1) for k,v in stage.items()})
                stage={k:0. for k in stage};stage_obs=0
                if web:web.set_report(row)
                log.write(json.dumps(row)+'\n');log.flush();events_log.flush();print(json.dumps(row),flush=True)
    finally:
        stop.set();reader.join(timeout=10);pool.shutdown(cancel_futures=True);log.close()
        if web:web.close()
    for e in merger.finish():events_log.write(json.dumps(dict(e,source='roi_added',wall=time.perf_counter(),clock=round(time.time(),2)))+'\n')
    events_log.close()
    for b in branches:b.close(dict(mode='live'))
    if reader.error:raise reader.error
    lat=sorted(all_latency)
    from merge_indexed import merge
    offline=merge(a.output/'full',a.output/'roi',a.output/'result_offline_check',band)
    summary=dict(duration_seconds=a.duration,startup_seconds=startup,reader=reader.stats,processed_observations=processed,
                 expected_observations_at_12_5fps=round(a.duration*FPS/STRIDE),index_gaps=gaps,
                 latency_p50=lat[len(lat)//2] if lat else None,latency_p95=lat[int(len(lat)*.95)] if lat else None,
                 latency_max=lat[-1] if lat else None,streaming_counts=merger.totals(),
                 offline_merge_counts=dict(ltr=offline['ltr'],rtl=offline['rtl'],events=offline['events']),
                 streaming_equals_offline=merger.totals()['events']==offline['events'],peak_process_rss_mb=peak_rss_mb(),
                 note='Counts use the calibrated glass preset; with a different camera view they are not accuracy measurements.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
