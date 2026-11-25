"""Live RTSP version of run_fused.py: camera -> decode -> 3 networks -> 2 counters -> streaming merge.

Purpose: verify the data flow for real-time operation (keeping up with 25 fps input,
bounded latency, no unbounded buffering, reconnection). Counting parameters are the
calibrated glass preset, so counts are meaningless for a camera with another view.

The RTSP URL is read from ~/.mitafes_rtsp (mode 600) unless --url is given; it is never
printed or written to outputs.
"""
import argparse,datetime,json,math,os,queue,signal,subprocess,threading,time,sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_fused import Branch,UMBRELLA_SHAPE,letterbox,to_items,view_branches,pose_model_paths
from views import load_view,detector_settings,detector_at
from gpu_backend import GpuModel
from stream_merge import StreamMerger
from runtime_control import peak_rss_mb

W,H,FPS,STRIDE=2880,1620,25.0,2   # overwritten from --view in main()


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


class PreviewPublisher(threading.Thread):
    """Encodes processed frames for the web UI and releases them evenly in time.

    Frames are processed in batches of 8 (0.64 s), so publishing them as they finish would
    show bursts. Each frame is shown ``delay`` seconds after it arrived from the camera,
    which is later than any normal processing latency, so playback is smooth. A frame that
    is already too late is skipped; a full queue drops the batch and never blocks counting.

    Overlay (the page legend uses the same colours): counting line red, margin band orange,
    counting area (feet) green, reconciliation crop blue, people white, umbrellas cyan,
    tracked head positions as dots with their recent trail.
    """
    PALETTE=[(255,99,71),(255,215,0),(124,252,0),(0,191,255),(238,130,238),(255,165,0),(64,224,208),(255,105,180)]

    def __init__(self,web,view,width,height,size=(768,432),quality=60,delay=1.2):
        super().__init__(daemon=True)
        from line_counter import CountingLine
        self.web=web;self.queue=queue.Queue(maxsize=3);self.size=size;self.quality=quality;self.delay=delay
        sx,sy=size[0]/width,size[1]/height;self.scale=(sx,sy)
        prof=view['profile'];line=CountingLine(prof,width,height)
        if not prof.get('line'):line.p1,line.p2=(prof['line_x']*width,0.),(prof['line_x']*width,float(height))
        n=np.asarray(line.normal)*prof['margin']*width
        pt=lambda q:(round(q[0]*sx),round(q[1]*sy))
        self.line=(pt(line.p1),pt(line.p2))
        self.margins=[(pt(np.asarray(line.p1)+k*n),pt(np.asarray(line.p2)+k*n)) for k in (-1,1)]
        self.roi=None
        if prof.get('roi_mode')=='polygon':
            self.roi=np.asarray([(round(x*size[0]),round(y*size[1])) for x,y in prof['roi_polygon']],np.int32)
        crops=[b['crop'] for b in view['branches'] if b['name']!='full']
        self.crops=[(pt((c[0]*width,c[1]*height)),pt((c[2]*width,c[3]*height))) for c in crops]
        self.trails={};self.skipped=0;self.dropped_batches=0;self.published=deque(maxlen=64)

    def put(self,frames):
        try:self.queue.put_nowait(frames)
        except queue.Full:self.dropped_batches+=1

    def fps(self):
        now=time.perf_counter();recent=[t for t in self.published if now-t<=3]
        return len(recent)/3 if recent else 0.

    def draw(self,small,frame,people,umbrellas):
        sx,sy=self.scale
        if self.roi is not None:cv2.polylines(small,[self.roi],True,(80,200,80),1,cv2.LINE_AA)
        for c1,c2 in self.crops:cv2.rectangle(small,c1,c2,(230,140,40),1)
        for m1,m2 in self.margins:cv2.line(small,m1,m2,(0,165,255),1,cv2.LINE_AA)
        cv2.line(small,*self.line,(60,60,240),2,cv2.LINE_AA)
        for u in umbrellas:
            x1,y1,x2,y2=u['xyxy'];cv2.rectangle(small,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),(255,255,0),1)
        seen=set()
        for q in people:
            x1,y1,x2,y2=q['xyxy'];cv2.rectangle(small,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),(255,255,255),1)
            if 'anchor' in q:
                tid=q['id'];seen.add(tid);a=(round(q['anchor'][0]*sx),round(q['anchor'][1]*sy))
                trail=self.trails.setdefault(tid,[deque(maxlen=15),frame]);trail[0].append(a);trail[1]=frame
                color=self.PALETTE[tid%len(self.PALETTE)][::-1]
                if len(trail[0])>1:cv2.polylines(small,[np.asarray(trail[0],np.int32)],False,color,2,cv2.LINE_AA)
                cv2.circle(small,a,4,color,-1,cv2.LINE_AA)
        for tid in [t for t,(_,last) in self.trails.items() if frame-last>50]:self.trails.pop(tid)

    def run(self):
        while True:
            frames=self.queue.get()
            if frames is None:return
            for arrival,image,frame,people,umbrellas in frames:
                wait=arrival+self.delay-time.perf_counter()
                if wait<-.3:self.skipped+=1;continue
                small=cv2.resize(image,self.size,interpolation=cv2.INTER_AREA)
                self.draw(small,frame,people,umbrellas)
                ok,jpeg=cv2.imencode('.jpg',small,[cv2.IMWRITE_JPEG_QUALITY,self.quality])
                wait=arrival+self.delay-time.perf_counter()
                if wait>0:time.sleep(wait)
                if ok:self.web.set_snapshot(jpeg.tobytes());self.published.append(time.perf_counter())


def gpu_power():
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=2).stdout.strip().split(',')
        return dict(power_w=float(out[0]),temp_c=int(out[1]),util=int(out[2]))
    except Exception:return None


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',help='RTSP URL (default: contents of ~/.mitafes_rtsp)')
    p.add_argument('--file',type=Path,help='Replay a recording at real-time pace instead of a camera')
    p.add_argument('--duration',type=float,default=600,help='Seconds to run (0 = until Ctrl-C / SIGTERM)')
    p.add_argument('--web-port',type=int,default=8080,help='Port of the live web UI (0 = off)')
    p.add_argument('--web-host',default='0.0.0.0',help='Address the web UI listens on')
    p.add_argument('--daily-dir',type=Path,default=ROOT/'runs'/'live_daily',help='One file per day with totals and per-minute counts; kept across restarts')
    p.add_argument('--no-snapshot',action='store_true',help='Do not publish camera preview images on the web UI')
    p.add_argument('--record-video',type=Path,help='Save the counting as a visualization video here (one file per hour, see live_recorder.py)')
    p.add_argument('--record-directions',nargs='+',default=['rtl'],choices=['rtl','ltr'],help='Directions highlighted and counted in the video (default rtl)')
    p.add_argument('--output',type=Path);p.add_argument('--device',default='0')
    p.add_argument('--models-dir',type=Path,default=ROOT/'models')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--prep-threads',type=int,default=6)
    p.add_argument('--queue-batches',type=int,default=4)
    p.add_argument('--view',default='glass',help='Camera view preset: name in views/ or a JSON path (default: glass)')
    p.add_argument('--simulate-disconnect-at',type=float,help='Drop and reopen the stream after N seconds')
    return p


def source_url(a):
    return a.file.resolve() if a.file else (a.url or (Path.home()/'.mitafes_rtsp').read_text().strip())


def main():
    a=build_parser().parse_args()
    if a.output is None:raise SystemExit('--output is required')
    view=load_view(a.view);web=None
    if a.web_port:
        from live_web import LiveState,serve
        web=LiveState(view['name'],daily_dir=a.daily_dir,entry_direction=view.get('entry_direction'));serve(web,a.web_host,a.web_port)
        print(json.dumps(dict(event='web_ui',port=a.web_port)),flush=True)
    stop=threading.Event()
    def request_stop(signum,frame):
        print(json.dumps(dict(event='stop_requested',signal=signum)),flush=True);stop.set()
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    until=time.time()+a.duration if a.duration>0 else None
    run_session(a,view,a.output,web=web,until=until,external_stop=stop)


def run_session(a,view,output,web=None,until=None,external_stop=None):
    """One counting session from the camera (or a paced recording) until ``until`` (clock time),
    ``external_stop`` or the end of the input. Writes the run folder and returns its summary."""
    started=time.perf_counter()
    global W,H,FPS
    W,H,FPS=view['width'],view['height'],float(view['fps'])
    url=source_url(a)
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose an unused output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    meta=dict(width=W,height=H,fps=FPS)
    det=detector_settings(view)
    branches=[Branch(n,s,c,o,meta,output,pose_model_paths(a.models_dir,det),a.device,a.batch,det['half'],STRIDE) for n,s,c,o in view_branches(view)]
    umbrella=GpuModel(a.models_dir/'yolo11n.pt',a.device,UMBRELLA_SHAPE,'graph',a.batch,False)
    nets=[b.net for b in branches]+[umbrella]
    jobs=[(b.shape,b.crop,det['enhance']) for b in branches]+[(UMBRELLA_SHAPE,[0,0,W,H],None)]
    merger=StreamMerger(profile=view['profile'],size=(W,H));startup=time.perf_counter()-started
    print(json.dumps(dict(event='ready',startup_seconds=round(startup,1),source=redact(url),detector=det)),flush=True)

    frames=queue.Queue(maxsize=a.queue_batches);stop=threading.Event()
    reader=LiveReader(url,frames,a.batch,stop,a.simulate_disconnect_at);reader.start()
    pool=ThreadPoolExecutor(a.prep_threads)
    log=(output/'live_status.jsonl').open('w');events_log=(output/'live_events.jsonl').open('w')
    session_start=time.time()
    (output/'run_info.json').write_text(json.dumps(dict(view=view['name'],started=session_start,pid=os.getpid(),until=until))+'\n')
    window=deque();all_latency=[];processed=0;last_report=time.perf_counter();last_index=-1;gaps=0
    last_web=0.;web_window=deque(maxlen=200);last_gpu=None;last_received=(time.perf_counter(),0)
    publisher=None
    if web and not a.no_snapshot:publisher=PreviewPublisher(web,view,W,H);publisher.start()
    if web:web.begin_session();web.set_state('running');web.set_status(session_started=session_start,session_until=until,next_start=None)
    recorder=None
    if getattr(a,'record_video',None):
        from live_recorder import LiveRecorder
        base=dict(web.base) if web else None    # the video shows the day's totals, as the web page does
        # Events are final up to merger.secondary_delay frames after they happen; the video waits longer.
        delay=max(8.,merger.secondary_delay/FPS+2)
        recorder=LiveRecorder(view,a.record_video,base=base,directions=tuple(a.record_directions),stride=STRIDE,delay=delay);recorder.start()
    completed=False;last_schedule=-1e9
    def prep(image,shape,crop,enhance):
        x1,y1,x2,y2=crop;return letterbox(image[y1:y2,x1:x2],shape,enhance)
    try:
        while True:
            if (until is not None and time.time()>=until) or (external_stop is not None and external_stop.is_set()):stop.set()
            try:batch=frames.get(timeout=1)
            except queue.Empty:
                if stop.is_set() and not reader.is_alive():break
                continue
            if batch is None:break
            arrays=[np.stack([f.result() for f in [pool.submit(prep,r['image'],s,c,e) for r in batch]]) for s,c,e in jobs]
            shapes=[[(c[3]-c[1],c[2]-c[0],3)]*len(batch) for _,c,_ in jobs]
            images=[r['image'] for r in batch];shown=[];recorded=[];wall0=time.time()-time.perf_counter()
            for r in batch:r.pop('image')
            for net,array,shape in zip(nets,arrays,shapes):
                if len(array)<a.batch:array=np.concatenate([array,np.zeros((a.batch-len(array),*array.shape[1:]),np.uint8)])
                net.enqueue_prepared(array,shape)
            if time.perf_counter()-last_schedule>=10:   # detector schedule of the view (time of day)
                last_schedule=time.perf_counter();now=detector_at(view,datetime.datetime.now())
                if (now['conf'],now['iou'])!=(det['conf'],det['iou']):
                    print(json.dumps(dict(event='detector_schedule',conf=now['conf'],iou=now['iou'])),flush=True);det=now
                    if web:web.set_status(detector_conf=det['conf'])
            people=[net.collect(det['conf'],det['iou'],[0]) for net in nets[:2]];umbrellas=umbrella.collect(.1,.7,[25])
            per_branch=[to_items(batch,ps,umbrellas,b.crop,W,H) for b,ps in zip(branches,people)]
            for i,source in enumerate(batch):
                if last_index>=0 and source['index']!=last_index+STRIDE:gaps+=1
                last_index=source['index']
                (rf,ef),(rr,er)=[b.update(items[i]) for b,items in zip(branches,per_branch)]
                if publisher:shown.append((source['arrival'],images[i],source['index'],rf['people'],rf['umbrellas']))
                added=merger.update(source['index'],rf,rr,ef,er)
                ef=merger.primary_accepted   # = ef unless dedup_passages delays primary decisions
                if recorder:
                    recorded.append((source['index'],source['arrival']+wall0,images[i],rf['people'],rr['people']))
                    if ef:recorder.add_events(ef,'full')
                    if added:recorder.add_events(added,'roi')
                now=time.perf_counter()
                clock=time.time()
                for e in ef:events_log.write(json.dumps(dict(e,source='full',wall=now,time=clock))+'\n')
                for e in added:events_log.write(json.dumps(dict(e,source='roi_added',wall=now,time=clock))+'\n')
                latency=now-source['arrival'];window.append(latency);all_latency.append(latency);processed+=1
                web_window.append(latency)
                if web:
                    if ef:web.add_events(ef,'full')
                    if added:web.add_events(added,'roi')
            if publisher:publisher.put(shown)
            if recorder:recorder.put(recorded)
            del images
            if web:
                web.set_counts(merger.totals());web.save()
                if time.perf_counter()-last_web>=1:
                    t=time.perf_counter();last_web=t
                    received=reader.stats['received'];fps=(received-last_received[1])/max(t-last_received[0],1e-6);last_received=(t,received)
                    lat=sorted(web_window)
                    stalled=reader.stats['reconnects']>0 and frames.qsize()==0 and fps<5
                    web.set_state('reconnecting' if stalled else 'running')
                    web.set_status(updated=time.time(),received_fps=fps,dropped_frames=reader.stats['dropped_frames'],
                                   reconnects=reader.stats['reconnects'],latency_p50=lat[len(lat)//2] if lat else None,
                                   latency_p95=lat[int(len(lat)*.95)] if lat else None,gpu=last_gpu,
                                   display_fps=publisher.fps() if publisher else None)
            if time.perf_counter()-last_report>=10:
                lat=sorted(window);window.clear();last_report=time.perf_counter()
                elapsed=last_report-reader.t0 if reader.t0 else 0
                row=dict(time=round(time.time(),1),t=round(elapsed,1),received_fps=round(reader.stats['received']/elapsed,2) if elapsed else None,
                         processed_obs=processed,queue_batches=frames.qsize(),dropped_frames=reader.stats['dropped_frames'],
                         reconnects=reader.stats['reconnects'],gaps=gaps,
                         decode_drift_s=round(reader.last_drift,3) if reader.last_drift is not None else None,
                         latency_p50=round(lat[len(lat)//2],3) if lat else None,latency_p95=round(lat[int(len(lat)*.95)],3) if lat else None,
                         counts=merger.totals(),gpu=gpu_power(),rss_mb=round(peak_rss_mb()))
                if recorder:row['recorder']=dict(recorder.stats,buffered=len(recorder.buffer),queued=recorder.frames.qsize())
                last_gpu=row['gpu']
                log.write(json.dumps(row)+'\n');log.flush();events_log.flush();print(json.dumps(row),flush=True)
        completed=True
    finally:
        stop.set();reader.join(timeout=10);pool.shutdown(cancel_futures=True);log.close()
        if publisher:publisher.put(None)
        if recorder and not completed:recorder.close()
    final=merger.finish();final_primary=merger.primary_accepted
    if recorder:recorder.add_events(final_primary,'full');recorder.add_events(final,'roi');recorder.close()
    for e in final_primary:events_log.write(json.dumps(dict(e,source='full',wall=time.perf_counter(),time=time.time()))+'\n')
    for e in final:events_log.write(json.dumps(dict(e,source='roi_added',wall=time.perf_counter(),time=time.time()))+'\n')
    if web:
        if final_primary:web.add_events(final_primary,'full')
        if final:web.add_events(final,'roi')
        web.set_counts(merger.totals());web.save(force=True)
    events_log.close()
    for b in branches:b.close(dict(mode='live'))
    if reader.error:raise reader.error
    lat=sorted(all_latency)
    from merge_indexed import merge
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):   # its printout is not needed and the console may be gone
        offline=merge(output/'full',output/'roi',output/'result_offline_check',profile=view['profile'],size=(W,H))
    ran=time.perf_counter()-reader.t0 if reader.t0 else 0
    summary=dict(view=view['name'],duration_seconds=ran,startup_seconds=startup,reader=reader.stats,processed_observations=processed,
                 expected_observations_at_12_5fps=round(ran*FPS/STRIDE),index_gaps=gaps,
                 latency_p50=lat[len(lat)//2] if lat else None,latency_p95=lat[int(len(lat)*.95)] if lat else None,
                 latency_max=lat[-1] if lat else None,streaming_counts=merger.totals(),
                 offline_merge_counts=dict(ltr=offline['ltr'],rtl=offline['rtl'],events=offline['events']),
                 streaming_equals_offline=merger.totals()['events']==offline['events'],peak_process_rss_mb=peak_rss_mb(),
                 recorder=dict(recorder.stats,segment_files=recorder.segments) if recorder else None,
                 note=f"Counts use the {view['name']} view preset; they are accuracy measurements only for that camera view.")
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    try:print(json.dumps(summary),flush=True)
    except BrokenPipeError:pass   # stopped together with the terminal / tee: the file above is what counts
    del nets,branches,umbrella
    torch.cuda.empty_cache()     # the service keeps running between sessions: give the GPU memory back
    return summary


if __name__=='__main__':main()
