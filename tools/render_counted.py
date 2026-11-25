"""Render a review video in which every counted person stands out.

Reads a finished run_fused.py output (result/tracks.jsonl and result/events.csv) and draws it
on the source recording with counted_overlay.CountedOverlay (same look as the main-gate live
view): every track with its ID, head trail and white box; a count as "<- #ID" / "-> #ID" on that
track, "+1 roi <- #ID" for a person added from the zoomed branch; IN/OUT totals top right, the
latest counts bottom right, the clock (from --clock-start) bottom left. A track walking over the
red line without turning orange / light blue is therefore a candidate missed count.

Output is 1920x1080 H.264 at the observation rate (12.5 fps at stride 2), so playback runs at
real time. Decoding and encoding use ffmpeg subprocesses; no GPU inference.
"""
import argparse,csv,json,queue,subprocess,sys,threading,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from views import load_view
from counted_overlay import CountedOverlay
OUT_W,OUT_H=1920,1080


def read_exact(stream,size):
    buffer=bytearray(size);view=memoryview(buffer);got=0
    while got<size:
        n=stream.readinto(view[got:])
        if not n:return None if got==0 else bytes(buffer[:got])
        got+=n
    return buffer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Recording the run was made from')
    p.add_argument('--run',type=Path,required=True,help='run_fused.py output directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',default='plaza')
    p.add_argument('--clock-start',required=True,help="Wall-clock time of frame 0, e.g. '2026-11-21 14:00:00'")
    p.add_argument('--directions',nargs='+',default=['rtl','ltr'],choices=['rtl','ltr'],help='Directions to highlight and count')
    p.add_argument('--start',type=float,default=0,help='Render from this many seconds (for previews)')
    p.add_argument('--end',type=float,help='Stop at this many seconds')
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--count-from-start',action='store_true',help='Totals in the corner start at zero at --start (default: they include the counts before it)')
    p.add_argument('--source-fps',type=float,help="Real frame rate of the recording when it differs from the view's (e.g. 20 for a night recording counted with run_fused.py --assume-fps 25): clock and playback use it")
    a=p.parse_args()
    view=load_view(a.view);fps=a.source_fps or float(view['fps'])
    clock0=datetime.fromisoformat(a.clock_start)
    clock=lambda frame:clock0+timedelta(seconds=frame/fps)
    # merge_indexed.py offsets the track IDs of roi additions by (largest primary event ID + 1)
    primary=[int(e['track_id']) for e in csv.DictReader((a.run/'full'/'events.csv').open())]
    overlay=CountedOverlay(view,(OUT_W,OUT_H),a.directions,a.stride,roi_offset=max(primary,default=0)+1)

    events=sorted(csv.DictReader((a.run/'result'/'events.csv').open()),key=lambda e:(int(e['frame']),int(e['event_id'])))
    numbers={d:0 for d in a.directions};listed=[]
    for e in events:
        if e['direction'] not in numbers:continue
        numbers[e['direction']]+=1;listed.append((e,numbers[e['direction']]))
        overlay.add_event(dict(e,frame=int(e['frame']),track_id=int(e['track_id'])))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.stem+'_events.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['clock','direction','number','seconds','track_id','x','y'])
        writer.writerows([clock(int(e['frame'])).strftime('%H:%M:%S'),e['direction'],n,e['seconds'],e['track_id'],e['x'],e['y']] for e,n in listed)

    start_frame=round(a.start*fps/a.stride)*a.stride
    end_frame=None if a.end is None else round(a.end*fps)
    overlay.skip_until(start_frame)
    if a.count_from_start:overlay.counts={d:0 for d in overlay.counts}
    decode=['ffmpeg','-hide_banner','-v','error','-nostdin','-threads','0']
    if start_frame:decode+=['-ss',f'{start_frame/fps:.3f}']
    decode+=['-i',str(a.source),'-map','0:v:0','-vf',f'select=not(mod(n\\,{a.stride})),scale={OUT_W}:{OUT_H}:flags=area',
             '-fps_mode','passthrough','-pix_fmt','bgr24','-f','rawvideo','pipe:1']
    encode=['ffmpeg','-hide_banner','-v','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{OUT_W}x{OUT_H}',
            '-framerate',str(fps/a.stride),'-i','pipe:0','-c:v','h264_nvenc','-preset','p5','-rc','vbr','-cq','23','-b:v','0',
            '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(a.output)]
    decoder=subprocess.Popen(decode,stdout=subprocess.PIPE,bufsize=0)
    encoder=subprocess.Popen(encode,stdin=subprocess.PIPE)
    frames_out=queue.Queue(8);errors=[]
    def write():
        try:
            while (image:=frames_out.get()) is not None:encoder.stdin.write(image)
        except BaseException as exc:errors.append(exc)
        finally:encoder.stdin.close()
    writer_thread=threading.Thread(target=write,daemon=True);writer_thread.start()

    frame_bytes=OUT_W*OUT_H*3;rendered=0;started=time.perf_counter()
    try:
        with (a.run/'result'/'tracks.jsonl').open() as tracks:
            for row in tracks:
                record=json.loads(row);frame=record['frame']
                if frame<start_frame:continue
                if end_frame is not None and frame>=end_frame:break
                expected=start_frame+rendered*a.stride
                if frame!=expected:raise RuntimeError(f'Track frame {frame} does not match decoded frame {expected}')
                raw=read_exact(decoder.stdout,frame_bytes)
                if raw is None or len(raw)!=frame_bytes:raise RuntimeError(f'Video ended at frame {frame}')
                image=np.frombuffer(raw,np.uint8).reshape(OUT_H,OUT_W,3).copy()
                overlay.draw(image,frame,record['people'],clock(frame))
                while True:
                    if errors:raise errors[0]
                    try:frames_out.put(image.tobytes(),timeout=.2);break
                    except queue.Full:pass
                rendered+=1
                if rendered%1500==0:
                    print(json.dumps(dict(clock=f'{clock(frame):%H:%M:%S}',frames=rendered,fps=round(rendered/(time.perf_counter()-started),1),
                                          **overlay.counts)),flush=True)
    finally:
        frames_out.put(None);writer_thread.join();decoder.kill();decoder.wait()
    if errors:raise errors[0]
    if encoder.wait():raise RuntimeError('Encoder failed')
    info=dict(output=str(a.output),frames=rendered,fps=fps/a.stride,start_frame=start_frame,clock_start=a.clock_start,
              directions=a.directions,**overlay.counts,events_listed=len(listed),render_seconds=round(time.perf_counter()-started,1))
    a.output.with_suffix('.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n');print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
