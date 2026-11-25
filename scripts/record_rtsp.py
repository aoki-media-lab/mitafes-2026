"""Record the RTSP camera to an mp4 (mp4v re-encode, all frames) for offline tuning.
URL from ~/.mitafes_rtsp; never printed. Reports dropped frames if encoding falls behind."""
import argparse,json,os,queue,threading,time
from pathlib import Path
import cv2
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=210);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp|stimeout;5000000'
cap=cv2.VideoCapture((Path.home()/'.mitafes_rtsp').read_text().strip(),cv2.CAP_FFMPEG)
if not cap.isOpened():raise SystemExit('cannot open stream')
size=(int(cap.get(3)),int(cap.get(4)))
w=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),25,size)
q=queue.Queue(maxsize=100);stats=dict(read=0,written=0,dropped=0)
def writer():
    while (f:=q.get()) is not None:w.write(f);stats['written']+=1
t=threading.Thread(target=writer);t.start();start=time.time()
while time.time()-start<a.seconds:
    ok,f=cap.read()
    if not ok:break
    stats['read']+=1
    try:q.put_nowait(f)
    except queue.Full:stats['dropped']+=1
q.put(None);t.join();w.release()
stats.update(start=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start)),size=size)
print(json.dumps(stats))
