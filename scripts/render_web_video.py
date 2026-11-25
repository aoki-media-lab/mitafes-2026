"""Render a recording with the browser view's overlays (trails, count flashes, line, ROI) at full
frame rate, from a run_fused.py / replay_fused.py output, for viewing offline.

The corner shows the counts since --start (in = the setup's in_direction, right-to-left by default) and an
optional --label. Output is H.264 MP4 (NVENC, else libx264) written through ffmpeg.
Usage: render_web_video.py --video V --run RUN --out OUT.mp4 [--width 1920] [--start S --seconds N] [--label TEXT]"""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from live_web import LiveWeb
p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--run',type=Path,required=True)
p.add_argument('--out',required=True);p.add_argument('--width',type=int,default=1920)
p.add_argument('--start',type=float,default=0);p.add_argument('--seconds',type=float,default=0)
p.add_argument('--warmup',type=float,default=3,help='Seconds tracked before --start without writing, so trails are already drawn')
p.add_argument('--label',default='',help='Text drawn under the counts (e.g. the settings)')
p.add_argument('--clock0',type=float,help='Epoch time of video t=0: draws the wall-clock time in the corner')
a=p.parse_args()
full_prof=json.loads((a.run/'full'/'summary.json').read_text())['profile']
summary=json.loads((a.run/'summary.json').read_text())
setup=summary.get('setup') if isinstance(summary.get('setup'),dict) else {}
roi_crop=setup.get('roi_crop') or [0.35,0,0.75,1]
cap=cv2.VideoCapture(a.video);W,H,fps=int(cap.get(3)),int(cap.get(4)),cap.get(5)
crop=[v*m for v,m in zip(roi_crop,[W,H,W,H])]                  # LiveWeb takes the crop in pixels
web=LiveWeb('127.0.0.1',0,W,H,full_prof,crop,a.width,90)
web.close()                                                  # only the drawing code is used
first=int(a.start*fps);last=int((a.start+a.seconds)*fps) if a.seconds else int(cap.get(7))
begin=max(0,first-int(a.warmup*fps))
rows=[{},{}]
for b,name in enumerate(['full','roi']):
    with (a.run/name/'tracks.jsonl').open() as f:
        for line in f:
            fr=int(line[10:line.index(',')])
            if fr<begin:continue
            if fr>last:break
            rows[b][fr]=json.loads(line)
# Counts as the merged result has them; roi additions shown at their own frame.
events={}
for e in csv.DictReader((a.run/'full'/'events.csv').open()):events.setdefault(int(e['frame']),[]).append((int(e['track_id']),e['direction'],'full',float(e['y']),int(e['frame'])))
for e in summary.get('added') or json.loads((a.run/'result'/'summary.json').read_text())['added']:
    events.setdefault(int(e['frame']),[]).append((int(e['track_id']),e['direction'],'roi_added',float(e['y']),int(e['frame'])))
# Times in the 'recent counts' list: --clock0, else the start time in an NVR file name (ch1_YYYYmmdd_HHMMSS-...).
import re,time
m=re.search(r'_(\d{8})_(\d{6})',Path(a.video).name)
t0=a.clock0 if a.clock0 is not None else (time.mktime(time.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S')) if m else 0)
web.clock_at=lambda index:t0+index/fps
size=(a.width,round(H*a.width/W))
enc=['-c:v','h264_nvenc','-preset','p5','-cq','23'] if subprocess.run(['ffmpeg','-hide_banner','-encoders'],capture_output=True,text=True).stdout.find('h264_nvenc')>=0 else ['-c:v','libx264','-preset','veryfast','-crf','22']
ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','pipe:0',
                     *enc,'-pix_fmt','yuv420p','-movflags','+faststart',a.out],stdin=subprocess.PIPE)
in_dir=web.in_dir;out_dir=next((d for d in web.directions if d!=in_dir),'ltr' if in_dir=='rtl' else 'rtl')
color={'ltr':(247,195,79),'rtl':(77,183,255)};count={'ltr':0,'rtl':0}
empty=dict(people=[]);recs=(empty,empty);people=[];written=0
cap.set(cv2.CAP_PROP_POS_FRAMES,begin)
def box(img,text,y,c,scale=1.1):
    (tw,th),_=cv2.getTextSize(text,0,scale,2)
    cv2.rectangle(img,(size[0]-tw-30,y),(size[0]-10,y+th+22),(20,20,20),-1)
    cv2.putText(img,text,(size[0]-tw-20,y+th+11),0,scale,c,2,cv2.LINE_AA);return y+th+26
for f in range(begin,last):
    write=f>=first
    ok,img=cap.read() if write else (cap.grab(),None)
    if not ok:break
    if f in rows[0]:
        recs=(rows[0][f],rows[1].get(f,empty));ev=events.get(f,[])
        # stitch / late_death counts carry a frame in the past: show them when they are decided
        web.update(f,recs,ev)
        if f>=first:
            for e in ev:count[e[1]]+=1
        people=[q['xyxy'] for q in recs[0]['people']]
    if not write:continue
    frame=web.render(f,img,people,[],recs)
    y=box(frame,f'IN (right->left)  {count[in_dir]}' if in_dir=='rtl' else f'IN  {count[in_dir]}',10,color[in_dir])
    y=box(frame,f'OUT (left->right)  {count[out_dir]}' if out_dir=='ltr' else f'OUT  {count[out_dir]}',y,color[out_dir])
    if a.label:                                   # bottom right, away from the people's labels
        (tw,th),_=cv2.getTextSize(a.label,0,.7,2);yb=size[1]-th-32
        cv2.rectangle(frame,(size[0]-tw-30,yb),(size[0]-10,yb+th+22),(20,20,20),-1)
        cv2.putText(frame,a.label,(size[0]-tw-20,yb+th+11),0,.7,(230,230,230),2,cv2.LINE_AA)
    if a.clock0 is not None:
        import time;c=time.strftime('%H:%M:%S',time.localtime(a.clock0+f/fps))
        cv2.rectangle(frame,(10,size[1]-48),(170,size[1]-10),(20,20,20),-1);cv2.putText(frame,c,(20,size[1]-20),0,.9,(255,255,255),2,cv2.LINE_AA)
    ff.stdin.write(frame.tobytes());written+=1
ff.stdin.close();ff.wait()
print(json.dumps(dict(out=a.out,frames=written,seconds=round(written/fps,1),counts=count)))
