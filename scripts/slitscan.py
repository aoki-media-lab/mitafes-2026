"""Slit-scan review image for counting-line validation.

For every kept frame (stride 2), copies the pixel columns at line-d, line, line+d.
Time runs left to right (2 px per observation). Each row covers --row-seconds.
Within a row: top strip = left slit, middle = line, bottom = right slit, so a person
walking left-to-right appears in the top strip first. Event ticks from an events.csv
(blue = ltr, orange = rtl) are drawn on the middle strip with their event id.
"""
import argparse,csv
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--line-x',type=float,required=True)
p.add_argument('--offset',type=float,default=.03);p.add_argument('--events');p.add_argument('--out',type=Path,required=True)
p.add_argument('--row-seconds',type=float,default=20);p.add_argument('--scale',type=float,default=.2)
p.add_argument('--start',type=float,default=0);p.add_argument('--seconds',type=float,default=0,help='Only this part of the video (0: to the end)')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(a.video);W=int(cap.get(3));H=int(cap.get(4));fps=cap.get(5)
xs=[int((a.line_x+o)*W) for o in (-a.offset,0,a.offset)];h=int(H*a.scale)
i=int(round(a.start*fps));i-=i%2;first=i;cap.set(cv2.CAP_PROP_POS_FRAMES,i);last=i+int(a.seconds*fps) if a.seconds else 1<<60
cols=[[],[],[]]
while i<last:
    ok=cap.grab()
    if not ok:break
    if i%2==0:
        _,f=cap.retrieve()
        for k,x in enumerate(xs):cols[k].append(cv2.resize(f[:,x-1:x+1],(2,h),interpolation=cv2.INTER_AREA))
    i+=1
strips=[np.hstack(c) for c in cols];n=strips[0].shape[1]//2
ev=list(csv.DictReader(open(a.events))) if a.events else []
per_row=int(a.row_seconds*fps/2)
for r in range(0,n,per_row):
    parts=[s[:,2*r:2*(r+per_row)] for s in strips];wpx=parts[0].shape[1]
    canvas=np.zeros((3*h+40+12,2*per_row,3),np.uint8);y=0
    for k,part in enumerate(parts):
        canvas[y:y+h,:wpx]=part;y+=h+(6 if k<2 else 0)
    for e in ev:
        o=(int(e['frame'])-first)//2
        if r<=o<r+per_row:
            x=2*(o-r);c=(255,160,0) if e['direction']=='ltr' else (0,140,255)
            cv2.line(canvas,(x,h+6),(x,2*h+6),c,2);cv2.putText(canvas,e['event_id'],(x-8,3*h+40),0,.6,c,2)
    for t in range(0,int(a.row_seconds)+1,5):
        x=int(t*fps);cv2.putText(canvas,f'{first/fps+r*2/fps+t:.0f}s',(x,3*h+26),0,.5,(255,255,255),1)
    cv2.imwrite(str(a.out/f'slit_{int(first/fps+r*2/fps):04d}s.jpg'),canvas)
print(n,'observations')
