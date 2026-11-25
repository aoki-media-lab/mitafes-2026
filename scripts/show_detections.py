"""Draw every dumped person detection of chosen observations, coloured by confidence, to see what a
lower detector/track threshold would add (green >= 0.3, yellow 0.1-0.3, red below 0.1; thin = roi branch).
Usage: show_detections.py --video V --run RUN (made with --dump-detections) --frames F ... --out DIR"""
import argparse,json
from pathlib import Path
import cv2
p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--run',type=Path,required=True)
p.add_argument('--frames',type=int,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,default=1440)
p.add_argument('--line-x',type=float,default=.5)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);need=set(a.frames);dets={}
for name in ['full','roi']:
    with (a.run/name/'detections.jsonl').open() as f:
        for line in f:
            i=int(line[10:line.index(',')])
            if i in need:dets.setdefault(i,{})[name]=json.loads(line)
            if i>max(need):break
cap=cv2.VideoCapture(a.video)
for fr in a.frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,img=cap.read()
    if not ok:continue
    H,W=img.shape[:2];x=int(a.line_x*W);cv2.line(img,(x,0),(x,H),(0,0,255),2)
    for name,t in [('full',5),('roi',2)]:
        for d,k in zip(dets.get(fr,{}).get(name,{}).get('detections',[]),dets.get(fr,{}).get(name,{}).get('keypoints',[])):
            if int(d[5])!=0:continue
            c=(0,220,0) if d[4]>=.3 else (0,220,255) if d[4]>=.1 else (0,0,255)
            cv2.rectangle(img,(int(d[0]),int(d[1])),(int(d[2]),int(d[3])),c,t)
            if name=='full':cv2.putText(img,f'{d[4]:.2f}',(int(d[0]),int(d[1])-6),0,1.1,c,3)
    cv2.imwrite(str(a.out/f'det_{fr}.jpg'),cv2.resize(img,(a.width,round(H*a.width/W))),[cv2.IMWRITE_JPEG_QUALITY,85])
