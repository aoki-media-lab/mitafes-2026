"""Render chosen frames of a recording with the run's full-branch tracks, count line and ROI.
Usage: review_frames.py --video V --run RUN_DIR --setup SETUP --frames 760 776 ... --out DIR"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_fused import load_setup,branch_specs,FULL_PROFILE
from live_web import geometry,color_for
p=argparse.ArgumentParser();p.add_argument('--video');p.add_argument('--run',type=Path);p.add_argument('--setup')
p.add_argument('--frames',type=int,nargs='+');p.add_argument('--out',type=Path);p.add_argument('--branch',default='full')
p.add_argument('--trail',type=int,default=30)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
prof=json.loads((a.run/a.branch/'summary.json').read_text())['profile']
rows={};want=set(a.frames);lo=min(want)-2*a.trail
for line in (a.run/a.branch/'tracks.jsonl').open():
    r=json.loads(line)
    if lo<=r['frame']<=max(want):rows[r['frame']]=r
events={}
import csv
for e in csv.DictReader((a.run/a.branch/'events.csv').open()):events.setdefault(int(e['track_id']),[]).append(e)
cap=cv2.VideoCapture(a.video)
for f in sorted(want):
    cap.set(1,f);ok,img=cap.read()
    for pts,kind in geometry(prof,img.shape[1],img.shape[0]):
        cv2.polylines(img,[pts.astype(np.int32)],kind=='roi',{'line':(0,0,255),'margin':(0,200,255),'roi':(0,255,0)}[kind],3)
    for person in rows.get(f,{}).get('people',[]):
        tid=person['id'];c=color_for(tid)
        trail=[rows[g]['people'] for g in range(f-2*a.trail,f+1,2) if g in rows]
        pts=[q['anchor'] for ps in trail for q in ps if q['id']==tid and 'anchor' in q]
        if len(pts)>1:cv2.polylines(img,[np.int32(pts)],False,c,4)
        x1,y1,x2,y2=map(int,person['xyxy']);cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
        ev=','.join(f"{e['direction']}@{e['frame']}" for e in events.get(tid,[]))
        x,y=map(int,person['anchor']);cv2.circle(img,(x,y),9,c,-1)
        cv2.putText(img,f'{tid} {ev}',(x+10,y-10),0,1.4,(0,0,0),6);cv2.putText(img,f'{tid} {ev}',(x+10,y-10),0,1.4,c,2)
    cv2.putText(img,f'frame {f} ({f/25:.2f}s)',(40,1560),0,2,(0,0,255),4)
    cv2.imwrite(str(a.out/f'f{f:05d}.jpg'),cv2.resize(img,(1440,810)))
