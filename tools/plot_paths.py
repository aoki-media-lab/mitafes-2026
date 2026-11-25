"""Draw every tracked person path from tracks.jsonl over a background frame.

Used to choose the counting line for a new camera view. Colour shows the direction of
motion (start = blue dot, end = red dot); the view's line, margin band and ROI are drawn.
"""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from views import load_view,counting_line

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tracks',type=Path,required=True);p.add_argument('--background',type=Path,required=True)
p.add_argument('--view',default=None);p.add_argument('--output',type=Path,required=True)
p.add_argument('--min-points',type=int,default=6);p.add_argument('--events',type=Path)
a=p.parse_args()
paths=defaultdict(list)
for line in a.tracks.open():
    r=json.loads(line)
    for q in r['people']:paths[q['id']].append(q['anchor'])
img=cv2.imread(str(a.background));h,w=img.shape[:2];s=w/2880
for tid,pts in paths.items():
    if len(pts)<a.min_points:continue
    pts=np.asarray(pts)*s;dx=pts[-1,0]-pts[0,0]
    color=(0,200,255) if dx<0 else (255,160,0)    # orange: moved left, blue: moved right
    cv2.polylines(img,[pts.astype(np.int32)],False,color,1,cv2.LINE_AA)
    cv2.circle(img,tuple(pts[0].astype(int)),3,(255,0,0),-1);cv2.circle(img,tuple(pts[-1].astype(int)),3,(0,0,255),-1)
if a.view:
    view=load_view(a.view);prof=view['profile'];L=counting_line(view)
    if prof.get('roi_mode')=='polygon':
        poly=(np.asarray(prof['roi_polygon'])*[w,h]).astype(np.int32);cv2.polylines(img,[poly],True,(0,255,0),2)
    p1,p2=[np.asarray(x)*s for x in L.pixels()]
    n=np.asarray(L.normal)*prof['margin']*2880*s
    cv2.line(img,tuple(p1.astype(int)),tuple(p2.astype(int)),(0,255,255),2)
    for k in (-1,1):cv2.line(img,tuple((p1+k*n).astype(int)),tuple((p2+k*n).astype(int)),(128,128,128),1)
if a.events:
    for l in a.events.open():
        e=json.loads(l);cv2.circle(img,(int(e['x']*s),int(e['y']*s)),5,(0,0,255) if e['direction']=='rtl' else (255,0,255),2)
cv2.imwrite(str(a.output),img);print(len(paths),'tracks')
