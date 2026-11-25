"""Contact sheets for checking count events by eye.

For each listed event: frames at -1.2 s, -0.4 s, +0.4 s, +1.2 s around the event, cropped to
the counting area, with the line drawn and the event's track marked. Several events per sheet.
Input is a JSON list of {"id", "frame", "x", "y", "direction", "label"} (frames at the recording's fps).
"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from views import load_view,counting_line

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--video',type=Path,required=True);p.add_argument('--events',type=Path,required=True)
p.add_argument('--view',default='plaza');p.add_argument('--output',type=Path,required=True)
p.add_argument('--per-sheet',type=int,default=6);p.add_argument('--offsets',type=float,nargs='+',default=[-1.2,-.4,.4,1.2])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
view=load_view(a.view);L=counting_line(view);fps=view['fps']
events=json.loads(a.events.read_text())
cap=cv2.VideoCapture(str(a.video))
CROP=(400,0,2500,1620);TW=480  # x1,y1,x2,y2 of the counting area; tile width
def tile(frame_no,e):
    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,frame_no));ok,im=cap.read()
    if not ok:return np.zeros((round(TW*1620/2100),TW,3),np.uint8)
    p1,p2=L.pixels();cv2.line(im,p1,p2,(0,255,255),3)
    cv2.circle(im,(int(e['x']),int(e['y'])),40,(0,0,255),4)
    im=im[CROP[1]:CROP[3],CROP[0]:CROP[2]];im=cv2.resize(im,(TW,round(TW*im.shape[0]/im.shape[1])))
    cv2.putText(im,f"{(frame_no-e['frame'])/fps:+.1f}s",(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    return im
for s in range(0,len(events),a.per_sheet):
    rows=[]
    for e in events[s:s+a.per_sheet]:
        tiles=[tile(e['frame']+round(o*fps),e) for o in a.offsets]
        row=np.hstack(tiles);bar=np.zeros((26,row.shape[1],3),np.uint8)
        cv2.putText(bar,f"#{e['id']} {e['direction']} {e['frame']/fps:.1f}s {e.get('label','')}",(5,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        rows.append(np.vstack([bar,row]))
    cv2.imwrite(str(a.output/f'sheet_{s//a.per_sheet:03d}.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,80])
print('sheets',(len(events)+a.per_sheet-1)//a.per_sheet)
