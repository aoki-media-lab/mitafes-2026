"""Compare the merged events of two runs of the same video and render review strips of the differences.

Events are matched one-to-one by direction, |time| <= --tol seconds and |y| <= --ytol px (greedy, nearest
in time first). Every event only one run has gets a row of crops around the count line at -1.2..+1.2 s,
with the tracked people of run A (green boxes, head dots) and run B (magenta) at those observations, the
count line (red) and the event's height (yellow tick), so it can be judged by eye whether someone crossed.
Output: <out>/diff.json, <out>/sheet_###.jpg (rows labelled 'A-only'/'B-only', event time, direction, rule).
"""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np

FPS=25.0;W,H=2880,1620
OFFSETS=[-40,-20,0,20,40]          # camera frames around the event (stride 2 -> even)
CW,CH=1000,760                     # crop around (line, event y), full-resolution pixels


def load(run):
    with (Path(run)/'result'/'events.csv').open() as f:return list(csv.DictReader(f))


def match(a,b,tol,ytol):
    pairs=[];used=set()
    cand=sorted(((abs(float(x['seconds'])-float(y['seconds'])),i,j) for i,x in enumerate(a) for j,y in enumerate(b)
                 if x['direction']==y['direction'] and abs(float(x['seconds'])-float(y['seconds']))<=tol and abs(float(x['y'])-float(y['y']))<=ytol))
    ua=set()
    for _,i,j in cand:
        if i in ua or j in used:continue
        ua.add(i);used.add(j);pairs.append((i,j))
    return pairs,[x for i,x in enumerate(a) if i not in ua],[y for j,y in enumerate(b) if j not in used]


def tracks_at(run,frames):
    out={};need=set(frames)
    with (Path(run)/'full'/'tracks.jsonl').open() as f:
        for line in f:
            fr=int(line[10:line.index(',')])
            if fr in need:out[fr]=json.loads(line)['people']
            if fr>max(need):break
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--video',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tol',type=float,default=1.5);p.add_argument('--ytol',type=float,default=200)
    p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=1e9)
    p.add_argument('--line-x',type=float,default=.5);p.add_argument('--direction',choices=['ltr','rtl'],help='Only events of this direction')
    p.add_argument('--scale',type=float,default=.32);p.add_argument('--rows',type=int,default=4);p.add_argument('--no-images',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    win=lambda ev:[e for e in ev if a.start<=float(e['seconds'])<a.end and (a.direction is None or e['direction']==a.direction)]
    A,B=win(load(a.a)),win(load(a.b))
    pairs,only_a,only_b=match(A,B,a.tol,a.ytol)
    diff=[dict(e,which='A-only') for e in only_a]+[dict(e,which='B-only') for e in only_b]
    diff.sort(key=lambda e:float(e['seconds']))
    count=lambda ev,d:sum(e['direction']==d for e in ev)
    summary=dict(a=a.a,b=a.b,window=[a.start,a.end],a_counts=dict(rtl=count(A,'rtl'),ltr=count(A,'ltr')),b_counts=dict(rtl=count(B,'rtl'),ltr=count(B,'ltr')),
                 matched=len(pairs),a_only=len(only_a),b_only=len(only_b),diff=diff)
    (a.out/'diff.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='diff'}))
    if a.no_images or not diff:return
    frames=sorted({f for e in diff for f in range(int(e['frame'])-60,int(e['frame'])+62,2)})
    ta,tb=tracks_at(a.a,frames),tracks_at(a.b,frames)
    cap=cv2.VideoCapture(a.video);s=a.scale;lx=a.line_x*W
    def trail(tracks,tid,f0):
        return [q['anchor'] for f in range(f0-60,f0+62,2) for q in tracks.get(f,[]) if q['id']==tid and 'anchor' in q]
    # One sequential decode pass (seeking HEVC per tile is far slower): keep only the crops needed.
    want={}
    for k,e in enumerate(diff):
        f0=int(e['frame']);f0-=f0%2;y0=float(e['y'])
        box=(int(min(max(lx-CW/2,0),W-CW)),int(min(max(y0-CH/2,0),H-CH)))
        for o in OFFSETS:want.setdefault(f0+o,[]).append(box)
    crops={};f=min(want);cap.set(cv2.CAP_PROP_POS_FRAMES,f);last=max(want)
    while f<=last:
        if f in want:
            ok,img=cap.read()
            if not ok:break
            for x1,y1 in want[f]:crops[(f,x1,y1)]=img[y1:y1+CH,x1:x1+CW].copy()
        elif not cap.grab():break
        f+=1
    rows=[]
    for k,e in enumerate(diff):
        f0=int(e['frame']);f0-=f0%2;y0=float(e['y'])
        x1=int(min(max(lx-CW/2,0),W-CW));y1=int(min(max(y0-CH/2,0),H-CH))
        mine=ta if e['which']=='A-only' else tb
        path=trail(mine,int(e['track_id']),f0)
        # 'Twins': other events of the same run within 1 s / 80 px (companion or duplicate track?), in cyan.
        same=A if e['which']=='A-only' else B
        twins=[trail(mine,int(t['track_id']),f0) for t in same if t['event_id']!=e['event_id'] and t['direction']==e['direction']
               and abs(float(t['seconds'])-float(e['seconds']))<=1 and abs(float(t['y'])-y0)<80]
        tiles=[]
        for o in OFFSETS:
            f=f0+o;img=crops.get((f,x1,y1));img=img.copy() if img is not None else np.zeros((CH,CW,3),np.uint8)
            for people,color,t in [(ta.get(f,[]),(0,220,0),9),(tb.get(f,[]),(255,0,255),3)]:
                for q in people:
                    b=q['xyxy'];cv2.rectangle(img,(int(b[0]-x1),int(b[1]-y1)),(int(b[2]-x1),int(b[3]-y1)),color,t)
                    if 'anchor' in q:cv2.circle(img,(int(q['anchor'][0]-x1),int(q['anchor'][1]-y1)),8 if t>5 else 5,color,-1)
            cv2.line(img,(int(lx-x1),0),(int(lx-x1),CH),(0,0,255),2)
            for tw in twins:
                if len(tw)>1:cv2.polylines(img,[(np.asarray(tw)-[x1,y1]).astype(np.int32)],False,(255,255,0),4)
            if len(path)>1:cv2.polylines(img,[(np.asarray(path)-[x1,y1]).astype(np.int32)],False,(0,255,255),4)
            if o==0:cv2.line(img,(int(lx-x1)-40,int(y0-y1)),(int(lx-x1)+40,int(y0-y1)),(0,255,255),6)
            t=cv2.resize(img,(int(CW*s),int(CH*s)),interpolation=cv2.INTER_AREA)
            cv2.putText(t,f'{o/FPS:+.1f}s',(6,int(CH*s)-10),0,.7,(255,255,255),2)
            tiles.append(t)
        row=np.hstack(tiles);bar=np.full((34,row.shape[1],3),30,np.uint8)
        m,sec=divmod(float(e['seconds']),60)
        cv2.putText(bar,f"#{k} {e['which']}  t={int(m)}:{sec:05.2f}  {e['direction']}  {e.get('rule','') or 'plain'}  y={y0:.0f}  track={'found' if path else 'roi-branch'}  twins={len(twins)}",(8,24),0,.8,(255,255,255),2)
        rows.append(np.vstack([bar,row]))
    for i in range(0,len(rows),a.rows):
        cv2.imwrite(str(a.out/f'sheet_{i//a.rows:03d}.jpg'),np.vstack(rows[i:i+a.rows]),[cv2.IMWRITE_JPEG_QUALITY,82])


if __name__=='__main__':main()
