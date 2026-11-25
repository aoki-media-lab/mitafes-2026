"""Visualization overlay in the style of the main-gate (aolab-1) live view; shared by
tools/render_counted.py and the live recorder.

* every tracked person: thin white box, head dot and fading trail in a colour of its own, and the
  track ID next to the head
* a count: the trail turns the count colour (orange = right->left, light blue = left->right) and a
  boxed label "<- #ID" / "-> #ID" (the arrow is the direction) shows for ~1.5 s; afterwards the
  label reads "ID ok"
* a count added from the zoomed (roi) branch, i.e. a person the full-frame tracking missed: a circle
  where its trajectory meets the count line, a boxed "+1 roi <- #ID", the trajectory drawn thick
  and the person's box outlined
* top right: IN / OUT totals (or right->left / left->right); bottom right: the latest counts
  (time, direction, #ID); bottom left: the clock
* count line red, margin orange, foot area green, reconciliation crop blue

Events may be added in any order (the live reconciliation confirms some ~7 s late); they take
effect when a frame at or after their frame is drawn. Track IDs of roi additions are offset by
``roi_offset`` in the merged data; they are shown with their own roi-branch ID.
"""
import heapq
from collections import deque
from datetime import datetime
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from line_counter import CountingLine

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
COLORS={'rtl':(77,183,255),'ltr':(247,195,79)}     # BGR: orange / light blue
NAMES={'rtl':'右→左','ltr':'左→右'}
ARROWS={'rtl':'<-','ltr':'->'}
FLASH=38            # source frames a fresh count stays highlighted (~1.5 s at 25 fps)
SMOOTH=4            # display-only moving average of head trails (+-4 observations)


def smooth(pts,k=SMOOTH):
    n=len(pts)
    if n<3:return pts
    c=np.vstack([np.zeros((1,2)),np.cumsum(pts,0)]);i=np.arange(n)
    lo=np.maximum(0,i-k);hi=np.minimum(n,i+k+1)
    return (c[hi]-c[lo])/(hi-lo)[:,None]


def color_for(tid):
    h=(int(tid)*47)%180
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h,200,255]]]),cv2.COLOR_HSV2BGR)[0,0])


def place_label(img,text,at,color,placed,scale=.75):
    """Boxed text next to ``at``, moved to the first nearby spot not overlapping an earlier label,
    so labels of people walking together stay readable; a thin line points to the person."""
    (tw,th),_=cv2.getTextSize(text,0,scale,2);x0,y0=at;H,W=img.shape[:2]
    for dx,dy in [(14,-14),(14,th+18),(14,-th-36),(14,2*th+40),(-tw-24,-14),(-tw-24,th+18),(14,-2*th-58),(14,3*th+62)]:
        x=int(min(max(x0+dx,0),W-tw-8));y=int(min(max(y0+dy,th+6),H-6))
        r=(x-4,y-th-6,x+tw+4,y+6)
        if not any(r[0]<q[2] and q[0]<r[2] and r[1]<q[3] and q[1]<r[3] for q in placed):break
    placed.append(r)
    cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]),(25,25,25),-1);cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]),color,1)
    cv2.putText(img,text,(x,y),0,scale,color,2,cv2.LINE_AA)
    cv2.line(img,(int(x0),int(y0)),(r[0] if r[0]>x0 else r[2],(r[1]+r[3])//2),color,1,cv2.LINE_AA)


def corner_box(img,text,y,color,scale=1.1):
    W=img.shape[1];(tw,th),_=cv2.getTextSize(text,0,scale,2)
    cv2.rectangle(img,(W-tw-30,y),(W-10,y+th+22),(20,20,20),-1)
    cv2.putText(img,text,(W-tw-20,y+th+11),0,scale,color,2,cv2.LINE_AA);return y+th+26


class CountedOverlay:
    def __init__(self,view,size,directions=('rtl','ltr'),stride=2,base=None,roi_offset=None):
        profile=view['profile'];W,H=view['width'],view['height']
        self.fps=float(view['fps']);self.stride=stride;self.size=size;self.s=size[0]/W
        self.directions=tuple(directions);self.roi_offset=roi_offset
        self.cline=CountingLine(profile,W,H);margin=profile['margin']*W;nx,ny=self.cline.normal
        p1,p2=np.asarray(self.cline.p1),np.asarray(self.cline.p2);n=np.array([nx,ny])*margin
        to=lambda pts:(np.asarray(pts,np.float64)*self.s).astype(np.int32)
        self.lines=[(to([p1-n,p2-n]),(0,160,255),1,False),(to([p1+n,p2+n]),(0,160,255),1,False),(to([p1,p2]),(54,67,244),3,False)]
        if profile.get('roi_polygon'):self.lines.append((to([(x*W,y*H) for x,y in profile['roi_polygon']]),(71,160,67),2,True))
        crops=[b['crop'] for b in view.get('branches',[]) if b['name']!='full']
        self.crops=[((round(c[0]*size[0]),round(c[1]*size[1])),(round(c[2]*size[0])-1,round(c[3]*size[1])-1)) for c in crops]
        entry=view.get('entry_direction')
        if entry and len(self.directions)>1:
            out='ltr' if entry=='rtl' else 'rtl';arrow={'rtl':'right->left','ltr':'left->right'}
            self.corner=[(entry,f'IN ({arrow[entry]})'),(out,f'OUT ({arrow[out]})')]
        else:self.corner=[(d,'right->left' if d=='rtl' else 'left->right') for d in self.directions]
        self.pending=[];self.order=0
        self.counts={d:(base or {}).get(d,0) for d in self.directions}
        self.trails={};self.flash={};self.counted={};self.roi_trails={};self.roi_box={};self.roi_marks=[]
        self.recent=deque(maxlen=6);self.recent_img=None;self.font=None

    def is_roi(self,tid):return self.roi_offset is not None and int(tid)>=self.roi_offset

    def shown_id(self,tid):return int(tid)-self.roi_offset if self.is_roi(tid) else int(tid)

    def add_event(self,event):
        """event: dict with direction, frame, track_id, x, y (source pixels)."""
        if event['direction'] not in self.directions:return
        self.order+=1;heapq.heappush(self.pending,(int(event['frame']),self.order,event))

    def skip_until(self,frame):
        """Apply events before ``frame`` to the totals without drawing (preview windows)."""
        while self.pending and self.pending[0][0]<frame:
            _,_,e=heapq.heappop(self.pending);self.counts[e['direction']]+=1

    def draw(self,image,frame,people,wall):
        """Draw on ``image`` (already ``size``); ``wall`` is the frame's clock time (datetime)."""
        s=self.s
        while self.pending and self.pending[0][0]<=frame:
            f,_,e=heapq.heappop(self.pending);d=e['direction'];tid=int(e['track_id'])
            self.counts[d]+=1
            when=datetime.fromtimestamp(wall.timestamp()-(frame-f)/self.fps)
            self.recent.appendleft((when,d,self.shown_id(tid),self.is_roi(tid)));self.recent_img=None
            if self.is_roi(tid):self.roi_marks.append((tid,d,frame+FLASH,f))
            else:self.counted[tid]=d;self.flash[tid]=(d,frame+FLASH)
        alive=set()
        for p in people:
            if not p.get('anchor'):continue
            tid=int(p['id'])
            if self.is_roi(tid):
                self.roi_trails.setdefault(tid,deque(maxlen=150)).append((frame,p['anchor']));self.roi_box[tid]=(frame,p['xyxy'])
            else:alive.add(tid);self.trails.setdefault(tid,deque(maxlen=50)).append(p['anchor'])
        self.flash={k:v for k,v in self.flash.items() if v[1]>=frame}
        self.roi_marks=[m for m in self.roi_marks if m[2]>=frame]
        if frame%100<self.stride:
            keep=alive|set(self.flash)
            self.trails={k:v for k,v in self.trails.items() if k in keep}
            self.counted={k:v for k,v in self.counted.items() if k in keep}
            marked={m[0] for m in self.roi_marks}
            self.roi_trails={k:v for k,v in self.roi_trails.items() if v[-1][0]>=frame-150 or k in marked}
            self.roi_box={k:v for k,v in self.roi_box.items() if k in self.roi_trails}
        img=image
        for pts,c,t,closed in self.lines:cv2.polylines(img,[pts],closed,c,t,cv2.LINE_AA)
        for a,b in self.crops:cv2.rectangle(img,a,b,(229,136,30),1)
        for p in people:
            if self.is_roi(p['id']):continue
            b=p['xyxy'];cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),(255,255,255),1)
        # Trails: same width everywhere, strength by opacity -- older parts fade, ordinary tracks pale,
        # counted tracks keep the count colour, a just-counted track fully opaque.
        layers={};labels=[]
        for tid,trail in self.trails.items():
            hit=self.flash.get(tid);done=self.counted.get(tid)
            if tid not in alive and not hit:continue
            pts=smooth(np.asarray(trail,np.float64).reshape(-1,2))*s
            c=COLORS[(hit or (done,))[0]] if hit or done else color_for(tid)
            top=1. if hit else (.8 if done else .55);n=len(pts)
            for j in range(1,n):
                a=round(top*(.15+.85*j/(n-1))*5)/5 or .2
                layers.setdefault(a,[]).append(('line',(tuple(map(int,pts[j-1])),tuple(map(int,pts[j]))),c))
            x,y=map(int,pts[-1]);layers.setdefault(top,[]).append(('dot',(x,y),c))
            labels.append(((x+8,y-6),f"{ARROWS[hit[0]]} #{tid}" if hit else (f"{tid} ok" if done else str(tid)),c,hit))
        for a in sorted(layers):
            ink=np.zeros_like(img);mask=np.zeros(img.shape[:2],np.uint8)
            for kind,g,c in layers[a]:
                if kind=='line':cv2.line(ink,*g,c,3,cv2.LINE_AA);cv2.line(mask,*g,255,3,cv2.LINE_AA)
                else:cv2.circle(ink,g,5,c,-1,cv2.LINE_AA);cv2.circle(mask,g,5,255,-1,cv2.LINE_AA)
            x,y,w,h=cv2.boundingRect(mask)
            if not w:continue
            m=mask[y:y+h,x:x+w].astype(np.float32)*(a/255)
            img[y:y+h,x:x+w]=cv2.blendLinear(img[y:y+h,x:x+w],ink[y:y+h,x:x+w],1-m,m)
        placed=[]
        for tid,d,_,f in self.roi_marks:
            c=COLORS[d]
            near=[q for g,q in self.roi_trails.get(tid,()) if abs(g-f)<=50]   # the passage itself (+-2 s)
            src=np.asarray(near,np.float64).reshape(-1,2)
            pts=(smooth(src)*s).astype(np.int32) if len(src) else np.zeros((0,2),np.int32)
            if len(pts)>1:
                cv2.polylines(img,[pts],False,(0,0,0),8,cv2.LINE_AA);cv2.polylines(img,[pts],False,c,4,cv2.LINE_AA)
            if len(pts):   # where the trajectory meets the count line
                k=int(np.argmin([abs(self.cline.offset(*q)) for q in smooth(src)]));x,y=map(int,pts[k])
            else:x,y=int(self.cline.p1[0]*s),int(self.cline.p1[1]*s)
            cv2.circle(img,(x,y),12,(0,0,0),6,cv2.LINE_AA);cv2.circle(img,(x,y),12,c,3,cv2.LINE_AA)
            place_label(img,f"+1 roi {ARROWS[d]} #{self.shown_id(tid)}",(x,y),c,placed)
            seen,b=self.roi_box.get(tid,(None,None))
            if seen is not None and frame-seen<=2*self.stride:
                cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),(0,0,0),6)
                cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),c,3)
        if self.recent:self.draw_recent(img)
        for org,text,c,hit in labels:
            if hit:place_label(img,text,(org[0]-8,org[1]+6),c,placed);continue
            cv2.putText(img,text,org,0,.5,(0,0,0),3,cv2.LINE_AA);cv2.putText(img,text,org,0,.5,(200,200,200),1,cv2.LINE_AA)
        y=10
        for d,label in self.corner:y=corner_box(img,f'{label}  {self.counts[d]}',y,COLORS[d])
        H=img.shape[0]
        cv2.rectangle(img,(10,H-48),(170,H-10),(20,20,20),-1)
        cv2.putText(img,f'{wall:%H:%M:%S}',(20,H-20),0,.9,(255,255,255),2,cv2.LINE_AA)
        return img

    def draw_recent(self,img):
        """'直近のカウント' (time, direction, #ID), newest first, outlined text, bottom right."""
        if self.recent_img is None:
            size=max(14,round(self.size[0]/1280*19))
            if self.font is None:
                try:self.font=ImageFont.truetype(FONT,size,index=0)
                except OSError:self.font=ImageFont.load_default()
            rows=[('直近のカウント',(230,230,230))]+[(f"{t:%H:%M:%S} {NAMES[d]} #{tid}"+(' roi' if roi else ''),COLORS[d][::-1])
                                                    for t,d,tid,roi in list(self.recent)[:6]]
            w=max(self.font.getbbox(t)[2] for t,_ in rows)+12;h=len(rows)*(size+6)+10
            canvas=Image.new('RGBA',(w,h),(0,0,0,0));draw=ImageDraw.Draw(canvas)
            for i,(t,c) in enumerate(rows):draw.text((6,4+i*(size+6)),t,font=self.font,fill=c+(255,),stroke_width=3,stroke_fill=(0,0,0,255))
            a=np.asarray(canvas);self.recent_img=(a[...,2::-1].astype(np.float32),a[...,3:].astype(np.float32)/255)
        rgb,alpha=self.recent_img;h,w=alpha.shape[:2];H,W=img.shape[:2]
        y0,x0=H-h-12,W-w-12
        if y0<0 or x0<0:return
        patch=img[y0:y0+h,x0:x0+w].astype(np.float32)
        img[y0:y0+h,x0:x0+w]=(patch*(1-alpha)+rgb*alpha).astype(np.uint8)
