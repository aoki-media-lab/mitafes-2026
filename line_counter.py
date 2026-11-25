"""Counting line of any orientation, for camera views where a vertical line_x does not fit.

A profile either keeps the original vertical ``line_x`` or gives
``line: [[x1, y1], [x2, y2]]`` in fractions of the frame. Positions are reduced to a
signed offset from the line in fractions of the frame width, so ``margin`` keeps its
meaning. The positive side is the one with larger image x, therefore ``ltr`` / ``rtl``
still mean image left -> right / right -> left. With a vertical ``line_x`` the offset is
exactly ``x / width - line_x``: the original counters' behaviour is unchanged.

``line_extent`` (optional, fraction of the line length, default unlimited) limits counting
to crossings whose foot point projects onto the drawn segment, extended by that fraction at
each end. This keeps people walking parallel to the line beyond its ends from being counted.
"""
from collections import deque
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from camera_counter import TemporalCounter


class CountingLine:
    def __init__(self,profile,width,height):
        self.width=width;self.height=height
        if profile.get('line'):
            (x1,y1),(x2,y2)=profile['line']
            x1,x2=x1*width,x2*width;y1,y2=y1*height,y2*height
        else:
            x1=x2=profile['line_x']*width;y1,y2=0.,float(height)
        dx,dy=x2-x1,y2-y1;length=math.hypot(dx,dy)
        if length==0:raise ValueError('Counting line needs two distinct points')
        nx,ny=dy/length,-dx/length
        if nx<0 or (nx==0 and ny<0):nx,ny=-nx,-ny
        self.p1=(x1,y1);self.p2=(x2,y2);self.length=length
        self.normal=(nx,ny);self.direction=(dx/length,dy/length)
        extent=profile.get('line_extent')
        self.extent=None if extent is None else extent*length

    def offset(self,x,y):
        """Signed distance from the line as a fraction of the frame width (+ = right side)."""
        return ((x-self.p1[0])*self.normal[0]+(y-self.p1[1])*self.normal[1])/self.width

    def along(self,x,y):
        """Pixel position along the line, measured from its first point."""
        return (x-self.p1[0])*self.direction[0]+(y-self.p1[1])*self.direction[1]

    def within(self,x,y):
        if self.extent is None:return True
        t=self.along(x,y);return -self.extent<=t<=self.length+self.extent

    def pixels(self):
        return [tuple(map(round,self.p1)),tuple(map(round,self.p2))]


class LineCounter(TemporalCounter):
    """TemporalCounter with the vertical-line test replaced by CountingLine.offset."""
    def __init__(self,profile,width,fps,height=None):
        super().__init__(profile,width,fps)
        self.line=CountingLine(profile,width,height or round(width*9/16))

    def update(self,tid,x,y,frame):
        p=self.profile
        state=self.states.setdefault(tid,{"stable":None,"candidate":0,"n":0,"last_frame":None})
        if state['last_frame'] is not None:
            gap=(frame-state['last_frame'])/self.fps
            if gap>p['counter_max_gap_seconds']:state.update(candidate=0,n=0)
            if gap>p['counter_stale_seconds']:state['stable']=None
        state.update(last_x=x,last_y=y,last_frame=frame)
        state.pop('linked_out',None)   # seen again: it may hand over its side once more
        distance=self.line.offset(x,y)
        side=-1 if distance<-p['margin'] else (1 if distance>p['margin'] else 0)
        if side==0:
            state.update(candidate=0,n=0);return
        if state['candidate']!=side:state.update(candidate=side,n=1,candidate_start=frame)
        else:state['n']+=1
        if state['n']<2 or (frame-state['candidate_start'])/self.fps<.06:return
        if state['stable'] is not None and state['stable']!=side and self.line.within(x,y):
            self.events.append(dict(event_id=len(self.events)+1,track_id=tid,
                                    direction='ltr' if side==1 else 'rtl',frame=frame,
                                    seconds=round(frame/self.fps,3),x=round(x,1),y=round(y,1)))
        state['stable']=side

    def pending(self,final_frame,tolerance):
        result=[]
        for tid,s in self.states.items():
            x=self.line.offset(s['last_x'],s['last_y'])
            if final_frame-s['last_frame']<=tolerance and ((s['stable']==1 and x<0) or (s['stable']==-1 and x>0)):
                result.append(dict(track_id=tid,direction='rtl' if s['stable']==1 else 'ltr',last_frame=s['last_frame']))
        return result


    def link(self,frame,active):
        """Hand a just-lost track's confirmed side to a new track that starts next to it.

        ``active`` is [(track_id, x, y)] for this observation. Near the frame edge a person
        is often only partly visible, the tracker splits them into fragments, and the
        fragment that finishes the crossing is born already past the line, so no side
        change is ever seen. Enabled by ``link_seconds`` (max gap) and ``link_distance``
        (max jump, fraction of frame width). Each lost track is handed over at most once.
        """
        self.frame_active={tid:(x,y) for tid,x,y in active}   # used by the birth rescue
        seconds=self.profile.get('link_seconds')
        if not seconds:return
        present={tid for tid,_,_ in active}
        new=[(tid,x,y) for tid,x,y in active if tid not in self.states]
        if not new:return
        margin=self.profile['margin']
        # The lost track must still be on its confirmed side (or inside the margin band)
        # when last seen, so a stale side is never passed on.
        donors=[(tid,s) for tid,s in self.states.items()
                if tid not in present and s['stable'] is not None and not s.get('linked_out')
                and 0<frame-s['last_frame']<=seconds*self.fps
                and s['stable']*self.line.offset(s['last_x'],s['last_y'])>=-margin]
        limit=self.profile.get('link_distance',.1)*self.width
        # A new track on top of another live track is a duplicate box, not a continuation.
        duplicate=self.profile.get('link_duplicate_distance',.03)*self.width
        old=[(x,y) for tid,x,y in active if tid in self.states]
        new=[(tid,x,y) for tid,x,y in new if all(math.hypot(x-ox,y-oy)>duplicate for ox,oy in old)]
        pairs=sorted((math.hypot(x-s['last_x'],y-s['last_y']),tid,did)
                     for tid,x,y in new for did,s in donors)
        used_new=set();used_donor=set()
        for dist,tid,did in pairs:
            if dist>limit:break
            if tid in used_new or did in used_donor:continue
            used_new.add(tid);used_donor.add(did);donor=self.states[did];donor['linked_out']=True
            self.states[tid]={"stable":donor['stable'],"candidate":0,"n":0,"last_frame":None,
                              "linked_from":did,"birth_done":True}


class LineOcclusionCounter(LineCounter):
    """occlusion_counter.OcclusionCounter in line coordinates (u: across, v: along the line)."""
    def update(self,tid,x,y,frame):
        previous=len(self.events)
        super().update(tid,x,y,frame)
        state=self.states[tid]
        if len(self.events)>previous:state['has_crossed']=True
        samples=state.setdefault('birth_samples',deque(maxlen=16))
        if state.get('birth_done'):return
        u=self.line.offset(x,y)*self.width;v=self.line.along(x,y)
        if not samples:
            state['birth_xy']=(frame,x,y)
            state['birth_others']={t:xy for t,xy in getattr(self,'frame_active',{}).items() if t!=tid}
        samples.append((frame,u,v))
        if frame-samples[0][0]>round(.8*self.fps):state['birth_done']=True;return
        if state.get('has_crossed') or len(samples)<4:return
        margin=self.profile['margin']*self.width
        origin=samples[0][1]
        band=self.profile.get('birth_band')
        legacy=self.width*.003<=abs(samples[0][1])<self.profile['margin']*self.width
        if band and not legacy and len(samples)==4:
            # A track born on top of another live track is a duplicate box, not a person.
            frame0,x0,y0=state['birth_xy']
            duplicate=self.profile.get('link_duplicate_distance',.03)*self.width
            if any(math.hypot(x0-ox,y0-oy)<=duplicate for other,(ox,oy) in state['birth_others'].items()):
                state['birth_done']=True;return
        if band:
            # Recall-oriented: a track that first appears within birth_band of the line
            # (typically just after crossing while hidden) and walks away from it counts
            # in its walking direction. A start clearly on the source side is left to the
            # ordinary crossing rule.
            if abs(origin)>band*self.width:return
            direction=1 if u>origin else -1
            if -direction*origin>margin:return
        else:
            if not self.width*.003<=abs(origin)<margin:return
            direction=1 if origin<0 else -1
        if direction*(u-samples[0][1])<self.width*.03:return
        target=1 if direction>0 else -1
        if state['stable']!=target or state['candidate']!=target or state['n']<2:return
        if (frame-state['candidate_start'])/self.fps<.06:return
        if not self.line.within(x,y):return
        du=[direction*(b[1]-a[1]) for a,b in zip(samples,list(samples)[1:])]
        if sum(d>=-5 for d in du)/len(du)<.7:return
        if abs(v-samples[0][2])>max(90,.5*abs(u-samples[0][1])):return
        state['has_crossed']=True;state['birth_done']=True
        self.events.append(dict(event_id=len(self.events)+1,track_id=tid,
                                direction='ltr' if direction>0 else 'rtl',frame=frame,
                                seconds=round(frame/self.fps,3),x=round(x,1),y=round(y,1)))


def crossed_both_sides(line,profile,anchors):
    """Reconciliation guard: the anchors were seen beyond the margin on both sides."""
    offsets=[line.offset(x,y) for x,y in anchors];m=profile['margin']
    return any(o<-m for o in offsets) and any(o>m for o in offsets)
