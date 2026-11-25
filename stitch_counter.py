"""Count crossings whose track broke while people overlapped at the line.

In a group, the tracker often loses a person for a moment right at the line and
continues them under a new ID (or not at all). Neither piece then has stable
observations on both sides, so the ordinary counter misses the passage. This
counter looks at each uncounted track once it has been gone for
stitch_window_seconds:

  * stitch:     it approached the line from one side and a new uncounted track
                appeared on the far side where its motion predicts -> one crossing;
  * late_death: it reached the line (or just past it) moving toward the far side
                and no continuation appeared -> one crossing (biased to over-count:
                missing a visitor is worse than counting one twice).

A track that already counted (by any rule), a continuation that is still on the
approach side, or a second detection of someone who is still tracked, prevents
the count. Enabled by profile['stitch_crossings']; thresholds come from the profile.
"""
from collections import deque
from late_birth_counter import LateBirthCounter

DEFAULTS=dict(stitch_window_seconds=1.5,      # wait this long after a track vanished
              stitch_max_distance=.15,        # of width: last position of the vanished track from the line
              stitch_radius_x=.06,            # of width: predicted vs. actual position of the continuation
              stitch_radius_y=120,            # px
              death_max_distance=.04,         # of width: approach-side distance from the line for late_death
              death_min_speed=60,             # px/s toward the line at the end of the track
              stitch_min_samples=4)


class StitchCounter(LateBirthCounter):
    def p(self,key):return self.profile.get(key,DEFAULTS.get(key)) if key in DEFAULTS else super().p(key)

    def update(self,tid,x,y,frame):
        super().update(tid,x,y,frame)
        state=self.states[tid]
        state.setdefault('path',deque(maxlen=40)).append((frame,float(x),float(y)))

    def tick(self,frame):
        """Call once per observation after all updates of that observation."""
        window=self.p('stitch_window_seconds')*self.fps
        for tid,s in list(self.states.items()):
            if s.get('stitch_done') or s.get('crossed_dir') or s['last_frame'] is None:continue
            if frame-s['last_frame']<window:continue
            s['stitch_done']=True
            self.judge(tid,s)

    def judge(self,tid,s):
        W=self.width;line=self.profile['line_x']*W;margin=self.profile['margin']*W
        path=list(s.get('path',()))
        if len(path)<self.p('stitch_min_samples') or s.get('stable') not in(1,-1):return
        side=s['stable'];direction='rtl' if side==1 else 'ltr'   # stable side = where it came from
        floor=self.profile.get('rule_min_confidence',0);weak=s.get('conf',1.)<floor   # faint fragment (see late_birth_counter)
        f1,x1,y1=path[-1]
        # velocity over the last ~0.6 s, toward the line is negative*side
        old=[q for q in path if q[0]>=f1-round(.6*self.fps)][0]
        dt=(f1-old[0])/self.fps
        if dt<=0:return
        vx=(x1-old[1])/dt;vy=(y1-old[2])/dt
        toward=-side*vx
        offset=side*(x1-line)                                   # >0: still on the approach side
        if offset>self.p('stitch_max_distance')*W or offset<-self.p('stitch_max_distance')*W:return
        # A second detection of someone who is still tracked (or was counted) moves with that track.
        for other,o in self.states.items():
            if other==tid:continue
            near=[(fa,xa,ya) for fa,xa,ya in o.get('path',()) if any(abs(fa-fb)<=2 and ((xa-xb)**2+(ya-yb)**2)**.5<=self.dup_radius(o,s)
                                                                  for fb,xb,yb in path[-8:])]
            if near and (o['last_frame']>f1 or o.get('crossed_dir')==direction) and o.get('first_frame',f1)<=path[-8:][0][0]:return
        # Continuation: a track born after this one vanished, where its motion predicts.
        window=self.p('stitch_window_seconds')*self.fps
        for other,o in self.states.items():
            if other==tid:continue
            first=o.get('path') and o['path'][0]
            if not first or not(f1-4<=first[0]<=f1+window) or o.get('first_frame',first[0])<first[0]:continue
            fb,xb,yb=first;t=min(max(fb-f1,0)/self.fps,self.p('stitch_window_seconds'))
            if abs(x1+vx*t-xb)>self.p('stitch_radius_x')*W or abs(y1+vy*t-yb)>self.p('stitch_radius_y'):continue
            if o.get('crossed_dir')==direction:return               # the continuation already counted
            if side*(xb-line)>-margin:return                        # continuation still on the approach side: it carries the person
            if weak and o.get('conf',1.)<floor:return               # both pieces faint: no count
            o['has_crossed']=True;o['crossed_dir']=direction;o['late_done']=True;o['stitch_done']=True
            s['crossed_dir']=direction
            self.emit(other,direction,f1,line,y1,'stitch');return
        # No continuation: count if it reached the line moving toward the far side.
        if offset>self.p('death_max_distance')*W or toward<self.p('death_min_speed'):return
        if weak:return
        s['crossed_dir']=direction
        self.emit(tid,direction,f1,line,y1,'late_death')

    def emit(self,tid,direction,frame,line,y,rule):
        frame-=frame%2
        self.events.append(dict(event_id=len(self.events)+1,track_id=tid,direction=direction,frame=frame,
                                seconds=round(frame/self.fps,3),x=round(line,1),y=round(y,1),rule=rule))
