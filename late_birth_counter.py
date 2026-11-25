"""Count people who first become visible just after crossing the line while occluded.

A companion hidden behind another walker can cross the line unseen and be detected
for the first time on the far side. Such a track never has a stable side before the
line, so TemporalCounter / OcclusionCounter cannot count it. This counter adds one
event when a track is born beyond the hysteresis margin but close to the line, then
walks steadily away from it, and back-extrapolating its speed puts it on the line
shortly before birth. A recently lost track nearby that already counted the same
direction is treated as an ID switch, not a new person.
Enabled by profile['late_birth_crossings']; all thresholds come from the profile.
profile['rule_min_confidence'] (default 0): only tracks whose best detection score reached it may be counted
by this rule (and by stitch / late_death), so faint fragments started under a low new-track threshold do not.
"""
from collections import deque
from occlusion_counter import OcclusionCounter

DEFAULTS=dict(late_birth_max_distance=.15,      # of width, birth distance from line
              late_birth_min_travel=.03,        # of width, travel away from line
              late_birth_observe_seconds=.48,   # decide after this much track life
              late_birth_max_back_seconds=1.0,  # back-extrapolated time to the line
              late_birth_fragment_seconds=1.0,  # lost-track window for ID switches
              late_birth_fragment_radius=200,   # px, lost-track distance for ID switches
              late_birth_duplicate_radius=60,   # px, live track at the same spot (upper bound)
              late_birth_duplicate_ratio=.06)   # of the larger person box height: companions walking
                                                # side by side are further apart than this


class LateBirthCounter(OcclusionCounter):
    def p(self,key):return self.profile.get(key,DEFAULTS[key])

    def dup_radius(self,a,b):
        """Distance under which two tracks are the same person, scaled by person size (box height 'h'
        stored on the state by the engine); falls back to the fixed radius without sizes."""
        h=max(a.get('h',0),b.get('h',0))
        r=self.p('late_birth_duplicate_radius')
        return min(r,self.p('late_birth_duplicate_ratio')*h) if h else r

    def update(self,tid,x,y,frame):
        previous=len(self.events)
        super().update(tid,x,y,frame)
        for e in self.events[previous:]:self.states[e['track_id']]['crossed_dir']=e['direction']
        state=self.states[tid];state.setdefault('first_frame',frame)
        state.setdefault('trail',deque(maxlen=16)).append((frame,float(x),float(y)))
        if state.get('late_done'):return
        samples=state.setdefault('late_samples',deque(maxlen=32));samples.append((frame,float(x),float(y)))
        if state.get('has_crossed'):state['late_done']=True;return
        f0,x0,y0=samples[0];age=(frame-f0)/self.fps
        if age<self.p('late_birth_observe_seconds'):return
        state['late_done']=True
        if state.get('conf',1.)<self.profile.get('rule_min_confidence',0):return
        line=self.profile['line_x']*self.width;margin=self.profile['margin']*self.width
        offset=x0-line
        if not margin<=abs(offset)<=self.p('late_birth_max_distance')*self.width:return
        side=1 if offset>0 else -1
        travel=side*(x-x0)
        if travel<self.p('late_birth_min_travel')*self.width:return
        steps=[side*(b[1]-a[1]) for a,b in zip(samples,list(samples)[1:])]
        if sum(v>=-5 for v in steps)/len(steps)<.7:return
        if abs(y-y0)>max(90,.5*abs(x-x0)):return
        speed=travel/age;back=abs(offset)/speed
        if back>self.p('late_birth_max_back_seconds'):return
        direction='ltr' if side>0 else 'rtl'
        window=self.p('late_birth_fragment_seconds')*self.fps;radius=self.p('late_birth_fragment_radius')
        for other,s in self.states.items():
            if other==tid or s['last_frame'] is None:continue
            distance=((s['last_x']-x0)**2+(s['last_y']-y0)**2)**.5
            if s['last_frame']<f0 and f0-s['last_frame']<=window and distance<=radius and s.get('crossed_dir')==direction:return
            # A second detection of someone already tracked moves with that track.
            here=((s['last_x']-x)**2+(s['last_y']-y)**2)**.5
            if s.get('first_frame',frame)<f0 and frame-s['last_frame']<=2 and here<=self.dup_radius(s,state):return
            # ...also when that track was lost again before this decision: if a track that
            # already counted this direction ran alongside this one, it is the same person.
            if s.get('crossed_dir')==direction and s.get('first_frame',frame)<f0 and any(
                    abs(fa-fb)<=2 and ((xa-xb)**2+(ya-yb)**2)**.5<=self.dup_radius(s,state)
                    for fa,xa,ya in s.get('trail',()) if fa>=f0-2 for fb,xb,yb in samples):return
        crossing=f0-round(back*self.fps);crossing-=crossing%2
        state['has_crossed']=True;state['crossed_dir']=direction
        self.events.append(dict(event_id=len(self.events)+1,track_id=tid,direction=direction,frame=crossing,
                                seconds=round(crossing/self.fps,3),x=round(line,1),y=round(y0,1),rule='late_birth'))
