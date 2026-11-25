"""Incremental version of merge_indexed.py for live input.

Same policy as merge_indexed.py / merge_passages.py:
  * every primary (full) event counts immediately,
  * a secondary (roi) event is suppressed when it is the same passage as a primary
    event within +-3 s (merge_passages.same_passage), otherwise it is added when its
    anchor was seen on both sides of the line within +-75 frames.

The offline merger sees the whole recording. Here a secondary event is decided once
every frame that the offline rule could look at has arrived:
  primary events up to +75 frames (3 s at 25 fps) and their trajectories up to
  +100 frames beyond that. Memory is bounded to a fixed window of recent frames.

With the profile key ``dedup_passages`` (as in merge_indexed.py) a primary event is also dropped
when an earlier kept primary event is the same passage (two tracks on one person), and an added
secondary event when an earlier added one is. Primary events are then decided after
DECISION_DELAY too, and secondary events after a further 3 s, so that every primary event they
are compared with is already decided: the result stays identical to the offline merge.
"""
from collections import deque
from itertools import chain
from merge_passages import same_passage

FPS_FRAMES_3S=75      # merge_indexed: |seconds difference| <= 3 at 25 fps
HISTORY=100           # merge_indexed.history(): frame +- 100
SIDE_WINDOW=75        # merge_indexed: anchors within +-75 frames of the secondary event
DECISION_DELAY=FPS_FRAMES_3S+HISTORY+1
KEEP=DECISION_DELAY+2*HISTORY+FPS_FRAMES_3S


class StreamMerger:
    def __init__(self,line_band=(1692,1764),profile=None,size=None):
        self.records=[deque(),deque()]      # per branch: (frame, {track_id: person})
        self.primary=deque()                # recent primary events
        self.pending=deque()                # secondary events awaiting a decision
        self.line_band=line_band
        if profile and profile.get('line'):
            from merge_indexed import both_sides_check
            self.side_check=both_sides_check(profile,size)
        else:
            lo,hi=line_band
            self.side_check=lambda anchors:any(a[0]<lo for a in anchors) and any(a[0]>hi for a in anchors)
        self.added=[];self.suppressed=0;self.primary_total=0;self.primary_ltr=0;self.last_frame=-1
        self.dedup=bool(profile and profile.get('dedup_passages'));self.match=(profile or {}).get('passage_match')
        self.pending_primary=deque();self.duplicates=0
        self.primary_dropped=deque()  # duplicates: not counted, but still a passage secondary events can match
        self.primary_accepted=[]     # primary events accepted by the latest update()/finish()
        self.secondary_delay=DECISION_DELAY+(FPS_FRAMES_3S if self.dedup else 0)

    def history(self,branch,tid,frame):
        lo,hi=frame-HISTORY,frame+HISTORY
        return {f:people[tid] for f,people in self.records[branch] if lo<=f<=hi and tid in people}

    def update(self,frame,full_record,roi_record,full_events,roi_events):
        """Feed one observation of both branches; returns secondary events newly added."""
        if frame<=self.last_frame:raise ValueError('Frames must arrive once, in order')
        self.last_frame=frame
        for branch,record in enumerate([full_record,roi_record]):
            self.records[branch].append((frame,{p['id']:p for p in record['people']}))
        if self.dedup:
            self.pending_primary.extend(full_events);self.primary_accepted=self._decide_primary(frame)
        else:
            for e in full_events:self._keep_primary(e)
            self.primary_accepted=list(full_events)
        self.pending.extend(roi_events)
        decided=self._decide(frame)
        cutoff=frame-KEEP
        for r in self.records:
            while r and r[0][0]<cutoff:r.popleft()
        for q in (self.primary,self.primary_dropped):
            while q and q[0]['frame']<cutoff:q.popleft()
        return decided

    def _keep_primary(self,e):
        self.primary.append(e);self.primary_total+=1;self.primary_ltr+=e['direction']=='ltr'

    def _decide_primary(self,frame,final=False):
        out=[]
        while self.pending_primary and (final or self.pending_primary[0]['frame']+DECISION_DELAY<=frame):
            a=self.pending_primary.popleft();pa=self.history(0,int(a['track_id']),int(a['frame']))
            if not pa:raise ValueError('Primary event has no observed trajectory')
            seconds=float(a['seconds'])
            if any(seconds-float(k['seconds'])<=3 and same_passage(k,self.history(0,int(k['track_id']),int(k['frame'])),a,pa,self.match) for k in self.primary):
                self.duplicates+=1;self.primary_dropped.append(a);continue
            self._keep_primary(a);out.append(a)
        return out

    def _decide(self,frame,final=False):
        out=[]
        while self.pending and (final or self.pending[0]['frame']+self.secondary_delay<=frame):
            b=self.pending.popleft();out+=self._judge(b)
        return out

    def _judge(self,b):
        bf=int(b['frame']);seconds=float(b['seconds'])
        pb=self.history(1,int(b['track_id']),bf)
        if not pb:raise ValueError('Secondary event has no observed trajectory')
        for a in chain(self.primary,self.primary_dropped):
            if seconds-3<=float(a['seconds'])<=seconds+3:
                if same_passage(a,self.history(0,int(a['track_id']),int(a['frame'])),b,pb,self.match):
                    self.suppressed+=1;return []
        if self.dedup and any(abs(seconds-float(x['seconds']))<=3 and same_passage(x,self.history(1,int(x['track_id']),int(x['frame'])),b,pb,self.match) for x in self.added):
            self.duplicates+=1;return []
        anchors=[p['anchor'] for f,p in pb.items() if abs(f-bf)<=SIDE_WINDOW]
        if self.side_check(anchors):
            self.added.append(b);return [b]
        return []

    def finish(self):
        self.primary_accepted=self._decide_primary(self.last_frame,final=True) if self.dedup else []
        return self._decide(self.last_frame,final=True)

    def totals(self):
        ltr=self.primary_ltr+sum(1 for e in self.added if e['direction']=='ltr')
        events=self.primary_total+len(self.added)
        return dict(ltr=ltr,rtl=events-ltr,events=events,primary=self.primary_total,added=len(self.added),
                    suppressed=self.suppressed,duplicates=self.duplicates,pending=len(self.pending)+len(self.pending_primary))
