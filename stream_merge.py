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
"""
from collections import deque
from merge_passages import same_passage

FPS_FRAMES_3S=75      # merge_indexed: |seconds difference| <= 3 at 25 fps
HISTORY=100           # merge_indexed.history(): frame +- 100
SIDE_WINDOW=75        # merge_indexed: anchors within +-75 frames of the secondary event
DECISION_DELAY=FPS_FRAMES_3S+HISTORY+1
KEEP=DECISION_DELAY+2*HISTORY+FPS_FRAMES_3S


class StreamMerger:
    def __init__(self,line_band=(1692,1764)):
        self.records=[deque(),deque()]      # per branch: (frame, {track_id: person})
        self.primary=deque()                # recent primary events
        self.pending=deque()                # secondary events awaiting a decision
        self.line_band=line_band
        self.added=[];self.suppressed=0;self.primary_total=0;self.primary_ltr=0;self.last_frame=-1

    def history(self,branch,tid,frame):
        lo,hi=frame-HISTORY,frame+HISTORY
        return {f:people[tid] for f,people in self.records[branch] if lo<=f<=hi and tid in people}

    def update(self,frame,full_record,roi_record,full_events,roi_events):
        """Feed one observation of both branches; returns secondary events newly added."""
        if frame<=self.last_frame:raise ValueError('Frames must arrive once, in order')
        self.last_frame=frame
        for branch,record in enumerate([full_record,roi_record]):
            self.records[branch].append((frame,{p['id']:p for p in record['people']}))
        for e in full_events:
            self.primary.append(e);self.primary_total+=1;self.primary_ltr+=e['direction']=='ltr'
        self.pending.extend(roi_events)
        decided=self._decide(frame)
        cutoff=frame-KEEP
        for r in self.records:
            while r and r[0][0]<cutoff:r.popleft()
        while self.primary and self.primary[0]['frame']<cutoff:self.primary.popleft()
        return decided

    def _decide(self,frame,final=False):
        out=[]
        while self.pending and (final or self.pending[0]['frame']+DECISION_DELAY<=frame):
            b=self.pending.popleft();out+=self._judge(b)
        return out

    def _judge(self,b):
        bf=int(b['frame']);seconds=float(b['seconds'])
        pb=self.history(1,int(b['track_id']),bf)
        if not pb:raise ValueError('Secondary event has no observed trajectory')
        for a in self.primary:
            if seconds-3<=float(a['seconds'])<=seconds+3:
                if same_passage(a,self.history(0,int(a['track_id']),int(a['frame'])),b,pb):
                    self.suppressed+=1;return []
        xs=[p['anchor'][0] for f,p in pb.items() if abs(f-bf)<=SIDE_WINDOW]
        lo,hi=self.line_band
        if b.get('rule') or (any(x<lo for x in xs) and any(x>hi for x in xs)):
            self.added.append(b);return [b]
        return []

    def finish(self):
        return self._decide(self.last_frame,final=True)

    def totals(self):
        ltr=self.primary_ltr+sum(1 for e in self.added if e['direction']=='ltr')
        events=self.primary_total+len(self.added)
        return dict(ltr=ltr,rtl=events-ltr,events=events,primary=self.primary_total,added=len(self.added),
                    suppressed=self.suppressed,pending=len(self.pending))
