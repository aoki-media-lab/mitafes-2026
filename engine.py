"""Bounded-state adaptation of the frozen camera algorithms, independent of inference."""
from collections import Counter,defaultdict,deque
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from camera_counter import (TemporalCounter,PoseAnchorTracker,make_tracker,objects,
                            person_quality,inside,foot,umbrella_point,associate_umbrellas)
from ultralytics.trackers.basetrack import BaseTrack


def make_people_counter(profile,width,fps):
    """Line counter for people, with the extra crossing rules the profile enables."""
    if profile.get('stitch_crossings',False):
        from stitch_counter import StitchCounter
        return StitchCounter(profile,width,fps)
    if profile.get('late_birth_crossings',False):
        from late_birth_counter import LateBirthCounter
        return LateBirthCounter(profile,width,fps)
    if profile.get('occlusion_crossings',False):
        from occlusion_counter import OcclusionCounter
        return OcclusionCounter(profile,width,fps)
    return TemporalCounter(profile,width,fps)


class StreamingCounter:
    def __init__(self,profile,width,height,fps,stride):
        self.profile=profile;self.width=width;self.height=height;self.fps=fps;self.stride=stride
        self.pose_mode=profile.get('tracker')=='pose_anchor'
        if self.pose_mode:
            if profile.get('robust_tracking',False):
                from robust_tracker import RobustPoseAnchorTracker
                self.people_tracker=RobustPoseAnchorTracker(fps,profile,width,height)
            else:
                from vectorized_tracker import VectorizedPoseAnchorTracker
                cls=VectorizedPoseAnchorTracker if profile.get('vectorized_tracking',True) else PoseAnchorTracker
                self.people_tracker=cls(fps,profile)
        else:self.people_tracker=make_tracker(fps/stride,profile)
        self.umbrella_tracker=make_tracker(fps/(stride*profile.get('umbrella_every',1)),profile,umbrella=True)
        self.people_counter=make_people_counter(profile,width,fps)
        self.umbrella_counter=TemporalCounter(profile,width,fps)
        self.totals=Counter(ltr=0,rtl=0)
        self.umbrella_total=0;self.umbrella_review_total=0;self.rejected=0
        self.links=defaultdict(Counter);self.umbrella_last={}
        self.recent_people=deque();self.pending_umbrellas=deque()
        self.last_frame=-1

    def update(self,item):
        frame=item['index'];p=self.profile;w=self.width;h=self.height
        if frame<=self.last_frame:raise ValueError('Frames must arrive once, in source order')
        self.last_frame=frame
        people_dets=[];points=[];head_flags=[]
        head_only=set(item.get('head_only_indices',[]))
        for i,d in enumerate(item['detections']):
            if int(d[5])!=0:continue
            if i not in head_only and not inside(foot(d[:4]),p,w,h):continue
            if not person_quality(d[:4],p,w,h)[0]:self.rejected+=1;continue
            people_dets.append(d)
            head_flags.append(i in head_only)
            if self.pose_mode:points.append(item['keypoints'][i])
        umbrella_dets=[d for d in item['detections'] if int(d[5])==25 and inside(umbrella_point(d[:4],p),p,w,h)]
        if self.pose_mode and p.get('robust_tracking',False):
            people=self.people_tracker.update(people_dets,points,frame,head_flags=head_flags)
        else:people=self.people_tracker.update(people_dets,points,frame) if self.pose_mode else objects(self.people_tracker,people_dets,(h,w))
        people=[x for x in people if person_quality(x['xyxy'],p,w,h)[0]]
        # A scheduled skipped inference is not evidence that umbrellas vanished.
        umbrellas=objects(self.umbrella_tracker,umbrella_dets,(h,w)) if item.get('umbrella_observed',True) else []
        for person in people:
            x,y=person.get('anchor',foot(person['xyxy']))
            self.people_counter.update(person['id'],x,y,frame)
            st=self.people_counter.states.get(person['id'])
            if st is not None:
                st['h']=person['xyxy'][3]-person['xyxy'][1]   # person size for duplicate checks
                st['conf']=max(st.get('conf',0.),person.get('confidence',1.))   # best detection score so far (rule_min_confidence)
        if hasattr(self.people_counter,'tick'):self.people_counter.tick(frame)
        events=self.people_counter.events[:];self.people_counter.events.clear()
        for event in events:
            self.totals[event['direction']]+=1;event['event_id']=sum(self.totals.values())
            self.recent_people.append(event)
        for umbrella in umbrellas:
            x,y=umbrella_point(umbrella['xyxy'],p)
            self.umbrella_counter.update(umbrella['id'],x,y,frame)
            self.umbrella_last[umbrella['id']]=frame
        for event in self.umbrella_counter.events:
            self.umbrella_total+=1;event['event_id']=self.umbrella_total;self.pending_umbrellas.append(event)
        self.umbrella_counter.events.clear()
        if p.get('vectorized_tracking',True):
            from vectorized_association import associate_umbrellas as associate
        else:associate=associate_umbrellas
        matches=associate(umbrellas,[x for x in people if not x.get('head_only',False)],p)
        for uid,pid,cost in matches:self.links[uid][pid]+=1
        audits=self.audit(frame/self.fps)
        # Ended track IDs cannot reconnect after the frozen tracker lifetime.
        # Keep ample extra history, then retire states rather than accumulating hours.
        if frame%max(1,round(self.fps))<self.stride:
            cutoff=frame-round(self.fps*10)
            for counter in [self.people_counter,self.umbrella_counter]:
                counter.states={tid:s for tid,s in counter.states.items() if s['last_frame']>=cutoff}
            for uid,last in list(self.umbrella_last.items()):
                if last<cutoff:
                    self.umbrella_last.pop(uid);self.links.pop(uid,None)
            while self.recent_people and self.recent_people[0]['seconds']<frame/self.fps-5:
                self.recent_people.popleft()
        record=dict(frame=frame,people=people,umbrellas=umbrellas,matches=matches)
        if 'person_observation_kind' in item:record['person_observation_kind']=item['person_observation_kind']
        return record,events,audits

    def audit(self,seconds,final=False):
        result=[];tol=self.profile['umbrella_event_tolerance_seconds']
        while self.pending_umbrellas and (final or self.pending_umbrellas[0]['seconds']+tol<=seconds):
            event=self.pending_umbrellas.popleft()
            candidates={pid for pid,n in self.links[event['track_id']].items() if n>=2}
            covered=[e['event_id'] for e in self.recent_people if e['track_id'] in candidates and e['direction']==event['direction'] and abs(e['seconds']-event['seconds'])<=tol]
            # Sparse auxiliary inference can erase a competing owner's evidence.
            # Keep its audit conservative instead of falsely resolving ambiguity.
            status='person_observed' if len(covered)==1 and self.profile.get('umbrella_every',1)==1 else 'needs_review'
            if status=='needs_review':self.umbrella_review_total+=1
            result.append(dict(umbrella_event_id=event['event_id'],umbrella_track_id=event['track_id'],direction=event['direction'],seconds=event['seconds'],associated_person_event_ids=covered,status=status))
        return result

    def summary(self):
        return dict(ltr=self.totals['ltr'],rtl=self.totals['rtl'],events=sum(self.totals.values()),
                    umbrella_crossings=self.umbrella_total,umbrella_review_candidates=self.umbrella_review_total,
                    rejected_observations=self.rejected,
                    live_states=dict(people=len(self.people_counter.states),umbrellas=len(self.umbrella_counter.states),
                                     recent_events=len(self.recent_people),umbrella_links=len(self.links)),
                    pending_at_end=self.people_counter.pending(self.last_frame,self.stride+1))

    def snapshot(self):
        # ByteTrack's allocator is a class variable, outside the instance pickle.
        return dict(engine=self,byte_track_id_counter=BaseTrack._count)

    @staticmethod
    def restore(snapshot):
        BaseTrack._count=snapshot['byte_track_id_counter']
        return snapshot['engine']
