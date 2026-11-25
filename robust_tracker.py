"""Pose tracking with fast-motion initialization and body-supported continuity."""
from collections import deque
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from pose_tracker import PoseAnchorTracker,head_anchor


def batch_anchors(detections,keypoints):
    boxes=np.asarray(detections,float)[:,:4];points=np.asarray(keypoints,float)
    visible=points[:,:5,2]>=.25;counts=visible.sum(axis=1);safe=np.maximum(counts,1)
    values=np.where(visible[:,:,None],points[:,:5,:2],np.inf)
    ordered=np.sort(values,axis=1);row=np.arange(len(boxes))
    anchors=(ordered[row,(safe-1)//2]+ordered[row,safe//2])/2
    maximum=np.where(visible[:,:,None],points[:,:5,:2],-np.inf).max(axis=1)
    minimum=values.min(axis=1)
    scales=np.maximum(30.,np.linalg.norm(maximum-minimum,axis=1)*1.5)
    heights=boxes[:,3]-boxes[:,1]
    scales=np.maximum(scales,np.minimum(heights*.1,90.))
    methods=np.full(len(boxes),'head_keypoints',dtype=object)
    for i in np.flatnonzero(counts==0):
        # Invisible-head fallbacks are uncommon; preserve the original arithmetic.
        anchors[i],scales[i],methods[i]=head_anchor(boxes[i],points[i])
    return anchors,scales,methods


def batch_predictions(tracks,time,max_velocity=4000.):
    predictions=np.array([t['anchor'] for t in tracks])
    groups={}
    for i,t in enumerate(tracks):groups.setdefault(len(t['history']),[]).append(i)
    for size,indices in groups.items():
        if size<2:continue
        histories=[list(tracks[i]['history']) for i in indices]
        times=np.array([[x[0] for x in hist] for hist in histories])
        values=np.array([[x[1] for x in hist] for hist in histories])
        eligible=times[:,-1]-times[:,0]>=.06
        centered=times-times.mean(axis=1)[:,None]
        velocity=(centered[:,:,None]*(values-values.mean(axis=1)[:,None,:])).sum(axis=1)/(centered**2).sum(axis=1)[:,None]
        velocity=np.clip(velocity,-max_velocity,max_velocity);velocity[~eligible]=0
        gaps=np.array([time-tracks[i]['last'] for i in indices])
        predictions[indices]+=velocity*gaps[:,None]
    return predictions


class RobustPoseAnchorTracker(PoseAnchorTracker):
    def __init__(self,fps,profile,width=2880,height=1620):
        super().__init__(fps,profile)
        self.width,self.height=width,height

    def update(self,detections,keypoints,frame,head_flags=None):
        time=frame/self.fps
        self.tracks={i:t for i,t in self.tracks.items() if time-t['last']<=self.profile['tracker_lost_seconds']}
        observations=[]
        if len(detections):
            if head_flags is None:head_flags=[False]*len(detections)
            anchors,scales,methods=batch_anchors(detections,keypoints)
            for i,flag in enumerate(head_flags):
                if flag:
                    scales[i]=max(30.,.8*max(detections[i][2]-detections[i][0],detections[i][3]-detections[i][1]))
            for d,k,anchor,scale,method,head_only in zip(detections,keypoints,anchors,scales,methods,head_flags):
                observations.append(dict(xyxy=list(d[:4]),keypoints=np.asarray(k,float),confidence=float(d[4]),anchor=anchor,scale=scale,anchor_method=method,head_only=head_only))
        unique=[];accepted_anchors=[];accepted_scales=[];accepted_heads=[]
        for obs in sorted(observations,key=lambda x:-x['confidence']):
            duplicate=False
            if accepted_anchors and obs['anchor_method']=='head_keypoints':
                distance=np.linalg.norm(np.asarray(accepted_anchors)-obs['anchor'],axis=1)
                threshold=self.profile.get('head_duplicate_fraction',.3)*np.minimum(obs['scale'],accepted_scales)
                duplicate=bool(np.any((distance<threshold)&np.asarray(accepted_heads)))
            if not duplicate and self.profile.get('pose_duplicate_filter',False):
                from fusion import same_pose
                duplicate=any(same_pose(obs['xyxy'],obs['keypoints'],other['xyxy'],other['keypoints']) for other in unique)
            if not duplicate:
                unique.append(obs);accepted_anchors.append(obs['anchor']);accepted_scales.append(obs['scale'])
                accepted_heads.append(obs['anchor_method']=='head_keypoints')
        observations=unique;ids=list(self.tracks)
        cost=np.full((len(ids),len(observations)),1e6)
        if ids and observations:
            tracks=[self.tracks[tid] for tid in ids]
            predictions=batch_predictions(tracks,time,self.profile.get('max_velocity',4000.))
            anchors=np.asarray([o['anchor'] for o in observations])
            gaps=np.asarray([time-t['last'] for t in tracks])
            radii=np.maximum([t['scale'] for t in tracks],30.)*self.profile.get('anchor_match_radius',1.6)+20*gaps
            for i,t in enumerate(tracks):
                if len(t['history'])<3:
                    # A new near-camera track can move several head widths
                    # before its velocity can be estimated from observations.
                    radius=90.
                    if t['anchor'][1]>.7*self.height:
                        radius=self.width*.09
                    radii[i]=max(radii[i],radius)
            distances=np.linalg.norm(predictions[:,None,:]-anchors[None,:,:],axis=2)
            raw=distances/radii[:,None]+.08*gaps[:,None]+.05*(1-np.asarray([o['confidence'] for o in observations]))[None,:]
            cost=np.where(distances<=radii[:,None],raw,1e6)
            if self.profile.get('body_continuity',True):
                body_states=[dict(anchor=t['body_anchor'],history=t['body_history'],last=t['last']) for t in tracks]
                body_predictions=batch_predictions(body_states,time,self.profile.get('max_velocity',4000.))
                delta=body_predictions-np.asarray([t['body_anchor'] for t in tracks])
                previous=np.asarray([t['keypoints'] for t in tracks])
                current=np.asarray([o['keypoints'] for o in observations])
                predicted=previous[:,:,:2]+delta[:,None,:]
                shared=(previous[:,None,5:,2]>=.3)&(current[None,:,5:,2]>=.3)
                errors=np.linalg.norm(predicted[:,None,5:,:]-current[None,:,5:,:2],axis=-1)
                ordered=np.sort(np.where(shared,errors,np.inf),axis=-1)
                counts=shared.sum(axis=-1);safe=np.maximum(counts,1)
                ri=np.arange(len(tracks))[:,None];ci=np.arange(len(observations))[None,:]
                median=(ordered[ri,ci,(safe-1)//2]+ordered[ri,ci,safe//2])/2
                heights=np.asarray([o['xyxy'][3]-o['xyxy'][1] for o in observations])
                tolerance=np.maximum(60,heights*.22)[None,:]
                unreliable=(np.asarray([t['method']!='head_keypoints' for t in tracks])[:,None]
                            |np.asarray([o['anchor_method']!='head_keypoints' for o in observations])[None,:])
                supported=(counts>=2)&(median<tolerance)&unreliable
                body_cost=.2+.5*median/tolerance+.08*gaps[:,None]
                cost=np.where(supported,np.minimum(cost,body_cost),cost)
        used=set();matches={}
        if cost.size:
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                if cost[row,col]<1:matches[col]=ids[row];used.add(col)
        for col,obs in enumerate(observations):
            new_threshold=self.profile.get('head_new_track_thresh',.2) if obs['head_only'] else self.profile.get('pose_new_track_thresh',.3)
            if col not in used and obs['confidence']>=new_threshold:
                tid=self.next_id;self.next_id+=1
                self.tracks[tid]=dict(last=time,anchor=obs['anchor'],scale=obs['scale'],history=deque(maxlen=8),
                                     body_anchor=np.asarray(obs['xyxy']).reshape(2,2).mean(axis=0),body_history=deque(maxlen=8),
                                     keypoints=obs['keypoints'],method=obs['anchor_method'])
                matches[col]=tid
        result=[]
        for col,tid in matches.items():
            obs=observations[col];track=self.tracks[tid]
            if time-track['last']>.4:track['history'].clear()
            track['history'].append((time,obs['anchor']))
            center=np.asarray(obs['xyxy']).reshape(2,2).mean(axis=0)
            if time-track['last']>.4:track['body_history'].clear()
            track['body_history'].append((time,center))
            track.update(last=time,anchor=obs['anchor'],scale=.8*track['scale']+.2*obs['scale'],
                         body_anchor=center,keypoints=obs['keypoints'],method=obs['anchor_method'])
            result.append(dict(id=tid,xyxy=obs['xyxy'],confidence=obs['confidence'],anchor=obs['anchor'].tolist(),anchor_method=obs['anchor_method'],head_only=obs['head_only']))
        return result
