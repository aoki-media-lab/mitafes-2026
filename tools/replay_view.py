"""Re-run tracking, counting and reconciliation from detections saved by run_fused.py --record-detections.

Detections depend only on the branch crops, so any view with the same crops (line, ROI,
counter / tracker parameters may differ) can be evaluated in seconds without the GPU.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ultralytics.trackers.basetrack import BaseTrack
from engine import StreamingCounter
from views import load_view
from merge_indexed import merge
import csv

def drop_low(item,min_conf):
    """Person detections below min_conf removed (NMS is greedy by score, so this equals a higher NMS conf)."""
    keep=[i for i,d in enumerate(item['detections']) if int(d[5])!=0 or d[4]>=min_conf]
    item['detections']=[item['detections'][i] for i in keep];item['keypoints']=[item['keypoints'][i] for i in keep]
    return item

def replay(cache,view,output,stride=2,min_conf=None):
    output.mkdir(parents=True,exist_ok=True);w,h,fps=view['width'],view['height'],view['fps']
    for b in view['branches']:
        BaseTrack._count=0
        profile=dict(b['profile'],vectorized_tracking=True)
        engine=StreamingCounter(profile,w,h,fps,stride);folder=output/b['name'];folder.mkdir(exist_ok=True);events=[]
        with (cache/b['name']/'detections.jsonl').open() as src,(folder/'tracks.jsonl').open('w') as dst:
            for line in src:
                item=json.loads(line)
                if min_conf:item=drop_low(item,min_conf)
                record,ev,_=engine.update(item);dst.write(json.dumps(record)+'\n');events+=ev
        with (folder/'events.csv').open('w') as f:
            wr=csv.writer(f);wr.writerow(['event_id','direction','frame','seconds','track_id','x','y'])
            wr.writerows([e['event_id'],e['direction'],e['frame'],e['seconds'],e['track_id'],e['x'],e['y']] for e in events)
    return merge(output/'full',output/'roi',output/'result',profile=view['profile'],size=(w,h))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--view',default='plaza')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=2)
    p.add_argument('--set',default='{}',help='JSON profile overrides applied to the view and both branches')
    p.add_argument('--min-conf',type=float,help='Drop person detections below this confidence (detections saved with a lower conf)')
    a=p.parse_args();view=load_view(a.view);extra=json.loads(a.set)
    view['profile'].update(extra)
    for b in view['branches']:b['profile'].update(extra)
    r=replay(a.cache,view,a.output,a.stride,a.min_conf)
    print(json.dumps(dict(rtl=r['rtl'],ltr=r['ltr'],events=r['events'])))
