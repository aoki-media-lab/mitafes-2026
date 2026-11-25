"""Combine independent trajectory branches using observed spatial identity.

This utility reads only counter outputs. No reference counts or annotations.
It preserves the primary branch, and suppresses corroborating secondary events.
"""
import argparse,csv,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from fusion import box_iou

def read_run(folder):
    with (folder/'events.csv').open() as f:events=list(csv.DictReader(f))
    ids={int(e['track_id']) for e in events};tracks=defaultdict(dict)
    with (folder/'tracks.jsonl').open() as f:
        for line in f:
            d=json.loads(line)
            for p in d['people']:
                if p['id'] in ids:tracks[p['id']][d['frame']]=p
    return events,tracks

PASSAGE_MATCH=dict(iou=.75,dist_fraction=.1,dist_min=25,dist_max=75,share=.5)


def same_passage(a,pa,b,pb,match=None):
    """Two counted tracks are one passage when, in >=3 shared frames around the events and in at
    least ``share`` of them, the boxes overlap by more than ``iou`` or the head anchors are closer
    than dist_fraction x body height (clamped to dist_min..dist_max px). ``match`` overrides
    PASSAGE_MATCH (profile key passage_match)."""
    m=PASSAGE_MATCH if not match else dict(PASSAGE_MATCH,**match)
    if a['direction']!=b['direction'] or abs(float(a['seconds'])-float(b['seconds']))>3:return False
    center=(int(a['frame'])+int(b['frame']))/2
    frames=[f for f in pa.keys()&pb.keys() if abs(f-center)<=60]
    if len(frames)<3:return False
    evidence=[]
    for f in frames:
        p,q=pa[f],pb[f];dist=np.linalg.norm(np.asarray(p['anchor'])-q['anchor'])
        height=min(p['xyxy'][3]-p['xyxy'][1],q['xyxy'][3]-q['xyxy'][1])
        evidence.append(box_iou(p['xyxy'],q['xyxy'])>m['iou'] or dist<max(m['dist_min'],min(m['dist_max'],m['dist_fraction']*height)))
    return sum(evidence)>=3 and np.mean(evidence)>=m['share']

def merge(primary,secondary,output):
    aa,ta=read_run(primary);bb,tb=read_run(secondary);output.mkdir(parents=True,exist_ok=True)
    offset=max(ta,default=0)+1;added=[];suppressed=[]
    for b in bb:
        pb=tb[int(b['track_id'])]
        matches=[a for a in aa if same_passage(a,ta[int(a['track_id'])],b,pb)]
        if matches:suppressed.append(dict(secondary=b,matches=[a['event_id'] for a in matches]));continue
        # Counter fallbacks cannot create an event from an implausible jump.
        xs=[p['anchor'][0] for f,p in pb.items() if abs(f-int(b['frame']))<=75]
        if not(any(x<1692 for x in xs) and any(x>1764 for x in xs)):continue
        added.append(b)
    ev=[dict(e) for e in aa]+[dict(e,track_id=int(e['track_id'])+offset) for e in added]
    ev.sort(key=lambda e:float(e['seconds']))
    for i,e in enumerate(ev,1):e['event_id']=i
    with (output/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['event_id','direction','frame','seconds','track_id','x','y']);w.writeheader();w.writerows(ev)
    added_ids={int(e['track_id']) for e in added}
    with (primary/'tracks.jsonl').open() as p,(secondary/'tracks.jsonl').open() as s,(output/'tracks.jsonl').open('w') as dst:
        secondary_rows={}
        for line in s:
            d=json.loads(line);people=[dict(x,id=x['id']+offset) for x in d['people'] if x['id'] in added_ids]
            if people:secondary_rows[d['frame']]=people
        for line in p:
            d=json.loads(line);d['people']+=secondary_rows.pop(d['frame'],[]);dst.write(json.dumps(d)+'\n')
        if secondary_rows:raise ValueError('Secondary frames must align with primary stream')
    c=Counter(e['direction'] for e in ev)
    result=dict(ltr=c['ltr'],rtl=c['rtl'],events=len(ev),primary=str(primary),secondary=str(secondary),added=added,suppressed=len(suppressed))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--secondary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();merge(a.primary,a.secondary,a.output)
