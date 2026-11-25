"""Re-count a finished live/fused run from its saved trajectories (no video, no inference).

Each branch's tracks.jsonl holds every tracked person per observation (after the tracker and the
quality filter), which is exactly what the engine feeds to its line counter. Replaying those
positions through the same counter classes with other counting directions (e.g. both, for runs
recorded when only right-to-left was kept) gives the events the live run would have produced,
then the two branches are reconciled with merge_indexed.py as in the live run.

With the run's own directions the replay reproduces its events.csv exactly (checked with --verify).
Output: <output>/{full,roi}/events.csv (+ tracks.jsonl symlinks), <output>/result/, summary.json.
"""
import argparse,csv,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from engine import make_people_counter
from camera_counter import foot
from run_fused import line_band
from merge_indexed import merge

W,FPS,STRIDE=2880,25.0,2
FIELDS=['event_id','direction','frame','seconds','track_id','x','y','rule']


def replay(tracks,profile,directions,max_frame=None):
    """Same feed as StreamingCounter.update for the people counter (positions, size, tick, state retirement)."""
    counter=make_people_counter(profile,W,FPS);events=[]
    with open(tracks) as f:
        for line in f:
            row=json.loads(line);frame=row['frame']
            if max_frame is not None and frame>max_frame:break
            for person in row['people']:
                x,y=person.get('anchor',foot(person['xyxy']))
                counter.update(person['id'],x,y,frame)
                st=counter.states.get(person['id'])
                if st is not None:
                    st['h']=person['xyxy'][3]-person['xyxy'][1]
                    st['conf']=max(st.get('conf',0.),person.get('confidence',1.))
            if hasattr(counter,'tick'):counter.tick(frame)
            for e in counter.events:
                if e['direction'] in directions:events.append(e)
            counter.events.clear()
            if frame%round(FPS)<STRIDE:
                cutoff=frame-round(FPS*10)
                counter.states={t:s for t,s in counter.states.items() if s['last_frame']>=cutoff}
    for i,e in enumerate(events,1):e['event_id']=i
    return events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Run folder with full/ and roi/ (tracks.jsonl, summary.json)')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--directions',nargs='+',default=['ltr','rtl'],choices=['ltr','rtl'])
    p.add_argument('--override',default='{}',help='JSON merged into both branch profiles (counter parameters only)')
    p.add_argument('--max-seconds',type=float,help='Only the first N seconds of the run')
    p.add_argument('--verify',action='store_true',help='Check that the run\'s own directions reproduce its events.csv')
    a=p.parse_args();a.run=a.run.resolve();a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an unused output directory')
    started=time.time();override=json.loads(a.override);check={}
    max_frame=round(a.max_seconds*FPS) if a.max_seconds else None
    for name in ['full','roi']:
        src=a.run/name;dst=a.output/name;dst.mkdir(parents=True)
        profile=json.loads((src/'summary.json').read_text())['profile']|override
        own=set(profile.get('count_directions',['ltr','rtl']))
        events=replay(src/'tracks.jsonl',profile,set(a.directions),max_frame)
        if a.verify:
            with (src/'events.csv').open() as f:old=[(r['direction'],int(r['frame']),int(r['track_id'])) for r in csv.DictReader(f)]
            if max_frame is not None:old=[o for o in old if o[1]<=max_frame]
            new=[(e['direction'],int(e['frame']),int(e['track_id'])) for e in events if e['direction'] in own]
            check[name]=dict(saved=len(old),replayed=len(new),identical=sorted(old)==sorted(new))
        with (dst/'events.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore',restval='');w.writeheader()
            w.writerows(dict(e,rule=e.get('rule','')) for e in events)
        (dst/'tracks.jsonl').symlink_to(src/'tracks.jsonl')
        (dst/'summary.json').write_text(json.dumps(dict(profile=profile|dict(count_directions=a.directions),source_run=str(a.run),
                                                       counts=dict(Counter(e['direction'] for e in events))),indent=1)+'\n')
        print(json.dumps(dict(branch=name,counts=dict(Counter(e['direction'] for e in events)),**({'verify':check[name]} if a.verify else {}))),flush=True)
    profile=json.loads((a.run/'full'/'summary.json').read_text())['profile']|override
    result=merge(a.output/'full',a.output/'roi',a.output/'result',line_band(profile,W))
    summary=dict(run=str(a.run),directions=a.directions,override=override,max_seconds=a.max_seconds,
                 ltr=result['ltr'],rtl=result['rtl'],events=result['events'],added=len(result['added']),
                 verify=check or None,seconds=round(time.time()-started,1))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
