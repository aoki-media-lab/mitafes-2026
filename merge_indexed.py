"""Reconcile passages with a disk trajectory index and a bounded query cache.

The spatial matching policy is identical to merge_passages.py. The full video
trajectory history is not retained in RAM. Canonical output is published only
after both ordered frame streams have been checked for alignment.
"""
import argparse,bisect,csv,json,sqlite3,tempfile,time
from collections import Counter,defaultdict
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from merge_passages import same_passage

FIELDS=['event_id','direction','frame','seconds','track_id','x','y']


def both_sides_check(profile=None,size=None):
    """Guard for added secondary events: anchors seen beyond the margin on both sides.

    Without a profile line the original glass band (x<1692 and x>1764) is used.
    """
    if not profile or not profile.get('line'):
        return lambda anchors:any(a[0]<1692 for a in anchors) and any(a[0]>1764 for a in anchors)
    from line_counter import CountingLine,crossed_both_sides
    line=CountingLine(profile,*size)
    return lambda anchors:crossed_both_sides(line,profile,anchors)

def events(folder):
    with (folder/'events.csv').open() as f:return list(csv.DictReader(f))


def merge(primary,secondary,output,profile=None,size=None):
    side_check=both_sides_check(profile,size)
    pm=(profile or {}).get('passage_match')
    same=lambda a,pa,b,pb:same_passage(a,pa,b,pb,pm)
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    aa,bb=events(primary),events(secondary)
    handle=tempfile.NamedTemporaryFile(prefix='merge_',suffix='.sqlite',dir=output,delete=False);scratch=Path(handle.name);handle.close()
    db=sqlite3.connect(scratch)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-8192')
    db.execute('CREATE TABLE obs(branch INTEGER,tid INTEGER,frame INTEGER,payload TEXT,PRIMARY KEY(branch,tid,frame)) WITHOUT ROWID')
    temp_events=output/'events.csv.tmp';temp_tracks=output/'tracks.jsonl.tmp'
    try:
        for branch,folder,ev in [(0,primary,aa),(1,secondary,bb)]:
            event_frames=defaultdict(list)
            for e in ev:event_frames[int(e['track_id'])].append(int(e['frame']))
            for values in event_frames.values():values.sort()
            pending=[]
            with (folder/'tracks.jsonl').open() as src:
                previous=-1
                for line in src:
                    row=json.loads(line);frame=row['frame']
                    if frame<=previous:raise ValueError('Track frames must be strictly ordered')
                    previous=frame
                    for p in row['people']:
                        frames=event_frames.get(p['id'])
                        if not frames:continue
                        i=bisect.bisect_left(frames,frame-100)
                        # Matching uses event midpoints +/-60 frames and permits
                        # event times to differ by <=75 frames at calibrated 25fps.
                        if i==len(frames) or frames[i]>frame+100:continue
                        pending.append((branch,p['id'],frame,json.dumps(p)))
                    if len(pending)>=1000:db.executemany('INSERT INTO obs VALUES(?,?,?,?)',pending);pending=[]
                if pending:db.executemany('INSERT INTO obs VALUES(?,?,?,?)',pending)
        db.commit()

        @lru_cache(maxsize=128)
        def history(branch,tid,frame):
            return {f:json.loads(payload) for f,payload in db.execute('SELECT frame,payload FROM obs WHERE branch=? AND tid=? AND frame BETWEEN ? AND ?',(branch,tid,frame-100,frame+100))}

        for a in aa:
            if not history(0,int(a['track_id']),int(a['frame'])):raise ValueError('Primary event has no observed trajectory')
        duplicates=0;witnesses=aa   # secondary events are matched against every primary passage, kept or not
        if profile and profile.get('dedup_passages'):
            # Two primary tracks on one person (e.g. two boxes for one blurred person)
            # count once: an event is dropped when an earlier kept one is the same passage.
            kept=[]
            for a in sorted(aa,key=lambda e:(int(e['frame']),int(e['event_id']))):
                pa=history(0,int(a['track_id']),int(a['frame']))
                if any(same(k,history(0,int(k['track_id']),int(k['frame'])),a,pa) for k in kept if float(a['seconds'])-float(k['seconds'])<=3):
                    duplicates+=1;continue
                kept.append(a)
            aa=kept
        by_time=sorted(witnesses,key=lambda e:float(e['seconds']));times=[float(e['seconds']) for e in by_time]
        offset=max((int(e['track_id']) for e in witnesses),default=0)+1;added=[];suppressed=0   # all primary IDs
        for b in bb:
            pb=history(1,int(b['track_id']),int(b['frame']));seconds=float(b['seconds'])
            if not pb:raise ValueError('Secondary event has no observed trajectory')
            nearby=by_time[bisect.bisect_left(times,seconds-3):bisect.bisect_right(times,seconds+3)]
            if any(same(a,history(0,int(a['track_id']),int(a['frame'])),b,pb) for a in nearby):suppressed+=1;continue
            if profile and profile.get('dedup_passages') and any(abs(seconds-float(x['seconds']))<=3 and same(x,history(1,int(x['track_id']),int(x['frame'])),b,pb) for x in added):
                duplicates+=1;continue
            anchors=[p['anchor'] for f,p in pb.items() if abs(f-int(b['frame']))<=75]
            if side_check(anchors):added.append(b)
        ev=[dict(e) for e in aa]+[dict(e,track_id=int(e['track_id'])+offset) for e in added]
        ev.sort(key=lambda e:float(e['seconds']))
        for i,e in enumerate(ev,1):e['event_id']=i
        with temp_events.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(ev)
        added_ids={int(e['track_id']) for e in added}
        with (primary/'tracks.jsonl').open() as p,(secondary/'tracks.jsonl').open() as s,temp_tracks.open('w') as dst:
            for pl,sl in zip_longest(p,s):
                if pl is None or sl is None:raise ValueError('Both streams must contain the same observed frames')
                pr,sr=json.loads(pl),json.loads(sl)
                if pr['frame']!=sr['frame']:raise ValueError('Secondary frames must align with the primary stream')
                pr['people']+=[dict(x,id=x['id']+offset) for x in sr['people'] if x['id'] in added_ids]
                dst.write(json.dumps(pr)+'\n')
        temp_events.replace(output/'events.csv');temp_tracks.replace(output/'tracks.jsonl')
        c=Counter(e['direction'] for e in ev)
        result=dict(ltr=c['ltr'],rtl=c['rtl'],events=len(ev),primary=str(primary),secondary=str(secondary),added=added,suppressed=suppressed,duplicates=duplicates,
                    reconciliation=dict(method='SQLite trajectory index',query_cache_limit=128,sqlite_cache_mib=8,index_bytes=scratch.stat().st_size,seconds=time.perf_counter()-started))
        (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result
    finally:
        db.close();scratch.unlink(missing_ok=True);temp_events.unlink(missing_ok=True);temp_tracks.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--secondary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();merge(a.primary,a.secondary,a.output)
