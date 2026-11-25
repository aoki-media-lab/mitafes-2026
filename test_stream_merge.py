import contextlib,csv,io,json,random,tempfile,unittest
from pathlib import Path
from merge_indexed import merge as indexed,FIELDS
from stream_merge import StreamMerger,KEEP


def scenario(seed,frames=1200):
    """Two branches observing the same walkers; the secondary sometimes sees extra people."""
    rng=random.Random(seed);walkers=[]
    for tid in range(1,25):
        start=rng.randrange(0,frames-120,2);direction=rng.choice([1,-1]);y=rng.uniform(400,1400)
        walkers.append(dict(tid=tid,start=start,direction=direction,y=y,primary=rng.random()>.2,secondary=rng.random()>.2,
                            jitter=rng.uniform(0,40)))
    branches={0:dict(tracks=[],events=[]),1:dict(tracks=[],events=[])}
    for branch in (0,1):
        for frame in range(0,frames,2):
            people=[]
            for w in walkers:
                if not w['primary' if branch==0 else 'secondary']:continue
                t=frame-w['start']
                if not 0<=t<=110:continue
                x=1728+w['direction']*(t-55)*12+(w['jitter'] if branch else 0)
                people.append(dict(id=w['tid'],anchor=[x,w['y']],xyxy=[x-80,w['y']-40,x+80,w['y']+500]))
            branches[branch]['tracks'].append(dict(frame=frame,people=people,umbrellas=[],matches=[]))
        for w in walkers:
            if w['primary' if branch==0 else 'secondary']:
                frame=w['start']+56;branches[branch]['events'].append(
                    dict(event_id=0,direction='ltr' if w['direction']>0 else 'rtl',frame=frame,seconds=round(frame/25,3),
                         track_id=w['tid'],x=1728.,y=w['y']))
        branches[branch]['events'].sort(key=lambda e:e['frame'])
        for i,e in enumerate(branches[branch]['events'],1):e['event_id']=i
    return branches


def write(folder,branch):
    folder.mkdir()
    with (folder/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(branch['events'])
    with (folder/'tracks.jsonl').open('w') as f:
        for row in branch['tracks']:f.write(json.dumps(row)+'\n')


def stream(branches):
    merger=StreamMerger();by_frame=[{},{}]
    for b in (0,1):
        for e in branches[b]['events']:by_frame[b].setdefault(e['frame'],[]).append(e)
    max_buffer=0
    for rf,rr in zip(branches[0]['tracks'],branches[1]['tracks']):
        merger.update(rf['frame'],rf,rr,by_frame[0].get(rf['frame'],[]),by_frame[1].get(rf['frame'],[]))
        max_buffer=max(max_buffer,len(merger.records[0]))
    merger.finish();return merger,max_buffer


class StreamMergeTests(unittest.TestCase):
    def test_streaming_decisions_match_offline_indexed_merge(self):
        for seed in range(8):
            branches=scenario(seed)
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);write(root/'p',branches[0]);write(root/'s',branches[1])
                with contextlib.redirect_stdout(io.StringIO()):offline=indexed(root/'p',root/'s',root/'out')
            merger,_=stream(branches)
            key=lambda e:(e['direction'],int(e['frame']),int(e['track_id']))
            self.assertEqual(sorted(map(key,merger.added)),sorted(map(key,offline['added'])),seed)
            self.assertEqual(merger.suppressed,offline['suppressed'],seed)
            totals=merger.totals()
            self.assertEqual((totals['ltr'],totals['rtl'],totals['events']),(offline['ltr'],offline['rtl'],offline['events']),seed)

    def test_secondary_late_birth_is_added_without_both_sides(self):
        # The roi branch alone saw a person first just past the line (late birth); the
        # side test cannot apply to it. The same track without the rule stays rejected.
        for rule,expected in (('late_birth',1),('',0)):
            people=lambda f:[dict(id=7,anchor=[1650-6*f,600.],xyxy=[1570-6*f,560,1730-6*f,1100])] if f<=60 else []
            branches={0:dict(events=[],tracks=[]),1:dict(tracks=[],events=[dict(event_id=1,direction='rtl',frame=10,
                      seconds=.4,track_id=7,x=1728.,y=600.,rule=rule)])}
            for b in (0,1):branches[b]['tracks']=[dict(frame=f,people=people(f) if b else [],umbrellas=[],matches=[]) for f in range(0,300,2)]
            with tempfile.TemporaryDirectory() as td:
                root=Path(td)
                for name,b in (('p',0),('s',1)):
                    (root/name).mkdir()
                    with (root/name/'events.csv').open('w') as f:
                        w=csv.DictWriter(f,fieldnames=FIELDS+['rule']);w.writeheader();w.writerows(branches[b]['events'])
                    with (root/name/'tracks.jsonl').open('w') as f:
                        for row in branches[b]['tracks']:f.write(json.dumps(row)+'\n')
                with contextlib.redirect_stdout(io.StringIO()):offline=indexed(root/'p',root/'s',root/'out')
            merger,_=stream(branches)
            self.assertEqual((len(offline['added']),len(merger.added)),(expected,expected),rule)

    def test_memory_is_bounded_by_the_decision_window(self):
        merger,max_buffer=stream(scenario(0,frames=6000))
        self.assertLessEqual(max_buffer,KEEP//2+2)

    def test_frames_must_be_strictly_increasing(self):
        merger=StreamMerger();row=dict(people=[])
        merger.update(2,row,row,[],[])
        with self.assertRaises(ValueError):merger.update(2,row,row,[],[])


if __name__=='__main__':unittest.main()
