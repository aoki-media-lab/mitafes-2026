import json,random,unittest
from pathlib import Path
from line_counter import CountingLine,LineCounter,LineOcclusionCounter,crossed_both_sides
from camera_counter import TemporalCounter
from occlusion_counter import OcclusionCounter
from views import load_view

ROOT=Path(__file__).resolve().parent
GLASS=json.loads((ROOT/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())


def walk(seed,n=400):
    rng=random.Random(seed);out=[];x=rng.uniform(1200,2200);y=rng.uniform(300,1300);f=0
    for _ in range(n):
        x+=rng.gauss(0,25);y+=rng.gauss(0,10);f+=rng.choice([2,2,2,4,10]);out.append((x,y,f))
    return out


class LineCounterTest(unittest.TestCase):
    def test_vertical_line_matches_original_counters(self):
        for seed in range(20):
            for old_cls,new_cls in [(TemporalCounter,LineCounter),(OcclusionCounter,LineOcclusionCounter)]:
                old=old_cls(GLASS,2880,25);new=new_cls(GLASS,2880,25,1620)
                for x,y,f in walk(seed):old.update(1,x,y,f);new.update(1,x,y,f)
                self.assertEqual(old.events,new.events)
                self.assertEqual(old.pending(10**6,10**6),new.pending(10**6,10**6))

    def test_diagonal_line_direction_and_extent(self):
        view=load_view('plaza');p=view['profile'];line=CountingLine(p,2880,1620)
        # Lower right -> upper left across the middle of the segment is rtl.
        c=LineCounter(p,2880,25,1620)
        for i,t in enumerate(range(0,40)):
            c.update(1,1900-30*t,1200-17*t,2*i)
        self.assertEqual([e['direction'] for e in c.events],['rtl'])
        c=LineCounter(p,2880,25,1620)
        for i,t in enumerate(range(0,40)):c.update(1,700+30*t,500+17*t,2*i)
        self.assertEqual([e['direction'] for e in c.events],['ltr'])
        # With line_extent, a crossing beyond the end of the drawn segment is not counted.
        short=dict(p,line=[[0.5,0.3],[0.5,0.7]],line_extent=0.0)
        c=LineCounter(short,2880,25,1620)
        for i,t in enumerate(range(0,40)):c.update(1,1000+30*t,200,2*i)
        self.assertEqual(c.events,[])
        c=LineCounter(short,2880,25,1620)
        for i,t in enumerate(range(0,40)):c.update(1,1000+30*t,800,2*i)
        self.assertEqual([e['direction'] for e in c.events],['ltr'])

    def test_lost_track_hands_side_to_its_continuation(self):
        p=dict(load_view('plaza')['profile'])
        def run(tracks):
            c=LineCounter(p,2880,25,1620)
            for frame,people in tracks:
                c.link(frame,people)
                for tid,x,y in people:c.update(tid,x,y,frame)
            return c.events
        # Track 1 is confirmed on the right, then a new ID continues on the left.
        right=[(f,[(1,1300-20*i,900)]) for i,f in enumerate(range(0,12,2))]   # line is at x~1147 for y=900
        left=[(f,[(2,1080-20*i,900)]) for i,f in enumerate(range(14,26,2))]
        self.assertEqual([e['direction'] for e in run(right+left)],['rtl'])
        self.assertEqual(run(right+[(f,[(2,1080-20*i,900)]) for i,f in enumerate(range(60,72,2))]),[])  # too late
        # A duplicate box beside a live track does not inherit a side.
        dup=right+[(f,[(1,1110-20*i,900),(2,1080-20*i,900)]) for i,f in enumerate(range(14,26,2))]
        self.assertEqual([e["track_id"] for e in run(dup)],[1])   # the live track counts once, the duplicate adds nothing
        # A lost track last seen beyond the line on the other side passes nothing on.
        stale=right+[(12,[(1,1000,900)])]+[(f,[(2,980-20*i,900)]) for i,f in enumerate(range(16,28,2))]
        self.assertEqual([e['direction'] for e in run(stale)],[])
        line=CountingLine(p,2880,1620)
        self.assertTrue(crossed_both_sides(line,p,[(1900,1200),(700,500)]))
        self.assertFalse(crossed_both_sides(line,p,[(1900,1200),(1800,1100)]))

    def test_birth_at_the_line_counts_in_walking_direction(self):
        p=dict(load_view('plaza')['profile'])
        def run(people_by_frame):
            c=LineOcclusionCounter(p,2880,25,1620)
            for frame,people in people_by_frame:
                c.link(frame,people)
                for tid,x,y in people:c.update(tid,x,y,frame)
            return c.events
        # First seen on the line (x~1147 at y=900, hidden before) and walking right: ltr.
        walk=[(f,[(1,1150+25*i,900)]) for i,f in enumerate(range(0,20,2))]
        self.assertEqual([e['direction'] for e in run(walk)],['ltr'])
        # The same birth right next to another live track is a duplicate box: nothing.
        dup=[(f,[(2,1140+25*i,905),(1,1150+25*i,900)]) for i,f in enumerate(range(0,20,2))]
        self.assertEqual([e['track_id'] for e in run(dup)],[])
        # Born far from the line: not a rescue.
        far=[(f,[(1,1450+25*i,900)]) for i,f in enumerate(range(0,20,2))]
        self.assertEqual(run(far),[])

    def test_glass_view_reproduces_original_preset(self):
        from run_fused import BRANCHES
        from run_accuracy import FULL_PROFILE,ROI_PROFILE
        old=[('full',(736,1280),(0,0,1,1),dict(GLASS,**FULL_PROFILE)),('roi',(736,512),(.4,0,.8,1),dict(GLASS,**ROI_PROFILE))]
        self.assertEqual(BRANCHES,old)


if __name__=='__main__':unittest.main()
