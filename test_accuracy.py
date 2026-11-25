import json,pickle,unittest
from pathlib import Path
import numpy as np
from engine import StreamingCounter
from occlusion_counter import OcclusionCounter
from late_birth_counter import LateBirthCounter
from stitch_counter import StitchCounter
from robust_tracker import RobustPoseAnchorTracker
from merge_passages import same_passage

PROFILE=json.loads((Path(__file__).parent/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())

class AccuracyTests(unittest.TestCase):
    def test_near_camera_fast_motion_keeps_identity(self):
        tracker=RobustPoseAnchorTracker(25,PROFILE)
        ids=[]
        for frame,x in enumerate([1100,1280,1460,1640,1820]):
            k=np.zeros((17,3));k[:5]=[x,1500,.9]
            people=tracker.update([[x-100,1420,x+100,1620,.9,0]],[k],frame*2)
            ids.append(people[0]['id'])
        self.assertEqual(len(set(ids)),1)

    def test_late_visible_person_requires_observations_on_both_sides(self):
        c=OcclusionCounter(PROFILE,2880,25)
        for frame,x in [(0,1740),(2,1705),(4,1660),(6,1610)]:c.update(1,x,600,frame)
        self.assertEqual([e['direction'] for e in c.events],['rtl'])
        for frame,x in [(8,1570),(10,1520)]:c.update(1,x,600,frame)
        self.assertEqual(len(c.events),1)

    def test_late_birth_on_one_side_does_not_invent_crossing(self):
        c=OcclusionCounter(PROFILE,2880,25)
        for frame,x in [(0,1710),(2,1670),(4,1620),(6,1570)]:c.update(1,x,600,frame)
        self.assertEqual(c.events,[])

    def test_short_jitter_and_large_vertical_jump_do_not_count(self):
        for values in [[(1740,600),(1710,605),(1690,600),(1750,600)],[(1740,600),(1705,900),(1650,950),(1600,1000)]]:
            c=OcclusionCounter(PROFILE,2880,25)
            for i,(x,y) in enumerate(values):c.update(1,x,y,2*i)
            self.assertEqual(c.events,[])

    def test_occluded_counter_checkpoint_preserves_pending_evidence(self):
        c=OcclusionCounter(PROFILE,2880,25);c.update(1,1740,600,0);c.update(1,1705,600,2)
        c=pickle.loads(pickle.dumps(c));c.update(1,1660,600,4);c.update(1,1610,600,6)
        self.assertEqual(len(c.events),1)

    def test_late_birth_counts_person_first_seen_past_the_line(self):
        c=LateBirthCounter(PROFILE|dict(late_birth_crossings=True),2880,25)
        for j in range(9):c.update(2,1650-15*j,600,2*j)
        self.assertEqual([(e['direction'],e['rule']) for e in c.events],[('rtl','late_birth')])

    def test_late_birth_ignores_second_track_of_already_counted_person(self):
        # Track 2 appears on top of counted track 1, which is then lost before the
        # late-birth decision: it is the same person and must not count again.
        c=LateBirthCounter(PROFILE|dict(late_birth_crossings=True),2880,25)
        updates=[(2*i,1,1950-30*i) for i in range(12)]+[(18+2*j,2,1660-15*j) for j in range(9)]
        for frame,tid,x in sorted(updates):c.update(tid,x,600,frame)
        self.assertEqual([e['track_id'] for e in c.events],[1])

    def stitch(self):
        c=StitchCounter(PROFILE|dict(late_birth_crossings=True,stitch_crossings=True),2880,25)
        return c

    def feed(self,c,updates,until):
        by={}
        for f,tid,x in updates:by.setdefault(f,[]).append((tid,x))
        for f in range(0,until,2):
            for tid,x in by.get(f,[]):c.update(tid,x,600,f)
            c.tick(f)
        return [(e['track_id'],e.get('rule')) for e in c.events]

    # PROFILE: line at 1728 px, margin 36 px.
    def test_stitch_counts_track_broken_at_the_line_once(self):
        # A slow walker (150 px/s) is lost 164 px before the line and reappears under a new ID
        # 52 px past it 1.44 s later, where its motion predicts. Too slow for late_birth
        # (travel), too far from the line for late_death: only stitching counts it, once.
        updates=[(2*i,1,2000-12*i) for i in range(10)]+[(54+2*j,2,1676-6*2*j) for j in range(15)]
        self.assertEqual(self.feed(self.stitch(),updates,140),[(2,'stitch')])

    def test_late_death_counts_track_lost_at_the_line(self):
        updates=[(2*i,1,1900-20*i) for i in range(9)]           # last seen at 1740 (margin band), moving left
        self.assertEqual(self.feed(self.stitch(),updates,120),[(1,'late_death')])

    def test_track_that_stops_short_of_the_line_is_not_counted(self):
        updates=[(2*i,1,2200-20*i) for i in range(9)]           # lost at 2040, ~0.11 width before the line
        self.assertEqual(self.feed(self.stitch(),updates,120),[])

    def test_companion_beside_a_counted_person_is_counted(self):
        # Two people walking side by side (heads 56 px apart, boxes 440 px tall); the second is
        # first detected past the line. A fixed 60 px duplicate radius would merge them.
        c=self.stitch()
        for f in range(0,120,2):
            x=1950-12.5*f
            c.update(1,x,600,f);c.states[1]['h']=440
            if f>=24:c.update(2,x-10,655,f);c.states[2]['h']=440
            c.tick(f)
        self.assertEqual(sorted(e['track_id'] for e in c.events),[1,2])

    def test_passages_are_matched_by_space_and_direction(self):
        a=dict(direction='ltr',seconds=2,frame=50)
        p={f:dict(anchor=[1600+f*3,600],xyxy=[1500,550,1800,1100]) for f in range(40,60,2)}
        q={f:dict(anchor=[1600+f*3,850],xyxy=[1500,800,1800,1400]) for f in range(40,60,2)}
        self.assertTrue(same_passage(a,p,a,p))
        self.assertFalse(same_passage(a,p,a,q))
        self.assertFalse(same_passage(a,p,a|dict(direction='rtl'),p))

if __name__=='__main__':unittest.main()
