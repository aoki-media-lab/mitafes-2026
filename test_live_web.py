"""Day totals, in-out difference and chart data of the browser view (no camera, no GPU)."""
import json,tempfile,time,unittest
from pathlib import Path
from live_web import LiveWeb,day_history,run_directions

PROFILE=dict(line_x=.5,margin=.0125,roi_mode='polygon',roi_polygon=[[0,0],[1,0],[1,1],[0,1]])
BOTH=PROFILE|dict(count_directions=['rtl','ltr'],in_direction='rtl',direction_labels=dict(rtl='入場',ltr='退場'))


def run(root,name,directions,events):
    d=Path(root)/name;d.mkdir()
    (d/'setup.json').write_text(json.dumps(dict(profile=dict(count_directions=directions))))
    t0=time.mktime(time.strptime(name,'%Y%m%d_%H%M%S'))
    with (d/'live_events.jsonl').open('w') as f:
        for dt,direction in events:f.write(json.dumps(dict(direction=direction,clock=t0+dt))+'\n')
    return d


def web(profile,**kw):
    w=LiveWeb('127.0.0.1',0,2880,1620,profile,[0,0,10,10],**kw);w.close();return w


class LiveWebTests(unittest.TestCase):
    def test_day_history_narrows_covered_directions(self):
        with tempfile.TemporaryDirectory() as root:
            run(root,'20261122_060000',['rtl'],[(10,'rtl'),(20,'rtl')])
            run(root,'20261122_120000',['rtl','ltr'],[(10,'rtl'),(15,'ltr')])
            covered={'rtl','ltr'}
            h=day_history(root,{'rtl','ltr'},date='20261122',covered=covered)
            self.assertEqual(sorted(d for _,d in h),['ltr','rtl','rtl','rtl'])
            self.assertEqual(covered,{'rtl'})
            covered={'rtl','ltr'}
            day_history(root,{'rtl','ltr'},date='20261123',covered=covered)
            self.assertEqual(covered,{'rtl','ltr'})          # no runs that day: nothing narrows it

    def test_recount_window_covering_a_run_supplies_its_directions(self):
        with tempfile.TemporaryDirectory() as root:
            run(root,'20261122_060000',['rtl'],[(10,'rtl'),(20,'rtl')])
            t0=time.mktime(time.strptime('20261122_060000','%Y%m%d_%H%M%S'))
            d=Path(root)/'20261122_recount_both';d.mkdir()
            (d/'window.json').write_text(json.dumps(dict(start=t0,end=t0+3600,directions=['rtl','ltr'])))
            with (d/'live_events.jsonl').open('w') as f:
                for dt,direction in [(10,'rtl'),(20,'rtl'),(15,'ltr')]:f.write(json.dumps(dict(direction=direction,clock=t0+dt))+'\n')
            covered={'rtl','ltr'}
            h=day_history(root,{'rtl','ltr'},date='20261122',covered=covered)
            self.assertEqual(sorted(d for _,d in h),['ltr','rtl','rtl']);self.assertEqual(covered,{'rtl','ltr'})

    def test_recount_window_directions(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)/'20261121_recount_1300-1400';d.mkdir()
            (d/'window.json').write_text(json.dumps(dict(start=0,end=1,directions=['rtl'])))
            self.assertEqual(run_directions(d),{'rtl'})
            self.assertEqual(run_directions(Path(root)),{'rtl','ltr'})

    def test_in_out_difference_only_when_out_counted_all_day(self):
        t=time.time()
        w=web(BOTH,history=[(t,'rtl'),(t,'rtl'),(t,'ltr')],covered={'rtl','ltr'})
        w.counts=dict(rtl=5,ltr=1)
        day=w.day_counts()
        self.assertEqual((day['rtl'],day['ltr'],day['net']),(7,2,5))
        self.assertEqual(w.directions,['rtl','ltr'])
        self.assertEqual([t for t,_ in w.total_lines()],['IN  7','OUT  2'])
        hist=w.status()['hist']
        self.assertEqual(hist['dirs'],['rtl','ltr']);self.assertEqual(sum(b[1] for b in hist['bins']),2)
        partial=web(BOTH,history=[(t,'rtl')],covered={'rtl'})
        self.assertIsNone(partial.day_counts()['net'])
        self.assertEqual(partial.status()['hist']['dirs'],['rtl'])

    def test_single_direction_keeps_old_behaviour(self):
        t=time.time()
        w=web(PROFILE|dict(count_directions=['rtl']),history=[(t,'rtl')])
        w.counts=dict(rtl=3,ltr=0)
        self.assertEqual(w.day_counts()['rtl'],4);self.assertIsNone(w.day_counts()['net'])
        self.assertEqual(w.total_lines(),[('right->left  4',(77,183,255))])


if __name__=='__main__':unittest.main()


class ScheduleTests(unittest.TestCase):
    def test_time_of_day_overrides_restore_branch_values(self):
        from run_fused import scheduled_overrides,schedule_keys,apply_overrides
        setup=dict(profile={},schedule=[{'from':'16:00','profile':{'pose_new_track_thresh':.05}},{'from':'06:00','profile':{'rule_min_confidence':.3}}])
        at=lambda hm:time.mktime(time.strptime('20261123'+hm,'%Y%m%d%H:%M'))
        full=dict(pose_new_track_thresh=.2);base={k:full.get(k) for k in schedule_keys(setup)}
        apply_overrides(full,base,scheduled_overrides(setup,at('17:00')))
        self.assertEqual(full,dict(pose_new_track_thresh=.05,rule_min_confidence=.3))
        apply_overrides(full,base,scheduled_overrides(setup,at('05:00')))
        self.assertEqual(full,dict(pose_new_track_thresh=.2))


class RuleConfidenceTests(unittest.TestCase):
    def test_faint_track_cannot_be_counted_by_late_death(self):
        import engine  # noqa: F401  (puts reference/ on sys.path for camera_counter)
        from stitch_counter import StitchCounter
        prof=dict(line_x=.5,margin=.0125,counter_max_gap_seconds=.24,counter_stale_seconds=2.2,rule_min_confidence=.3)
        counts=[]
        for conf in [.2,.5]:
            c=StitchCounter(prof,2880,25.)
            for i,f in enumerate(range(0,40,2)):            # walks right-to-left and vanishes at the line
                c.update(1,1700-i*14,500,f);c.states[1]['conf']=conf;c.tick(f)
            for f in range(40,120,2):c.tick(f)
            counts.append(len(c.events))
        self.assertEqual(counts,[0,1])
