import json,os,tempfile,unittest
from unittest import mock
from live_web import LiveState


class LiveStateTest(unittest.TestCase):
    def test_day_totals_carry_over_and_reset_at_midnight(self):
        with tempfile.TemporaryDirectory() as d:
            a=LiveState('plaza',daily_dir=d);path=os.path.join(d,'%s.json'%LiveState.today())
            a.add_events([dict(direction='rtl',seconds=1.0)]*3+[dict(direction='ltr',seconds=2.0)],'full')
            a.set_counts(dict(rtl=3,ltr=1,events=4));a.save(force=True)
            b=LiveState('plaza',daily_dir=d)      # restart on the same day
            b.set_counts(dict(rtl=2,ltr=0,events=2))
            s=b.as_dict();self.assertEqual((s['counts']['rtl'],s['counts']['ltr'],s['counts']['events']),(5,1,6))
            self.assertTrue(s['carried_over']);self.assertEqual(sum(x['rtl'] for x in s['bins']),3)
            with mock.patch.object(LiveState,'today',staticmethod(lambda:'2099-01-01')):
                b.save(force=True)                      # date changed: start from zero
                self.assertEqual(b.as_dict()['counts']['events'],0)
                b.set_counts(dict(rtl=3,ltr=0,events=3));b.save(force=True)   # one passage after midnight
                self.assertEqual(b.as_dict()['counts']['rtl'],1)
            self.assertEqual(json.load(open(path))['rtl'],5)                 # the old day is kept
            self.assertEqual(json.load(open(os.path.join(d,'2099-01-01.json')))['rtl'],1)
            self.assertEqual([x['date'] for x in b.days()],['2099-01-01',LiveState.today()])
            c=LiveState('plaza',daily_dir=d)            # today's file only: 2099 is not today
            self.assertTrue(c.as_dict()['carried_over']);self.assertEqual(c.as_dict()['counts']['rtl'],5)

    def test_entry_direction_is_published(self):
        self.assertIsNone(LiveState('glass').as_dict()['entry'])
        self.assertEqual(LiveState('plaza',entry_direction='rtl').as_dict()['entry'],'rtl')
        with self.assertRaises(ValueError):LiveState('plaza',entry_direction='in')

    def test_plaza_view_enters_right_to_left(self):
        from views import load_view
        self.assertEqual(load_view('plaza')['entry_direction'],'rtl')


if __name__=='__main__':unittest.main()
