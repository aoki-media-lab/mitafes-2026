import datetime,unittest
from live_service import parse_hours,window_at,next_start


class ScheduleTest(unittest.TestCase):
    def test_hours_and_next_start(self):
        h=parse_hours('6-20');self.assertEqual(h,(6,20))
        day=datetime.datetime(2026,11,21)
        s,e=window_at(day.replace(hour=12),h);self.assertEqual((s.hour,e.hour),(6,20))
        self.assertEqual(next_start(day.replace(hour=1),h),day.replace(hour=6))            # before: today 6:00
        self.assertEqual(next_start(day.replace(hour=12),h),day.replace(hour=12))          # inside: now
        self.assertEqual(next_start(day.replace(hour=20),h),day.replace(hour=6)+datetime.timedelta(days=1))  # after: tomorrow
        for bad in ['20-6','6-25','x']:
            with self.assertRaises(ValueError):parse_hours(bad)


if __name__=='__main__':unittest.main()
