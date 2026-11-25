import json,subprocess,tempfile,time,unittest
from pathlib import Path
import numpy as np
from views import load_view
from live_recorder import LiveRecorder,ROI_ID_OFFSET


def person(tid,x,y):
    return dict(id=tid,xyxy=[x-60,y-40,x+60,y+300],anchor=[x,y],confidence=.9)


class LiveRecorderTest(unittest.TestCase):
    def run_recorder(self,frames,events,**kw):
        view=load_view('plaza');folder=Path(tempfile.mkdtemp())
        rec=LiveRecorder(view,folder,size=(640,360),delay=1.,**kw);rec.start()
        image=np.zeros((1620,2880,3),np.uint8);wall=time.time()
        for start in range(0,frames,8):
            items=[(2*i,wall+2*i/25,image,[person(1,1500-30*i,800)],[person(7,1400,700)]) for i in range(start,min(frames,start+8))]
            for e in events:
                if start<=e['frame']//2<start+8:rec.add_events([e],e.pop('source','full'))
            rec.put(items)
            while rec.frames.qsize():time.sleep(.01)   # the test feeds as fast as the recorder takes it
        rec.close()
        return rec,folder

    def count_frames(self,path):
        out=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(path)],
                           capture_output=True,text=True,check=True).stdout
        return int(out.strip())

    def test_all_frames_written_and_delayed_events_counted(self):
        events=[dict(direction='rtl',frame=20,track_id=1,x=1100,y=800,seconds=.8),
                dict(direction='rtl',frame=30,track_id=7,x=1100,y=700,seconds=1.2,source='roi'),
                dict(direction='ltr',frame=40,track_id=9,x=1500,y=700,seconds=1.6)]
        rec,folder=self.run_recorder(60,events)
        self.assertIsNone(rec.stats['failed'])
        self.assertEqual(rec.stats['recorded_frames'],60);self.assertEqual(len(rec.segments),1)
        self.assertEqual(self.count_frames(rec.segments[0]),60)
        self.assertEqual(rec.overlay.counts,{'rtl':2})          # ltr is not shown in the default video
        self.assertIn(7+ROI_ID_OFFSET,rec.added_roi)

    def test_late_roi_event_inside_delay_is_drawn(self):
        # A roi count confirmed after its frame (as the live reconciliation does) still appears.
        rec,folder=self.run_recorder(40,[dict(direction='rtl',frame=60,track_id=7,x=1100,y=700,seconds=2.4,source='roi')])
        self.assertEqual(rec.overlay.counts,{'rtl':1})

    def test_put_never_blocks_when_queue_full(self):
        view=load_view('plaza');rec=LiveRecorder(view,Path(tempfile.mkdtemp()),size=(640,360))   # not started
        image=np.zeros((1620,2880,3),np.uint8)
        t=time.perf_counter()
        for i in range(10):rec.put([(2*i,time.time(),image,[],[])])
        self.assertLess(time.perf_counter()-t,.5)
        self.assertEqual(rec.stats['dropped_frames'],7)


if __name__=='__main__':unittest.main()
