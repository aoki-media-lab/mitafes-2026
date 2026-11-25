import unittest
import torch
from ultralytics.engine.results import Results
from gpu_backend import EnsembleModel,FLIP_KEYPOINTS
import datetime
from views import load_view,detector_settings,detector_at


class DetectorSettingsTest(unittest.TestCase):
    def test_defaults_and_views(self):
        glass=detector_settings(load_view('glass'))
        self.assertEqual((glass['pose_model'],glass['conf'],glass['iou'],glass['tta_flip']),('yolo11m-pose.pt',.1,.7,False))
        plaza=detector_settings(load_view('plaza'))
        self.assertEqual((plaza['pose_model'],plaza['conf'],plaza['tta_flip']),('yolo11m-pose.pt',.1,False))
        with self.assertRaises(ValueError):detector_settings(load_view('glass'),{'unknown':1})

    def test_plaza_evening_schedule(self):
        view=load_view('plaza');at=lambda h,m:detector_at(view,datetime.datetime(2026,11,23,h,m))['conf']
        self.assertEqual([at(5,59),at(6,0),at(12,0),at(15,59),at(16,0),at(18,59),at(23,59)],[.1,.1,.1,.1,.05,.05,.05])
        self.assertEqual(detector_at(view,datetime.datetime(2026,11,23,18,0),{'conf':.2})['conf'],.2)
        with self.assertRaises(ValueError):detector_at(dict(view,schedule=[dict(view['schedule'][0],detector={'tta_flip':True})]),datetime.datetime(2026,11,23,18,0))

    def test_unflip_maps_mirrored_detections_back(self):
        class Shape:shape=(100,200,3)
        boxes=torch.tensor([[10.,5.,50.,60.,.9,0.]])
        points=torch.zeros(1,17,3);points[0,:,0]=torch.arange(17.);points[0,:,2]=1
        r=Results(Shape(),'',{0:'person'},boxes=boxes,keypoints=points)
        b,k=EnsembleModel.unflip(r)
        self.assertEqual(b[0,:4].tolist(),[150.,5.,190.,60.])
        # joint i of the result is the mirrored joint FLIP_KEYPOINTS[i], at x = width - x
        self.assertEqual(k[0,:,0].tolist(),[200.-j for j in FLIP_KEYPOINTS])


if __name__=='__main__':unittest.main()
