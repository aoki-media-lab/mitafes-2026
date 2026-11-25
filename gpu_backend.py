"""Fixed-shape GPU inference, pinned input buffers and optional CUDA graphs.

All postprocessing thresholds and scaling come from Ultralytics. No reference
counts, person IDs or video timestamps are read here.
"""
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
# Loaded on purpose: Ultralytics' NMS uses torchvision.ops.nms when torchvision is already imported,
# otherwise a Python loop with a GPU sync per kept box. Night clip 18:10 (10 min, conf 0.05): inference
# 347 s -> 108 s, detections equal to <1 px in 98% of frames (rest: run-to-run float noise), 237 vs 240 counts.
import torchvision  # noqa: F401


class GpuModel:
    def __init__(self,path,device,shape,backend='graph',batch=1,half=False):
        self.device=torch.device(f'cuda:{int(device)}');self.backend=backend
        self.half=half;self.batch=batch;self.shape=(batch,3,*shape)
        self.graph_precision='FP16' if half else 'FP32'
        torch.cuda.set_device(self.device)
        yolo=YOLO(str(path));self.task=yolo.task
        self.model=yolo.model.to(self.device).eval().fuse(verbose=False)
        self.model=self.model.half() if half else self.model.float()
        self.names=self.model.names;self.nc=len(self.names)
        self.kpt_shape=getattr(self.model,'kpt_shape',None) if self.task=='pose' else None
        self.host=torch.empty(self.shape,dtype=torch.uint8,pin_memory=True)
        self.bytes=torch.zeros(self.shape,dtype=torch.uint8,device=self.device)
        self.stream=torch.cuda.Stream(device=self.device)
        self.graph=None
        with torch.inference_mode(),torch.cuda.stream(self.stream):
            for _ in range(4):self._forward()
        self.stream.synchronize()
        if backend=='graph':
            self.graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(self.graph,stream=self.stream):
                self.prediction=self._forward()
        self.stream.synchronize()

    def _forward(self):
        tensor=self.bytes.to(torch.float16 if self.half else torch.float32)/255.
        result=self.model(tensor)
        return result[0] if isinstance(result,(tuple,list)) else result

    def enqueue(self,images):
        if not 1<=len(images)<=self.batch:raise ValueError('Invalid batch size')
        self.images=images
        for i,image in enumerate(images):
            transformed=LetterBox(new_shape=self.shape[-2:],auto=False,stride=32)(image=image)
            array=np.ascontiguousarray(transformed[...,::-1].transpose(2,0,1))
            self.host[i].copy_(torch.from_numpy(array))
        # The unused tail only occurs for the final offline batch.
        if len(images)<self.batch:self.host[len(images):].zero_()
        with torch.inference_mode(),torch.cuda.stream(self.stream):
            self.bytes.copy_(self.host,non_blocking=True)
            if self.graph:self.graph.replay()
            else:self.prediction=self._forward()

    def enqueue_prepared(self,array,shapes):
        # Only image shape is needed for postprocessing; pixels were prepared
        # in the parent so large full-resolution frames never cross IPC.
        class ImageShape:
            def __init__(self,shape):self.shape=tuple(shape)
        self.images=[ImageShape(shape) for shape in shapes]
        self.host.copy_(torch.from_numpy(array))
        with torch.inference_mode(),torch.cuda.stream(self.stream):
            self.bytes.copy_(self.host,non_blocking=True)
            if self.graph:self.graph.replay()
            else:self.prediction=self._forward()

    def collect(self,conf,iou,classes):
        output=[]
        with torch.inference_mode(),torch.cuda.stream(self.stream):
            detections=non_max_suppression(self.prediction,conf_thres=conf,iou_thres=iou,classes=classes,nc=self.nc,max_det=300)
            for image,pred in zip(self.images,detections):
                pred[:,:4]=ops.scale_boxes(self.shape[2:],pred[:,:4],image.shape)
                if self.kpt_shape:
                    points=pred[:,6:].view(-1,*self.kpt_shape)
                    ops.scale_coords(self.shape[2:],points,image.shape)
                # One transfer includes both boxes and joints.
                cpu=pred.cpu()
                output.append(Results(image,'',self.names,boxes=cpu[:,:6],
                                      keypoints=cpu[:,6:].view(-1,*self.kpt_shape) if self.kpt_shape else None))
        return output

    def predict(self,images,imgsz,conf,iou,classes,device=None,half=False,verbose=False):
        self.enqueue(images)
        return self.collect(conf,iou,classes)


FLIP_KEYPOINTS=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]   # COCO left/right swap


class EnsembleModel:
    """Several detectors on the same input; their detections are united by one more NMS.

    ``flips[i]`` runs member i on the horizontally mirrored input and maps its boxes and
    joints back (test-time augmentation with the same weights).

    Each model runs its own NMS first (as a single model would), then the boxes of all
    models are merged per image with class-wise NMS at the same IoU. A person found by
    either model is kept; the same person found by both is kept once (highest score).
    Used for the person branches: a larger model finds people in dark clothes and
    partly hidden people that the smaller one misses, and vice versa.
    """
    def __init__(self,paths,device,shape,backend='graph',batch=1,half=False,flips=None):
        self.models=[GpuModel(p,device,shape,backend,batch,half) for p in paths]
        self.flips=list(flips) if flips else [False]*len(paths)
        first=self.models[0];self.task=first.task;self.names=first.names;self.kpt_shape=first.kpt_shape
        if any(m.task!=self.task or m.kpt_shape!=self.kpt_shape for m in self.models):
            raise ValueError('Ensemble members must be the same kind of model')

    def enqueue_prepared(self,array,shapes):
        # Letterboxing pads both sides equally, so mirroring the prepared input equals
        # preparing the mirrored image (to within one pixel).
        for m,f in zip(self.models,self.flips):m.enqueue_prepared(np.ascontiguousarray(array[...,::-1]) if f else array,shapes)

    def enqueue(self,images):
        for m,f in zip(self.models,self.flips):m.enqueue([np.ascontiguousarray(i[:,::-1]) for i in images] if f else images)

    @staticmethod
    def unflip(result):
        w=result.orig_shape[1];boxes=result.boxes.data.clone()
        boxes[:,[0,2]]=w-boxes[:,[2,0]]
        points=None
        if result.keypoints is not None:
            points=result.keypoints.data[:,FLIP_KEYPOINTS].clone();points[...,0]=w-points[...,0]
        return boxes,points

    def collect(self,conf,iou,classes):
        parts=[m.collect(conf,iou,classes) for m in self.models];output=[]
        for results in zip(*parts):
            pairs=[self.unflip(r) if f else (r.boxes.data,r.keypoints.data if self.kpt_shape else None) for r,f in zip(results,self.flips)]
            boxes=torch.cat([b for b,_ in pairs])
            points=torch.cat([k for _,k in pairs]) if self.kpt_shape else None
            if len(boxes):
                offset=boxes[:,5:6]*4096.   # class-wise NMS: boxes of different classes never overlap
                keep=torchvision.ops.nms(boxes[:,:4]+offset,boxes[:,4],iou)
                boxes=boxes[keep];points=points[keep] if points is not None else None
            output.append(Results(results[0].orig_img,'',self.names,boxes=boxes,keypoints=points))
        return output

    def predict(self,images,imgsz,conf,iou,classes,device=None,half=False,verbose=False):
        self.enqueue(images)
        return self.collect(conf,iou,classes)
