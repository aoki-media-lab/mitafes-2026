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
# Loaded on purpose: Ultralytics' NMS uses torchvision.ops.nms when torchvision is already imported, otherwise
# a Python loop with a GPU sync per kept box. Same boxes (checked 96/96 identical on crowded frames at conf 0.1
# and 0.03), ~30x faster: 125-200 ms -> ~5 ms per batch of 8 in a crowd.
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
