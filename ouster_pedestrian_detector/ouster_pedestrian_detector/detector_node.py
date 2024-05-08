
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync


class YoloDetector:
    
    _DNN = False
    _IMAGE_SIZE = 1024
    _HALF = False
    _AUGMENT = False
    _CLASSES = None
    _AGNOSTIC_NMS = False
    _MAX_DET = 1000
    
    def __init__(self,
                 weights_path: str,
                 device: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45) -> None:
        self._model, self._img_size, self._stride, self._auto, self._half = YoloDetector._load_model(weights_path, device)
        self._device = device
        self._conf_thres = conf_threshold
        self._iou_thres = iou_threshold
    
    @torch.no_grad()
    def __call__(self, img: np.ndarray, bgr_input: bool = False):
        original_shape = img.shape
        img = self._process_image(img, bgr_input=bgr_input)
        
        im = torch.from_numpy(img).to(self._device)
        im = im.half() if self._half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = im[None]
        
        gn = torch.tensor(original_shape)[[1, 0, 1, 0]]
        
        pred = self._model(im, augment=YoloDetector._AUGMENT, visualize=False)
        pred = non_max_suppression(pred, 
                                   self._conf_thres, 
                                   self._iou_thres, 
                                   YoloDetector._CLASSES, 
                                   YoloDetector._AGNOSTIC_NMS, 
                                   max_det=YoloDetector._MAX_DET)
        
        for i, det in pred:
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], original_shape).round()
            
        result = []
            
        for *xyxy, conf, cls in reversed(det):
            result.append(xyxy)
            
        return result
    
    def _process_image(self,
                       img: np.ndarray, 
                       bgr_input: bool) -> np.ndarray:
        img = letterbox(img, self._img_size, stride=self._stride, auto=self._pt)[0]
        if bgr_input:
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img    

    @staticmethod
    def _load_model(weights_path: str,
                    device: str) -> nn.Module:
        model = DetectMultiBackend(weights_path, device=device, dnn=YoloDetector._DNN)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(YoloDetector._IMAGE_SIZE, s=stride)

        half = YoloDetector._HALF
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
            
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        
        return model, imgsz, stride, pt and not jit, half


DEVICE = "cpu"


def main():
    detector = YoloDetector(weights_path="/home/captain/data/last.pt",
                            device="cpu")
    img = cv2.imread("/home/captain/data/00500.jpg")
    result = detector(img, bgr_input=True)
    print(result)
    # print('Hi from ouster_pedestrian_detector.')
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    # model.load_state_dict(torch.load("/home/captain/data/yolo_v5s_ft.pt", map_location=DEVICE))
    # print("Model loaded")


if __name__ == '__main__':
    main()
