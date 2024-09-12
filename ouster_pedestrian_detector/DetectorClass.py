from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

Persons = namedtuple('Persons', ['result', 'quantity', 'center', 'conf', 'polar_position', 'cart_position'])


class PedestrianDetector():
    def __init__(self, model_path, center_radius=10, imgsz=640, imgwidth=1024, angle_offset=0, conf_threshold=0.0) -> None:
        torch.cuda.set_device(0)
        self.model = YOLO(model_path, task="detect")
        self.model.to("cuda")
        self.imgsz = imgsz
        self.imgwidth = imgwidth
        self.center_radius = center_radius
        self.angle_offset = angle_offset
        self.conf_threshold = conf_threshold
        self.distance_scale = 4     # ouster github issue

    def process_image(self, img, type) -> np.ndarray:
        # pprint(img)
        # img = image_to_cvimage(img)
        match type:
            case "/ouster/reflec_image":
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
            case "/ouster/range_image":
                return img
            case _:
                raise TypeError(f"Unsupported type of image: {type}")

    def get_bearing(self, centers) -> np.ndarray:
        return ((1-(centers[:,0] / self.imgwidth)) * 2 * np.pi)[:, np.newaxis] + self.angle_offset
    
    def get_distance(self, img_range, centers) -> np.ndarray:
        # print("centers",centers.shape)
        min_x = (centers[:,0] - self.center_radius).astype(np.int16)
        max_x = (centers[:,0] + self.center_radius).astype(np.int16)
        min_y = (centers[:,1] - self.center_radius).astype(np.int16)
        max_y = (centers[:,1] + self.center_radius).astype(np.int16)

        amount = centers.shape[0]
        distance = np.zeros((amount,1))
        for i in range(amount):
            mask = np.zeros(img_range.shape, dtype=bool)
            mask[min_y[i]:max_y[i], min_x[i]:max_x[i]] = True
            filtered_pixels = img_range[mask]
            distance[i] = np.mean(filtered_pixels, axis=0)
        return (distance * self.distance_scale / 1000).astype(float)


    def find_people(self, img, img_range) -> namedtuple:
        result = self.model.predict(img, imgsz=self.imgsz, verbose=False)[0]
        
        if result:
            confs = result.boxes.conf.cpu().numpy()
            confidence_mask = confs > self.conf_threshold
            confs = confs[confidence_mask]

            centers = result.boxes.xywh[:,0:2].cpu().numpy()[confidence_mask]

            bearing = self.get_bearing(centers)
            distance = self.get_distance(img_range, centers)
            polar_position = np.column_stack((bearing, distance))
            cart_position = np.column_stack((distance * np.cos(bearing), distance * np.sin(bearing)))
            # print(f"conf: {confs.shape}, bear: {bearing.shape}, dist: {distance.shape}, polar: {polar_position.shape}, cart: {cart_position.shape}")
            quantity = centers.shape[0]
            if quantity:
                return Persons(result,
                            quantity,
                            centers,
                            confs,
                            polar_position,
                            cart_position)

    def plot_results(results, conf_threshold=0.0):
        results = results[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
        frame = results.orig_img
        

        for box, conf in zip(boxes, results.boxes.conf):
            if conf>conf_threshold:
                color = (0,0,255)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        return frame