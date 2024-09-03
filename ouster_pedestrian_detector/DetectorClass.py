from ultralytics import YOLO
import cv2
import numpy as np
from time import time
from pprint import pprint
import matplotlib.pyplot as plt

# from pathlib import Path
# from rosbags.highlevel import AnyReader
# from rosbags.image import image_to_cvimage

from collections import namedtuple

Persons = namedtuple('Persons', ['result', 'quantity', 'center', 'conf', 'polar_position', 'cart_position'])

class Viewer():
    def __init__(self, size=5000):
        plt.ion()  # Включаем интерактивный режим

        # Создаем объект для графика и изображения
        self.fig, self.ax = plt.subplots(2, 1, figsize=(12, 10))  # Увеличенные размеры 12x10 дюймов
        self.sc = self.ax[0].scatter([], [], c="blue")  # График точек
        self.ax[0].scatter(0, 0, c='red', s=100, marker='s')  # Красная точка

        # Настройка осей для графика
        self.ax[0].set_xlim(-size, size)
        self.ax[0].set_ylim(-size, size)
        self.ax[0].set_xlabel('X-axis')
        self.ax[0].set_ylabel('Y-axis')
        self.ax[0].grid(True)
        self.ax[0].set_title('2D Point Visualization')

        # Устанавливаем соотношение сторон на квадратное
        self.ax[0].set_aspect('equal', adjustable='box')

        # Создание области для изображения
        self.ax[1].axis('off')  # Выключаем оси для изображения
        self.image_display = self.ax[1].imshow(np.zeros((64, 1024, 3), dtype=np.uint8))  # Инициализируем изображение

        # Рисуем 3 тонкие вертикальные зеленые линии на изображении
        self.add_vertical_lines_to_image()

        # Добавляем горизонтальную и вертикальную линии на графике
        self.ax[0].axhline(0, color='green', linestyle='-', linewidth=1)  # Горизонтальная линия
        self.ax[0].axvline(0, color='green', linestyle='-', linewidth=1)  # Вертикальная линия

    def add_vertical_lines_to_image(self):
        """Рисует 3 тонкие вертикальные линии на изображении."""
        # Получаем текущее изображение в виде numpy array
        image = np.zeros((64, 1024, 3), dtype=np.uint8)  # Заполняем черным
        # Рисуем линии
        for x in [256, 512, 768]:
            image[:, x, :] = [0, 255, 0]  # Зеленый цвет

        # Устанавливаем измененное изображение
        self.image_display.set_array(image)

    def plot(self, person):
        """Функция для обновления графика с точками и изображением."""
        # Обновляем положение точек
        self.sc.set_offsets(person.cart_position)  # Обновляем точки
        
        # Получение изображения от person и его отображение
        image = person.result.plot()  # Предполагается, что result.plot() возвращает numpy array
        # Проверка размерности изображения, чтобы оно подходило к 64x1024x3
        if image.shape == (64, 1024, 3):
            for x in [256, 512, 768]:
                image[:, x, :] = [0, 255, 0]  # Зеленый цвет
            self.image_display.set_array(image)  # Обновляем только массив изображения

        plt.draw()             # Обновляем график
        plt.pause(0.01)        # Позволяем графику обновиться



class PedestrianDetector():
    def __init__(self, model_path, center_radius=10, imgsz=640, imgwidth=1024, angle_offset=0, conf_threshold=0.0) -> None:
        self.model = YOLO(model_path, task="detect")
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
                raise TypeError("Unsupported type of image")

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
        return distance * self.distance_scale


    def find_people(self, img, img_range) -> namedtuple:
        result = self.model.predict(img, imgsz=self.imgsz, verbose=False)[0]
        
        if result:
            centers = result.boxes.xywh[:,0:2].cpu().numpy() # array[(x,y) * amount]\
            confs = result.boxes.conf.cpu().numpy()
            bearing = self.get_bearing(centers)
            distance = self.get_distance(img_range, centers)
            polar_position = np.column_stack((bearing, distance))
            cart_position = np.column_stack((distance * np.cos(bearing), distance * np.sin(bearing)))
            # print(f"conf: {confs.shape}, bear: {bearing.shape}, dist: {distance.shape}, polar: {polar_position.shape}, cart: {cart_position.shape}")
            confidence_mask = confs > self.conf_threshold
            quantity = centers[confidence_mask].shape[0]
            if quantity:
                return Persons(result,
                            quantity,
                            centers[confidence_mask],
                            confs[confidence_mask],
                            polar_position[confidence_mask],
                            cart_position[confidence_mask])

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