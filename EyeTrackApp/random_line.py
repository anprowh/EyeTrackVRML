import numpy as np
import keyboard as kb
import cv2
from random import choices
from string import ascii_letters, digits
import os
import csv
from threading import Thread


FOW_HALF_WIDTH = 675
FOW_CENTER_X = 0
FOW_HALF_HEIGHT = 600
FOW_CENTER_Y = -50

DOT_SPEED = 0.01

def bezier_interp4(x1, y1, x2, y2, x3, y3, x4, y4, pos):
    c1 = bezier_interp3(x1, y1, x2, y2, x3, y3, 0.5 + pos / 2)
    c2 = bezier_interp3(x2, y2, x3, y3, x4, y4, pos / 2)
    return c1[0] * (1 - pos) + c2[0] * pos, c1[1] * (1 - pos) + c2[1] * pos

def bezier_interp3(x1, y1, x2, y2, x3, y3, pos):
    dx1 = x1 + 2 * (x2 - x1) * pos
    dx2 = x2 + (x2 - x3) + 2 * (x3 - x2) * pos
    dy1 = y1 + 2 * (y2 - y1) * pos
    dy2 = y2 + (y2 - y3) + 2 * (y3 - y2) * pos
    return dx1 * (1 - pos) + dx2 * pos, dy1 * (1 - pos) + dy2 * pos

def generate_point(last_point, FOW_CENTER_X, FOW_CENTER_Y, FOW_HALF_WIDTH, FOW_HALF_HEIGHT):
    dx, dy = 1000, 1000
    tmpx = (dx - FOW_CENTER_X) / FOW_HALF_WIDTH  # Normalized distance from center
    tmpy = (dy - FOW_CENTER_Y) / FOW_HALF_HEIGHT
    tmpsx = (dx - last_point[0]) / FOW_HALF_WIDTH  # Normalized distance from last point
    tmpsy = (dy - last_point[1]) / FOW_HALF_HEIGHT
    while tmpx ** 2 + tmpy ** 2 >= 1.0 or tmpsx ** 2 + tmpsy ** 2 < 0.7:
        dx = np.random.uniform(-FOW_HALF_WIDTH, FOW_HALF_WIDTH)
        dy = np.random.uniform(-FOW_HALF_HEIGHT, FOW_HALF_HEIGHT)
        tmpx = (dx - FOW_CENTER_X) / FOW_HALF_WIDTH
        tmpy = (dy - FOW_CENTER_Y) / FOW_HALF_HEIGHT
        tmpsx = (dx - last_point[0]) / FOW_HALF_WIDTH
        tmpsy = (dy - last_point[1]) / FOW_HALF_HEIGHT
    return dx, dy

def generate_smooth_line(FOW_CENTER_X, FOW_CENTER_Y, FOW_HALF_WIDTH, FOW_HALF_HEIGHT):
    points = [(-1000, -1000)]
    for i in range(4):
        points.append(generate_point(points[i-1], FOW_CENTER_X, FOW_CENTER_Y, FOW_HALF_WIDTH, FOW_HALF_HEIGHT))
    x, y = points[1]
    dx, dy = points[2]
    dist = np.sqrt((dx - x) ** 2 + (dy - y) ** 2)
    iters = int(dist / DOT_SPEED)
    v = 1.0 / iters
    pstate = 0.0

    while True:
        yield bezier_interp4(
            *points[0], *points[1], *points[2], *points[3], pstate
        )

        pstate += v
        if pstate >= 1.0:
            points.pop(0)
            points.append(generate_point(points[-1], FOW_CENTER_X, FOW_CENTER_Y, FOW_HALF_WIDTH, FOW_HALF_HEIGHT))
            x, y = points[1]
            dx, dy = points[2]
            dist = np.sqrt((dx - x) ** 2 + (dy - y) ** 2)
            iters = int(dist / DOT_SPEED)
            v = 1.0 / iters
            pstate = 0.0

DATASET_FOLDER = "C:\\Users\\anpro\\dataset\\EyeTracking"

def save_img(filename, img):
    cv2.imwrite(filename, img)

def random_line(self):
    if not hasattr(self, 'line_generator'):
        h, w = self.current_image_gray.shape
        # self.line_generator = generate_smooth_line(0, -0.5, 0.7, 0.4)
        self.line_generator = generate_smooth_line(0, -0.2, 0.8, 0.8)
        self.random_chars = ''.join(choices(digits + ascii_letters, k=10))
        os.makedirs(f"{DATASET_FOLDER}\\{self.random_chars}")
        print(self.random_chars)
        self.img_num = 0
        self.img_csv_writer = csv.writer(open(f"{DATASET_FOLDER}\\{self.random_chars}.csv", 'w'))
        
        
    point = next(self.line_generator)
    # self.current_image_gray_clean
    if kb.is_pressed('space'):
        self.img_num += 1
        filename = f"{DATASET_FOLDER}\\{self.random_chars}\\{self.img_num}.jpg"
        th = Thread(target=save_img, args=(filename, self.current_image_gray_clean.copy()))
        th.start()
        # save_img(filename, img=self.current_image_gray_clean.copy())
        print(f"Saved? {self.img_num} images", end='\r')
        self.img_csv_writer.writerow((filename, point[0], point[1]))
    
    if kb.is_pressed('q'):
        self.random_chars = ''.join(choices(digits + ascii_letters, k=10))
        os.makedirs(f"{DATASET_FOLDER}\\{self.random_chars}")
        print(self.random_chars)
        self.img_num = 0
        self.img_csv_writer = csv.writer(open(f"{DATASET_FOLDER}\\{self.random_chars}.csv", 'w'))
        
            
    # print(point)
    return point