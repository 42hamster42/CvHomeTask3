import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


def hough_inner(img, buff1, buff2, x_min, x_max):
    if x_max - x_min <= 1:
        buff1[x_min] = img[x_min]
    else:
        x_mid = (x_min + x_max) // 2
        hough_inner(img, buff2, buff1, x_min, x_mid)
        hough_inner(img, buff2, buff1, x_mid, x_max)
        for shift in range(x_max - x_min):
            y = x_mid + shift // 2
            buff1[x_min + shift] = buff2[x_min + shift // 2] + np.concatenate(
                (buff2[y, shift // 2 + shift % 2:], buff2[y, :shift // 2 + shift % 2]))


def hough(img):
    h, w = img.shape[:2]
    size = 1
    while size < h or size < w:
        size *= 2
    padded_img = np.zeros((size, size), np.int)
    padded_img[:h, :w] = img
    buffer1 = np.zeros((size, size), np.int)
    buffer2 = np.zeros((size, size), np.int)
    hough_inner(padded_img, buffer1, buffer2, 0, size)
    return buffer1, size


IN_DIR = Path('input')
OUT_DIR = Path('output')
HOUGH_DIR = Path('hough')
TANSFORMED_DIR = Path('transformed')


def main():
    OUT_DIR.mkdir(exist_ok=True)
    HOUGH_DIR.mkdir(exist_ok=True)
    TANSFORMED_DIR.mkdir(exist_ok=True)

    inputs = os.listdir(str(IN_DIR))
    for fname in inputs:
        print(fname)
        im = cv2.imread(str(IN_DIR / fname))
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        square_im = cv2.Canny(gray_im, 100, 200)
        cv2.imwrite(str(TANSFORMED_DIR / fname), square_im)
        hough_im, size = hough(square_im)
        cv2.imwrite(str(HOUGH_DIR / fname), (hough_im * 255 / np.max(hough_im)).astype(np.uint8))
        rows_var = np.var(hough_im, axis=0)
        max_var = np.argmax(rows_var)
        print(max_var)
        h, w = im.shape[:2]
        alpha = (max_var - (size // 2)) / (size // 2)
        alpha *= 45
        print(alpha)
        cX = w // 2
        cY = h // 2
        M = cv2.getRotationMatrix2D((cX, cY), alpha, 1.0)
        im_rotated = cv2.warpAffine(im, M, (w, h))
        cv2.imwrite(str(OUT_DIR / fname), im_rotated)

    areas = []
    times_per_pixel = []
    for i in range(9, 16):
        size = 2 ** i
        img = np.zeros((size, size), np.uint8)
        start = datetime.now()
        hough(img)
        end = datetime.now()
        time = (end - start).total_seconds() / 1000
        area = size * size / 1000000
        areas.append(area)
        time_per_pixel = time / (area)
        times_per_pixel.append(time_per_pixel)
    plt.plot(areas, times_per_pixel)
    plt.savefig('plot.png')
    plt.show()


if __name__ == "__main__":
    main()
