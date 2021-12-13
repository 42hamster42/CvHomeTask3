import os
import cv2
import glob
from pathlib import Path
import numpy as np


def hough_inner(img, buff1, buff2, x_min, x_max):
    width = img.shape[0]
    res = buff1[x_min:x_max, :]
    if x_max - x_min <= 1:
        res[0] = img[x_min]
    else:
        x_mid = (x_min + x_max) // 2
        ans1 = hough_inner(img, buff2, buff1, x_min, x_mid)
        ans2 = hough_inner(img, buff2, buff1, x_mid, x_max)
        for shift in range(x_max - x_min):
            for x in range(width):
                res[shift, x] = ans1[shift // 2, x] + ans2[shift // 2, (x + shift // 2 + shift % 2) % width]
    return res


def hough(img):
    h, w = img.shape[:2]
    size = 1
    while size < h or size < w:
        size *= 2
    padded_img = np.zeros((size, size), np.int)
    padded_img[:h, :w] = img
    buffer1 = np.zeros((size, size), np.int)
    buffer2 = np.zeros((size, size), np.int)
    return hough_inner(padded_img, buffer1, buffer2, 0, size), size


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
        gray_im = cv2.resize(gray_im, (gray_im.shape[1] // 2, gray_im.shape[0] // 2))
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
        M = cv2.getRotationMatrix2D((cX, cY), int(alpha), 1.0)
        im_rotated = cv2.warpAffine(im, M, (w, h))
        cv2.imwrite(str(OUT_DIR / fname), im_rotated)


if __name__ == "__main__":
    main()
