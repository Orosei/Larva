import cv2
import numpy as np


def cal_ext_eng1(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = img.copy()
    else:
        raise ValueError("仅支持3通道彩色图或单通道灰度图")

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)

    gx_sq = gx ** 2
    gy_sq = gy ** 2
    g_mag = np.sqrt(gx_sq + gy_sq)
    mag_max = np.max(g_mag)
    if mag_max > 0:
        g_mag = g_mag / mag_max

    return g_mag


def cal_ext_eng2(img):
    img1 = img.astype(np.float32)
    diff = np.zeros((8, *img1.shape[:2]), np.float32)
    directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    for i, (dy, dx) in enumerate(directions):
        k = np.zeros((3, 3), np.float32)
        k[1 + dy, 1 + dx], k[1, 1] = 1, -1
        for c in range(3):
            diff[i] += cv2.filter2D(img1[..., c], -1, k) ** 2
    return np.sqrt(diff)


def cal_ext_eng3(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gx_sq = np.mean(gx ** 2, axis=2)
    gy_sq = np.mean(gy ** 2, axis=2)
    g_mag = np.sqrt(gx_sq + gy_sq)
    return g_mag


def cal_ext_eng_v4(img):
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]

    b_gx = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=3)
    b_gy = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=3)

    g_gx = cv2.Sobel(g_channel, cv2.CV_64F, 1, 0, ksize=3)
    g_gy = cv2.Sobel(g_channel, cv2.CV_64F, 0, 1, ksize=3)

    r_gx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0, ksize=3)
    r_gy = cv2.Sobel(r_channel, cv2.CV_64F, 0, 1, ksize=3)

    gx = np.sqrt(np.power(b_gx, 2) + np.power(g_gx, 2) + np.power(r_gx, 2))
    gy = np.sqrt(np.power(b_gy, 2) + np.power(g_gy, 2) + np.power(r_gy, 2))

    g = cv2.magnitude(gx, gy)

    return g
