import cv2
import numpy as np
from fontTools.varLib import plot
from matplotlib import pyplot as plt
from math import log10, sqrt

def func_return(img):
    m = 0
    variance = 0.005
    gaussian = np.random.normal(m, variance, (360, 520))

    n_img = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        n_img = img + gaussian
    else:
        n_img[:, :, 0] = img[:, :, 0] + gaussian
        n_img[:, :, 1] = img[:, :, 1] + gaussian
        n_img[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(n_img, n_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = n_img.astype(np.uint8)
    return noisy_image

def filter_img(img):
    m = 0
    variance = 0.005
    gaussian = np.random.normal(m, variance, (360, 520))

    n_img = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        n_img = img + gaussian
    else:
        n_img[:, :, 0] = img[:, :, 0] + gaussian
        n_img[:, :, 1] = img[:, :, 1] + gaussian
        n_img[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(n_img, n_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = n_img.astype(np.uint8)

    cv2.imshow("img", img)
    cv2.imshow("gaussian", gaussian)
    cv2.imshow("noisy", noisy_image)
    cv2.waitKey(0)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
