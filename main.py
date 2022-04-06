import cv2
import numpy as np
from matplotlib import pyplot as plt
from histogram_func import hist_eq
img= cv2.imread("jetplane.tif")
hist_eq("jetplane.tif", 256)
hist_eq("jetplane.tif", 128)
hist_eq("jetplane.tif", 64)


