import cv2
import numpy as np
from fontTools.varLib import plot
from matplotlib import pyplot as plt
###############################################################################################################

def hist_eq (imgs,bins):
    img = cv2.imread(imgs)  # insert  image
    grey2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to grey scale

    cv2.imshow('greyscale', grey2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist = cv2.calcHist([grey2], [0], None, [bins], [0, bins])  # calculate image histogram

    plt.plot(hist)
    plt.show()

    grey2new = cv2.equalizeHist(grey2)  # histogram equalization

    cv2.imshow('greyscale', grey2new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    histnew = cv2.calcHist([grey2new], [0], None, [bins], [0, bins])  # display histogram after equalization

    plt.plot(histnew)
    plt.show()
#########################################################################################################
    

