## @package opencv_utils
# Encapsulates opencv functions to use them quicker in our project.
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:19:26 2018

@author: romai
"""

import numpy as np
import cv2
import yaml

PARAMS = yaml.load(open("params.yaml"))

## Show an image.
# @param image The image.
# @param maxwidth The maximum width of the image.
# This function of course conserves proportions.
def show(image, maxwidth = np.nan):
    if(np.isnan(maxwidth)):
        maxwidth = PARAMS["render"]["maxwidth"]
    cv2.imshow('image',cv2.resize(image, (maxwidth, int(maxwidth * image.shape[0] / image.shape[1]))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## Load an image.
# @param path The path to the classifier.
# @return The image.
def loadImage(path):
    if(PARAMS["verbose"]):
        print("Loading image from " + path + " ... ", end = "")
    image = cv2.imread(path) #Load image
    if(PARAMS["verbose"]):
        print("Done")
    return image

## Convert an RGB image in a grayscale image.
# @param image The image.
# @return The grayscaled image.
def cvtGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert to gray