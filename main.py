# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:01:57 2018

@author: romai
"""

import yaml
import numpy as np
import opencv_utils as mycv2
import cv2
import matplotlib.pyplot as plt
from data import Data
from convolution import Convolution

PARAMS = yaml.load(open("params.yaml"))

image = cv2.resize(mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"])), (128, 128))
plt.figure(figsize = (8, 8))
plt.subplot(221)
plt.imshow(image, cmap = "gray")
plt.title("Original image")
blurringkernel = np.array([[0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 1.],
                           [0., 0., 1., 0., 0.],
                           [1., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]])
blurringkernel /= np.sum(blurringkernel)
#image = Kernel(blurringkernel).convolveScipy(image) #Blurring image
blurredimage = Convolution(image.shape, blurringkernel).convolve(image)
plt.subplot(222)
plt.title("Blurred image")
plt.imshow(blurredimage, cmap = "gray")

plt.subplot(223)
plt.imshow(Convolution(image.shape, PARAMS["derivativefilters"][[k for k in PARAMS["derivativefilters"]][0]]).convolve(image), cmap = "gray")
plt.title("Original derivative, " + [k for k in PARAMS["derivativefilters"]][0])

derivatives = [Convolution(blurredimage.shape, PARAMS["derivativefilters"][k]).convolve(blurredimage) for k in PARAMS["derivativefilters"]]        

plt.subplot(224)
plt.title("Blurred derivative, " + [k for k in PARAMS["derivativefilters"]][0])
plt.imshow(derivatives[0], cmap = "gray")
plt.show()

#ALGO
data = Data(blurredimage, derivativeSpace = True, truek = blurringkernel, truex = image)
data.deconv()
