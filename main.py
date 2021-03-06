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
from freeEnergy import FreeEnergy
from convolution import Convolution

PARAMS = yaml.load(open("params.yaml"))

image = np.array(cv2.resize(mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"])), (128, 128))).astype(float)
#image = np.array(np.meshgrid(np.arange(128), np.arange(128))).transpose(1, 2, 0)
#image = (np.sum((image - np.array([64, 64]))**2, axis = -1) < 32**2).astype(float)

image /= 255.

plt.figure(figsize = (8, 8))
plt.subplot(221)
plt.imshow(image, cmap = "gray")
plt.title("Original image")
"""
blurringkernel = np.array([[0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 1.],
                           [0., 0., 1., 0., 0.],
                           [1., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]])
"""
blurringkernel = np.array([[0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [.2, .5, 1., .5, .2],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]])

#blurringkernel = np.lib.pad(blurringkernel, ((2, 2), (2, 2)), mode = "constant", constant_values = (0))
blurringkernel /= np.sum(blurringkernel)
#image = Kernel(blurringkernel).convolveScipy(image) #Blurring image
blurredimage = Convolution(image.shape, blurringkernel).convolve(image)# + 0.001 * (2 * np.random.random(image.shape) - 1)
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
algo = FreeEnergy(blurredimage, derivativeSpace = True, truek = blurringkernel, truex = image)
algo.deconv()
