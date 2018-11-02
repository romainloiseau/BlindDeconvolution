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

image = cv2.resize(mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"])), (100, 100))
plt.figure(figsize = (8, 8))
plt.subplot(221)
plt.imshow(image, cmap = "gray")
plt.title("Original image")
blurringkernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
#image = Kernel(blurringkernel).convolveScipy(image) #Blurring image
blurredimage = Convolution(image.shape, blurringkernel).convolve(image)
plt.subplot(222)
plt.title("Blurred image")
plt.imshow(blurredimage, cmap = "gray")

"""
fe = FreeEnergy(image, blurringkernel)
fe.renderImage()
fe.initialize()
fe.iterate()
"""

plt.subplot(223)
plt.imshow(Convolution(image.shape, PARAMS["filters"][[k for k in PARAMS["filters"]][0]]).convolve(image), cmap = "gray")
plt.title("Original derivative")

derivatives = [Convolution(blurredimage.shape, PARAMS["filters"][k]).convolve(blurredimage) for k in PARAMS["filters"]]        

plt.subplot(224)
plt.title("Blurred derivative")
plt.imshow(derivatives[0], cmap = "gray")
plt.show()

fe = FreeEnergy(derivatives[0], blurringkernel)
fe.initialize()
fe.iterate()
