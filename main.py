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
from kernel import Kernel

PARAMS = yaml.load(open("params.yaml"))

image = cv2.resize(mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"])), (50, 50))
plt.imshow(image, cmap = "gray")
plt.title("Original image")
plt.show()
image = Kernel(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9).convolveScipy(image) #Blurring image
plt.title("Blurred image")
plt.imshow(image, cmap = "gray")
plt.show()

derivatives = [Kernel(PARAMS["filters"][k]).convolveScipy(image) for k in PARAMS["filters"]]        

fe = FreeEnergy(derivatives[0])
fe.renderImage()
fe.initialize()
fe.iterate()