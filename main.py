# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:01:57 2018

@author: romai
"""

import cv2
import opencv_utils as mycv2
from scipy import ndimage

image = mycv2.cvtGray(mycv2.loadImage("data/image.jpg"))

mycv2.show(image)

fv = ndimage.convolve(image, [[-1, 1]], mode='constant', cval=0.0)
fh = ndimage.convolve(image, [[-1], [1]], mode='constant', cval=0.0)

mycv2.show(fv)
mycv2.show(fh)
