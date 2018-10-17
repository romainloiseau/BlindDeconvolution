# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:24:47 2018

@author: romai
"""

from scipy import ndimage

class Kernel:
    
    def __init__(self, kernel):
        self.kernel = kernel
        
    def getShape(self):
        return self.kernel.shape
        
    def convolve(self, image, mode = "constant", cval = 0.0):
        return ndimage.convolve(image, self.kernel, mode = mode, cval = cval)