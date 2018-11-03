# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:55:13 2018

@author: romai
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:47:50 2018

@author: romai
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
import numpy as np
import yaml
import time
import opencv_utils as mycv2

PARAMS = yaml.load(open("params.yaml"))
verbose = PARAMS["verbose"]

class Convolution:
    
    def __init__(self, inputShape, kernel):
        #self.image = image.astype(np.float)
        self.inputShape = inputShape
        self.kernel = np.array(kernel).astype(np.float)
        self.kernelFT = self.computeKernelFT()
        
    def computeKernelFT(self):
        if(not self.kernel.shape == self.inputShape):
            if(self.kernel.shape[0] == self.kernel.shape[1]):
                k = np.zeros(self.inputShape)
                for i in range(self.kernel.shape[0]):
                    for j in range(self.kernel.shape[1]):
                        k[(i - 1) % self.inputShape[0], (j - 1) % self.inputShape[1]] = self.kernel[i, j]
                
                self.kernel = k
        
        return np.real(fft2(self.kernel))
        
    def convolve(self, image, mode = "direct"):
        if(image.shape != self.inputShape):
            raise ValueError("The convolution (shape = " + str(self.inputShape) + ") is not meant to be used with this image shape (" + str(image.shape) + ")")
        if(mode == "direct"):
            return np.real(ifft2(fft2(image) * self.kernelFT))
        elif(mode == "adjoint"):
            return np.real(ifft2(fft2(image) * (np.conjugate(self.kernelFT)).transpose()))
        else:
            raise ValueError("Mode for convolve must be either 'direct' or 'adjoint'. You call for convolve with mode = " + str(mode))
    
    def deconvolve(self, image):
        if(PARAMS["deconvolution"]["algorithm"] == "L2"):
            return self.l2(image)
        if(PARAMS["deconvolution"]["algorithm"] == "Sobolev"):
            return self.sobolev(image)
        else:
            raise ValueError("ERROR !! No algorithm ...     Solver : ", PARAMS["deconvolution"]["algorithm"])
    
    def l2(self, image):
        return np.real(ifft2(fft2(image) * self.kernelFT / (abs(self.kernelFT)**2 + PARAMS["deconvolution"]["l2"]["lambda"])))
    
    def sobolev(self, image):
        n = self.inputShape[0]
        x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
        [Y, X] = np.meshgrid(x, x)
        S = (X**2 + Y**2) * (2/n)**2
        return np.real(ifft2(fft2(image) * self.kernelFT / (abs(self.kernelFT)**2 + S*PARAMS["deconvolution"]["sobolev"]["lambda"] + 10**(-15))))
    
def runTests():
    print("Convolution tests")
    image = mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"]))
    
    plt.figure(figsize = (9, 6))
    
    plt.subplot(231)
    plt.imshow(image, cmap = "gray")
    plt.title("Original image")

    kernel = PARAMS["filters"]["dh"]
    
    plt.subplot(234)
    plt.imshow(kernel, cmap = "gray")
    plt.title("Kernel")
    
    t1 = time.time()
    convolution = Convolution(image.shape, kernel)
    convolved = convolution.convolve(image)
    t2 = time.time()
    
    plt.subplot(235)
    plt.imshow(convolution.kernelFT, cmap = "gray")
    plt.title("Kernel fourier")
    
    plt.subplot(232)
    plt.imshow(convolved, cmap = "gray")
    plt.title("Convolved image")
    
    deconvolved = convolution.deconvolve(convolved)
    
    plt.subplot(233)
    plt.imshow(deconvolved, cmap = "gray")
    plt.title("Deconvolved image")
    plt.subplot(236)
    plt.imshow(np.abs(deconvolved - image), cmap = "gray")
    plt.title("Error")
    plt.show()
    
    print("Error", np.sum((deconvolved - image)**2)**0.5)
    print("Solved in", t2 - t1, "secs")
    print()