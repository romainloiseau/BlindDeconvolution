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
    
    def __init__(self, image, kernel):
        self.image = image.astype(np.float)
        self.kernel = np.array(kernel).astype(np.float)
        self.kernelFT = self.computeKernelFT()
        
    def computeKernelFT(self):
        if(not self.kernel.shape == self.image.shape):
            if(self.kernel.shape == (3, 3)):
                k = np.zeros(self.image.shape)
                
                for i in range(3):
                    for j in range(3):
                        k[(i - 1) % self.image.shape[0], (j - 1) % self.image.shape[1]] = self.kernel[i, j]
                
                self.kernel = k
        
        return np.real(fft2(self.kernel))
        
    def convolve(self):
        return np.real(ifft2(fft2(self.image) * self.kernelFT))
    
    def deconvolve(self):
        if(PARAMS["deconvolution"]["algorithm"] == "L2"):
            return self.l2()
        if(PARAMS["deconvolution"]["algorithm"] == "Sobolev"):
            return self.sobolev()
        else:
            print("ERROR !! No algorithm ...     Solver : ", PARAMS["deconvolution"]["algorithm"])
    
    def l2(self):
        return np.real(ifft2(fft2(self.image) * self.kernelFT / (abs(self.kernelFT)**2 + PARAMS["deconvolution"]["l2"]["lambda"])))
    
    def sobolev(self):
        n = self.image.shape[0]
        x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
        [Y, X] = np.meshgrid(x, x)
        S = (X**2 + Y**2) * (2/n)**2
        return np.real(ifft2(fft2(self.image) * self.kernelFT / (abs(self.kernelFT)**2 + S*PARAMS["deconvolution"]["sobolev"]["lambda"] + 10**(-15))))
    
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
    convolution = Convolution(image, kernel)
    convolved = convolution.convolve()
    t2 = time.time()
    
    plt.subplot(235)
    plt.imshow(convolution.kernelFT, cmap = "gray")
    plt.title("Kernel fourier")
    
    plt.subplot(232)
    plt.imshow(convolved, cmap = "gray")
    plt.title("Convolved image")
    
    deconvolved = Convolution(convolved, kernel).deconvolve()
    
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