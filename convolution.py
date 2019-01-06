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

#A class to compute convolutions and deconvolutions
class Convolution:
    
    def __init__(self, inputShape, kernel):
        self.kernel = np.array(kernel).astype(np.float)
        self.ksz1, self.ksz2 = self.kernel.shape
        self.inputShape = inputShape
        self.kernelFT = self.computeKernelFT()
        
    def computeKernelFT(self):
        if(not self.kernel.shape == self.inputShape):
            if(self.kernel.shape[0] == self.kernel.shape[1]):
                k = np.zeros(self.inputShape)
                for i in range(self.ksz1):
                    for j in range(self.ksz2):
                        k[(i - int(self.ksz1 / 2.)) % self.inputShape[0], (j - int(self.ksz2 / 2.)) % self.inputShape[1]] = self.kernel[i, j]
                
                self.kernel = k
        return fft2(self.kernel)
        
    def convolve(self, image, mode = "direct"):
        if(image.shape != self.inputShape):raise ValueError("The convolution (shape = " + str(self.inputShape) + ") is not meant to be used with this image shape (" + str(image.shape) + ")")
        if(mode == "direct"):return np.real(ifft2(fft2(image) * self.kernelFT))
        elif(mode == "adjoint"):return np.real(ifft2(fft2(image) * (np.conjugate(self.kernelFT)).transpose()))
        else:raise ValueError("Mode for convolve must be either 'direct' or 'adjoint'. You call for convolve with mode = " + str(mode))
    
    def deconvolve(self, image, mode = "direct"):
        if(PARAMS["deconvolution"]["algorithm"] == "L2"):return self.l2(image, mode)
        if(PARAMS["deconvolution"]["algorithm"] == "Sobolev"):return self.sobolev(image, mode)
        else:raise ValueError("ERROR !! No algorithm ...     Solver : ", PARAMS["deconvolution"]["algorithm"])
    
    def l2(self, image, mode):
        if(mode == "direct"):kFT = self.kernelFT
        elif(mode == "adjoint"):kFT = np.conjugate(self.kernelFT).transpose()
        else:raise ValueError("Mode for convolve must be either 'direct' or 'adjoint'. You call for convolve with mode = " + str(mode))
        return np.real(ifft2(fft2(image) * kFT / (abs(kFT)**2 + PARAMS["deconvolution"]["l2"]["lambda"])))
    
    def sobolev(self, image, mode):
        if(mode == "direct"):kFT = self.kernelFT
        elif(mode == "adjoint"):kFT = np.conjugate(self.kernelFT).transpose()
        else:raise ValueError("Mode for convolve must be either 'direct' or 'adjoint'. You call for convolve with mode = " + str(mode))
        n = self.inputShape[0]
        x = np.concatenate( (np.arange(0,n/2), np.arange(-n/2,0)) );
        [Y, X] = np.meshgrid(x, x)
        S = (X**2 + Y**2) * (2/n)**2
        return np.real(ifft2(fft2(image) * kFT / (abs(kFT)**2 + S*PARAMS["deconvolution"]["sobolev"]["lambda"] + 10**(-15))))
    
def runTests():
    
    kernel = np.array([[.2, .5, .2], [.5, 1., .5], [.2, .5, .2]])
    kernel /= np.sum(kernel)
    convolution = Convolution((10, 10), kernel)
    
    plt.figure(figsize = (15, 5))
    plt.subplot(131)
    plt.imshow(kernel, norm = None, cmap = "gray")
    plt.axis("off")
    plt.title("Original kernel")
    plt.subplot(132)
    plt.imshow(convolution.kernel, cmap = "gray")
    plt.axis("off")
    plt.title("Resized kernel")
    plt.subplot(133)
    plt.imshow(np.abs(convolution.kernelFT).astype(np.float), cmap = "gray")
    plt.axis("off")
    plt.title("Kernel Fourier transform")
    plt.show()
    
    print("Convolution tests")
    image = mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"]))
    
    plt.figure(figsize = (9, 6))
    
    plt.subplot(231)
    plt.imshow(image, cmap = "gray")
    plt.title("Original image")

    #kernel = PARAMS["derivativefilters"]["dh"]
    
    kernel = np.array([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [.25, .5, 1., .5, .25],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]])
    kernel /= np.sum(kernel)
    
    plt.subplot(234)
    plt.imshow(kernel, cmap = "gray")
    plt.title("Kernel")
    
    t1 = time.time()
    convolution = Convolution(image.shape, kernel)
    convolved = convolution.convolve(image)
    t2 = time.time()
    
    plt.subplot(235)
    plt.imshow(np.abs(convolution.kernelFT).astype(np.float), cmap = "gray")
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