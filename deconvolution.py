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
from kernel import Kernel
import yaml
import opencv_utils as mycv2

PARAMS = yaml.load(open("params.yaml"))

class Deconvolution:
    
    def __init__(self, image, kernel):
        self.image = image
        self.kernel = kernel
        self.computeKernelFT()
        
    def computeKernelFT(self):
        padx = self.image.shape[0]-self.kernel.shape[0]
        pady = self.image.shape[1]-self.kernel.shape[1]
        self.kernelPadded = np.pad(self.kernel,
                                   ((int(np.floor(padx/2)),int(np.ceil(padx/2))),(int(np.floor(pady/2)),int(np.ceil(pady/2)))),
                                   'constant')
        
        self.kernelFT = np.real(fft2(self.kernelPadded))
        
    def solve(self):
        if(PARAMS["deconvolution"]["algorithm"] == "Sobolev"):
            return self.sobolev()
        else:
            print("ERROR !! No algorithm ...     Solver : ", PARAMS["deconvolution"]["algorithm"])
    
    def sobolev(self):
        # FROM TP3
        x = np.concatenate( (np.arange(0,self.image.shape[0]/2), np.arange(-self.image.shape[0]/2,0)) );
        [Y, X] = np.meshgrid(x, x)
        S = (X**2 + Y**2) * (2/self.image.shape[0])**2
        
        return np.real(ifft2(fft2(self.image) * self.kernelFT / (abs(self.kernelFT)**2 + PARAMS["deconvolution"]["lambda"] * S + 10**(-10))))
    
def runTests():
    image = mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"]))
    plt.imshow(image, cmap = "gray")
    plt.show()
    
    x = np.concatenate( (np.arange(0,10/2), np.arange(-10/2,0)) );
    [Y, X] = np.meshgrid(x, x)
    kernel = np.exp((-X**2-Y**2)/ (2*0.75**2))
    kernel = kernel/sum(kernel.flatten())

    plt.imshow(kernel, cmap = "gray")
    plt.show()
    
    convolved = Kernel(kernel).convolveScipy(image)
    plt.imshow(convolved, cmap = "gray")
    plt.show()
    
    deconv = Deconvolution(convolved, kernel)
    
    plt.imshow(deconv.kernelPadded, cmap = "gray")
    plt.show()
    plt.imshow(deconv.kernelFT, cmap = "gray")
    plt.show()
    
    plt.imshow(deconv.solve(), cmap = "gray")
    plt.show()
    
runTests()