# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:47:50 2018

@author: romai
"""

import numpy as np
import time as time
from convolution import Convolution
import yaml

PARAMS = yaml.load(open("params.yaml"))

class AxequalsbSolver:
    
    def __init__(self, dico, option = "matrix"):
        self.option = option
        if(self.option == "matrix"):
            self.A = dico["A"]
            self.b = dico["b"]
        if(self.option == "updateMu"):
            self.image = dico["image"]
            self.kernel = dico["kernel"]
            self.gammakernel = dico["gammakernel"]
            self.Wgamma = dico["Wgammadiag"].reshape(self.image.shape)
            self.factorkernel = dico["factorkernel"]
            self.convolution = Convolution(self.image.shape, self.kernel)
            self.gammaConvolution = Convolution(self.image.shape, self.gammakernel)
            self.b = self.factorkernel * self.convolution.convolve(self.image, mode = "adjoint").flatten()
            
    def solve(self):
        if(PARAMS["axequalsbSolver"]["algorithm"] == "conjugateGradient"):
            return self.conjugateGradient()
        else:
            print("ERROR !! No algorithm ...     Solver : ", PARAMS["axequalsbSolver"]["algorithm"])
    
    def multiplyA(self, x):
        if(self.option == "matrix"):
            return self.A.dot(x)
        if(self.option == "updateMu"):
            
            reshapedx = x.reshape(self.image.shape)
            
            withKernel = self.factorkernel * self.convolution.convolve(self.convolution.convolve(reshapedx), mode = "adjoint")
            withGamma = self.gammaConvolution.convolve(self.Wgamma * self.gammaConvolution.convolve(reshapedx), mode = "adjoint")
            
            return (withKernel + withGamma).flatten()
        
    def conjugateGradient(self):
        
        def step(k, x, r, p):
            sumr2 = np.sum(r**2)
            Adotp = self.multiplyA(p)
            alpha = sumr2 / np.sum(p * Adotp)
            x_ = x + alpha * p
            r_ = r - alpha * Adotp
            beta = np.sum(r_**2) / sumr2
            p_ = r_ + beta * p
            return k+1, x_, r_, p_
        
        x = np.random.random(len(self.b))
        r = self.b - self.multiplyA(x)
        p = r.copy()
        k = 0
        
        while(np.sqrt(np.sum(r**2)) > PARAMS["axequalsbSolver"]["epsilon"]): 
            k, x, r, p = step(k, x, r, p)
            
        return x
    
def runTests():
    print("AxequalsbSolver tests")
    
    A = np.array([[1, 0.1], [0.2, 1]])
    b = np.array([2, 3])
    t = time.time()
    x = AxequalsbSolver({"A": A, "b": b}).solve()
    
    print("A")
    print(A)
    print("b", b)
    print("x", x)
    print("Error", ((A.dot(x) - b)**2).sum()**0.5)
    print("Solved in", time.time() - t, "secs")
    print()
