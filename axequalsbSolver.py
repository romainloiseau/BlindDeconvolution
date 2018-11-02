# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:47:50 2018

@author: romai
"""

import numpy as np
import time as time
from convolution import Convolution
from kernel import Kernel
from scipy.fftpack import fft2, ifft2
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
            #self.gammasigma = dico["gammasigma"]
            #self.gammapi = dico["gammapi"]
            self.Wgamma = dico["Wgamma"]
            
            self.factorkernel = dico["factorkernel"]
            self.convolution = Convolution(self.image.shape, self.kernel)
            self.gammaConvolution = Convolution(self.image.shape, self.gammakernel)
            
            #withKernel = self.factorkernel * np.conjugate(self.convolution.kernelFT).transpose() * self.convolution.kernelFT
            #withGamma = np.conjugate(self.gammaConvolution.kernelFT).transpose() * fft2(self.Wgamma) * self.gammaConvolution.kernelFT
            #self.fourier = withKernel + withGamma
            
            #self.b = self.factorkernel * self.convolution.convolve(self.image, mode = "adjoint").flatten()
            self.b = np.real(ifft2(self.factorkernel * np.conjugate(self.convolution.kernelFT).transpose() * fft2(self.image)).flatten())
            
    def solve(self):
        if(PARAMS["axequalsbSolver"]["algorithm"] == "conjugateGradient"):
            return self.conjugateGradient()
        else:
            raise ValueError("ERROR !! No algorithm ...     Solver : ", PARAMS["axequalsbSolver"]["algorithm"])
    
    def multiplyA(self, x):
        if(self.option == "matrix"):
            return self.A.dot(x)
        
        elif(self.option == "updateMu"):
            
            #reshapedxFT = fft2(x.reshape(self.image.shape))
            reshapedx = x.reshape(self.image.shape)
            
            withKernel = self.factorkernel * self.convolution.convolve(self.convolution.convolve(reshapedx), mode = "adjoint")
            withGamma = self.gammaConvolution.convolve(self.Wgamma * self.gammaConvolution.convolve(reshapedx), mode = "adjoint")
            
            return (withKernel + withGamma).flatten()
            #return np.real(ifft2(self.fourier * reshapedxFT).flatten())
        
        else:
            raise ValueError("No valid option for multiplyA ... current = " + str(self.option))
        
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
        
        while(np.sqrt(np.sum(np.real(r * np.conjugate(r)))) > PARAMS["axequalsbSolver"]["epsilon"]):
            k, x, r, p = step(k, x, r, p)
            
        print("nstep", k)
        
        return np.real(x)
    
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
    
    k = np.random.random((3, 3))
    I = np.random.random((100, 100))
    
    Tk = Kernel(k).getToepliz(I.shape[0], I.shape[1])
    
    Ak = Tk.transpose().dot(Tk)
    bk = Tk.transpose().dot(I.flatten())
    mu1 = AxequalsbSolver({"A": Ak, "b": bk}).solve()
    
    convo = Convolution(I.shape, k)
    
    bI = np.real(ifft2(fft2(I) * np.conjugate(convo.kernelFT).transpose()))
    mu2 = np.real(ifft2(fft2(bI) * convo.kernelFT * np.conjugate(convo.kernelFT).transpose()/ (np.abs(convo.kernelFT * np.conjugate(convo.kernelFT).transpose())**2))).flatten()
    
    print("ERROR", np.sqrt(np.sum((mu1 - mu2)**2)), np.mean(np.abs(mu1 - mu2)))
    
    mu3 = AxequalsbSolver({
            "image": I,
            "kernel": k,
            "gammakernel": np.zeros(k.shape),
            "Wgamma": np.zeros(I.shape),
            "factorkernel" : 1
            }, option = "updateMu").solve()
    
    print("ERROR", np.sqrt(np.sum((mu1 - mu3)**2)), np.mean(np.abs(mu1 - mu3)))