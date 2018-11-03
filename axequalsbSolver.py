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
            self.maxite = 10**9
            self.A = dico["A"]
            self.b = dico["b"]
        elif(self.option == "updatex"):
            self.maxite = PARAMS["axequalsbSolver"]["maxite"]
            self.image = dico["image"]
            self.kernel = dico["kernel"]
            self.weightpen = dico["weightpen"]
            self.w = dico["w"]
            
            self.convolution = Convolution(self.image.shape, self.kernel)
            self.b = self.convolution.convolve(self.image, mode = "adjoint").flatten()
        
        else:    
            raise ValueError("No valid option for multiplyA ... current: " + str(self.option) + ". Possible choices: {matrix, updatex}")
            
            
    def solve(self):
        if(PARAMS["axequalsbSolver"]["algorithm"] == "conjugateGradient"):
            return self.conjugateGradient()
        else:
            raise ValueError("No valid algorithm for AxequalsbSolver ... current = " + str(PARAMS["axequalsbSolver"]["algorithm"]) + ". Possible choices: {conjugateGradient}")
    
    def multiplyA(self, x):
        if(self.option == "matrix"):
            return self.A.dot(x)
        
        elif(self.option == "updatex"):
            
            reshapedx = x.reshape(self.image.shape)
            
            withKernel = self.convolution.convolve(self.convolution.convolve(reshapedx), mode = "adjoint")
            withGamma = self.weightpen * self.w * reshapedx
            return (withKernel + withGamma).flatten()
        
        else:
            raise ValueError("No valid option for multiplyA ... current: " + str(self.option) + ". Possible choices: {matrix, updatex}")
        
    def conjugateGradient(self):
        
        
        def step(k, x, r, p):
            sumr2 = np.sum(r**2)
            Adotp = self.multiplyA(p)
            
            alpha = sumr2 / np.sum(p * Adotp)
            x_ = x + alpha * p
            r_ = r - alpha * Adotp
            beta = np.sum(r_**2) / sumr2
            #if(beta > 1):print("WARNING, beta > 1 for conjuguate gradient method", sumr2, np.sum(r_**2))
            p_ = r_ + beta * p
            return k+1, x_, r_, p_
        
        if(self.option == "matrix"):
            x = np.random.random(len(self.b)) * np.sum(self.b) / np.sum(self.A)
        elif(self.option == "updatex"):
            x = self.image.copy().flatten()
            
        r = self.b - self.multiplyA(x)
        if(PARAMS["verbose"]):print("AxequalsbSolver initial error :", np.sum(r**2))
        p = r.copy()
        k = 0
        
        while(np.sum(r**2) > PARAMS["axequalsbSolver"]["epsilon"]):
            
            if(k < self.maxite):
                k, x, r, p = step(k, x, r, p)
            else:
                break
        if(PARAMS["verbose"]):print("AxequalsbSolver final error :  ", np.sum(r**2))   
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
            "w": np.zeros(I.shape),
            "weightpen" : 0
            }, option = "updatex").solve()
    
    print("ERROR", np.sqrt(np.sum((mu1 - mu3)**2)), np.mean(np.abs(mu1 - mu3)))