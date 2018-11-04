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
        self.maxite = 10**9
        if(self.option == "matrix"):
            self.A = dico["A"]
            self.b = dico["b"]
            
        elif(self.option == "updatex"):
            self.image = dico["image"]
            self.kernel = dico["kernel"]
            self.weightpen = dico["weightpen"]
            self.w = dico["w"]
            
            self.convo = Convolution(self.image.shape, self.kernel)
            self.b = self.convo.convolve(self.image, mode = "adjoint").flatten()
            
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
            
            #withKernel = self.convo.convolve(self.convolution.convolve(reshapedx), mode = "adjoint")
            withKernel = np.real(ifft2(fft2(reshapedx) * self.convo.kernelFT * np.conjugate(self.convo.kernelFT).transpose()))
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
            x = np.random.random(len(self.b))
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
    
    A = np.random.random((100, 100))
    A = A.transpose().dot(A)
    b = np.random.random((100))
    t = time.time()
    x = AxequalsbSolver({"A": A, "b": b}).solve()
    
    print("Error", ((A.dot(x) - b)**2).sum()**0.5)
    print("||x||", (x**2).sum()**0.5)
    print("Solved in", time.time() - t, "secs")
    print()
    
    PARAMS["verbose"] = False
    
    k = np.random.random((3, 3))
    I = np.random.random((100, 100))
    
    convo = Convolution(I.shape, k)
    Ik = convo.convolve(I)
    
    print("INITIAL ERROR :", np.sqrt(np.sum((I - Ik)**2)))
    print()
    
    bI = np.real(ifft2(fft2(Ik) * np.conjugate(convo.kernelFT).transpose()))
    mu1 = np.real(ifft2(fft2(bI) * convo.kernelFT * np.conjugate(convo.kernelFT).transpose()/ (np.abs(convo.kernelFT * np.conjugate(convo.kernelFT).transpose())**2))).flatten()
    
    print("ERROR - fourier                                     ", np.sqrt(np.sum((I.flatten() - mu1)**2)))
     
    
    Tk = Kernel(k).getToepliz(Ik.shape[0], Ik.shape[1])
    Ak = Tk.transpose().dot(Tk)
    bk = Tk.transpose().dot(Ik.flatten())
    mu2 = AxequalsbSolver({"A": Ak, "b": bk}).solve()
    print("ERROR - conjuguate gradient toepliz                 ", np.sqrt(np.sum((I.flatten() - mu2)**2)))
    
    
    mu3solver = AxequalsbSolver({
            "image": Ik,
            "kernel": k,
            "w": np.zeros(Ik.shape),
            "weightpen" : 0
            }, option = "updatex")
    
    mu3 = mu3solver.solve()
    
    print("ERROR - conjuguate gradient fourier                 ", np.sqrt(np.sum((I.flatten() - mu3)**2)))
    print("np.sqrt(np.sum((bI.flatten() - mu3solver.b)**2))    ", np.sqrt(np.sum((bI.flatten() - mu3solver.b)**2)))