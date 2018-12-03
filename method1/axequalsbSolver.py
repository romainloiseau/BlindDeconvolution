# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:47:50 2018

@author: romai
"""

import numpy as np
import time as time
from convolution import Convolution
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
import opencv_utils as mycv2
import cv2
import yaml

PARAMS = yaml.load(open("params.yaml"))

class AxequalsbSolver:
    
    def __init__(self, dico, option = "matrix"):
        self.option = option
        self.maxite = 10**6
        if(self.option == "matrix"):
            self.A = dico["A"]
            self.b = dico["b"]
            
        elif(self.option == "updatex"):
            self.image = dico["image"]
            self.kernel = dico["kernel"]
            """
            print(self.mask[:int(self.kernel.shape[0] / 2.), :].shape)
            print(self.mask[-int(self.kernel.shape[0] / 2.):, :].shape)
            plt.imshow(self.mask)
            plt.show()
            """
            self.weightpen = dico["weightpen"]
            self.w = dico["w"]
            self.maxite = PARAMS["axequalsbSolver"]["maxite"]
            self.convo = Convolution(self.image.shape, self.kernel)
            self.b = self.convo.convolve(self.image, mode = "adjoint").flatten()
            
        else:raise ValueError("No valid option for multiplyA ... current: " + str(self.option) + ". Possible choices: {matrix, updatex}")
            
            
    def solve(self, init = None):
        if(PARAMS["axequalsbSolver"]["algorithm"] == "conjugateGradient"):return self.conjugateGradient(init)
        else:raise ValueError("No valid algorithm for AxequalsbSolver ... current = " + str(PARAMS["axequalsbSolver"]["algorithm"]) + ". Possible choices: {conjugateGradient}")
    
    def multiplyA(self, x):
        if(self.option == "matrix"):return self.A.dot(x)
        elif(self.option == "updatex"):
            reshapedx = x.reshape(self.image.shape)
            
            withKernel = self.convo.convolve(self.convo.convolve(reshapedx), mode = "adjoint")
            #withKernel = np.real(ifft2(fft2(reshapedx) * self.convo.kernelFT * np.conjugate(self.convo.kernelFT).transpose()))
            
            withGamma = self.weightpen * self.w * reshapedx
            return (withKernel + withGamma).flatten()
        else:
            raise ValueError("No valid option for multiplyA ... current: " + str(self.option) + ". Possible choices: {matrix, updatex}")
        
    def conjugateGradient(self, init = None):
        
        def step(k, x, r, p):
            sumr2 = np.sum(r**2)
            Adotp = self.multiplyA(p)
            alpha = sumr2 / np.sum(p * Adotp)
            x_ = x + alpha * p
            r_ = r - alpha * Adotp
            beta = np.sum(r_**2) / sumr2
            p_ = r_ + beta * p
            return k+1, x_, r_, p_
        
        if(init is None):
            if(self.option == "matrix"):
                x = np.random.random(len(self.b))
            elif(self.option == "updatex"):
                x = self.image.copy().flatten()
        else:
            x = init.copy().flatten()
            
        r = self.b - self.multiplyA(x)
        self.error = np.sum(r**2)**0.5
        if(PARAMS["verbose"]):print("AxequalsbSolver initial error :", self.error)
        
        p = r.copy()
        self.ite = 0
        
        while(self.error > PARAMS["axequalsbSolver"]["epsilon"]):
            if(self.ite < self.maxite):
                self.ite, x, r, p = step(self.ite, x, r, p)
                self.error = np.sum(r**2)**0.5
            else:
                break
        if(PARAMS["verbose"]):print("AxequalsbSolver final error :  ", self.error)   
        return x
    
def runTests():
    print("AxequalsbSolver tests")
    
    A = np.array([[1, 0.1], [0.1, 2]])
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
    
    #PARAMS["verbose"] = False
    
    k = np.array([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [.25, .5, 1., .5, .25],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]])
    k /= np.sum(k)
    plt.imshow(k, cmap = "gray")
    plt.show()
    I = cv2.resize(mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"])), (64, 64))
    
    plt.figure(figsize = (9, 9))
    plt.subplot(221)
    plt.imshow(I, cmap = "gray")
    #plt.hist(I.flatten())
    plt.title("true")
    plt.axis("off")
    
    convo = Convolution(I.shape, k)
    Ik = convo.convolve(I)
    
    plt.subplot(222)
    plt.imshow(Ik, cmap = "gray")
    #plt.hist(Ik.flatten())
    plt.title("convolved")
    plt.axis("off")
    
    print("INITIAL ERROR :                        ", np.sqrt(np.sum((I - Ik)**2)))
    
    #Direct resolution
    mu1 = convo.deconvolve(Ik)
    plt.subplot(223)
    plt.imshow(mu1, cmap = "gray")
    #plt.hist(mu1.flatten())
    plt.title("direct resolution")
    plt.axis("off")
    
    print("ERROR - fourier                        ", np.sqrt(np.sum((I.flatten() - mu1.flatten())**2)))
    
    #Conjuguate gradient with overloaded function multiplyA with fourier
    mu3solver = AxequalsbSolver({
            "image": Ik,
            "kernel": k,
            "w": np.zeros(Ik.shape),
            "weightpen" : 0.}, option = "updatex")
    mu3 = mu3solver.solve().reshape(Ik.shape)
    
    print("ERROR - conjuguate gradient fourier    ", np.sqrt(np.sum((I.flatten() - mu3.flatten())**2)))
    print("nite", mu3solver.ite)
    
    plt.subplot(224)
    plt.imshow(mu3, cmap = "gray")
    #plt.hist(mu3.flatten())
    plt.title("solver")
    plt.axis("off")
    plt.show()
    