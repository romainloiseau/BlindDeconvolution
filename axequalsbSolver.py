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
            self.kernel = dico["kernel"]
            self.image = dico["image"]
            self.Wgamma = dico["Wgammadiag"]
            self.factor = dico["factor"]
            self.convolution = Convolution(self.image.shape, self.kernel)
            self.b = self.factor * self.convolution.convolve(self.image)
        
    def solve(self):
        if(PARAMS["axequalsbSolver"]["algorithm"] == "conjugateGradient"):
            return self.conjugateGradient()
        else:
            print("ERROR !! No algorithm ...     Solver : ", PARAMS["axequalsbSolver"]["algorithm"])
    
    def multiplyA(self, x):
        if(self.option == "matrix"):
            return self.A.dot(x)
        if(self.option == "updateMu"):
            raise ValueError("C'est la merde ...")
        
    def conjugateGradient(self):
        
        def step(k, x, r, p):
            alpha = np.sum(r**2) / np.sum(p * self.multiplyA(p))
            x_ = x + alpha * p
            r_ = r - alpha * self.multiplyA(p)
            beta = np.sum(r_**2) / np.sum(r**2)
            p_ = r_ + beta * p
            return k+1, x_, r_, p_
        
        x = np.zeros(len(self.b))
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
