# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:47:50 2018

@author: romai
"""

import numpy as np
import yaml

PARAMS = yaml.load(open("params.yaml"))

class AxequalsbSolver:
    
    def __init__(self, A, b):
        self.A = A
        self.b = b
        
    def solve(self):
        if(PARAMS["axequalsbSolver"]["algorithm"] == "conjugateGradient"):
            return self.conjugateGradient()
        else:
            print("ERROR !! No algorithm ...     Solver : ", PARAMS["axequalsbSolver"]["algorithm"])
    
    def conjugateGradient(self):
        
        def step(k, x, r, p):
            alpha = np.sum(r**2) / np.sum(p * self.A.dot(p))
            x_ = x + alpha * p
            r_ = r - alpha * self.A.dot(p)
            beta = np.sum(r_**2) / np.sum(r**2)
            p_ = r_ + beta * p
            return k+1, x_, r_, p_
        
        x = np.zeros(len(self.b))
        r = self.b - self.A.dot(x)
        p = r.copy()
        k = 0
        
        while(np.sqrt(np.sum(r**2)) > PARAMS["axequalsbSolver"]["epsilon"]): 
            k, x, r, p = step(k, x, r, p)
        print("solved")
        return x
    
def runTests():
    A = np.array([[1, 0.1], [0.2, 1]])
    b = np.array([2, 3])
    
    x = AxequalsbSolver(A, b).solve()
    print(x, A.dot(x) - b)