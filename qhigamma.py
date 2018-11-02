# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:45:40 2018

@author: romai
"""

import numpy as np

class Qhigamma:
    
    def __init__(self, N, J):
        self.N = N
        self.q = np.zeros((N, J))
        
    def getQ(self):
        return self.q
        
    def update(self, pi, sigma, mu, C):
        E = mu**2 + C
        print("np.mean(E)    ", np.mean(E), "              np.mean(sigma)", np.mean(sigma))
        print("np.mean(mu**2)", np.mean(mu**2), "              np.mean(C)", np.mean(C))
        print("Qhigamma update         E               np.sum(tempq)")
        for i in range(self.N):
            
            tempq = pi * np.exp(- E[i] / (2 * sigma**2)) / sigma
            if(i < 3):
                print("Qhigamma update", E[i], "      ", np.sum(tempq))
            tempq /= np.sum(tempq)
            self.q[i] = tempq.copy()