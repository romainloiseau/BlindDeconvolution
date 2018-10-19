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
        for i in range(self.N):
            E = mu[i]**2 + C[i]
            tempq = pi * np.exp(- E / (2 * sigma**2)) / sigma
            tempq /= (np.sum(tempq) + 10**(-6))
            self.q[i] = tempq
        
        