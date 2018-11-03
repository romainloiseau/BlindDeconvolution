# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:46:27 2018

@author: romai
"""

import numpy as np
from convolution import Convolution
from kernel import Kernel
from axequalsbSolver import AxequalsbSolver
import matplotlib.pyplot as plt
import yaml

PARAMS = yaml.load(open("params.yaml"))

class Data:
    
    def __init__(self, y, derivativeSpace = False, truek = np.nan, truex = np.nan):
        
        self.y = y.copy() / 256.
        self.x = y.copy() / 256.
        
        self.checkx = not np.isnan(truex.all())
        if(self.checkx):
            self.truex = truex / 256.
            
        self.checkk = not np.isnan(truek.all())
        if(self.checkk):
            self.truek = truek
            
        self.derivativeSpace = derivativeSpace
        
        if(self.derivativeSpace):
            self.filters = [PARAMS["derivativefilters"][k] for k in PARAMS["derivativefilters"]]
            self.nfilters = len(self.filters)
            convs = [Convolution(y.shape, f) for f in self.filters]
            self.dy = [conv.convolve(self.y) for conv in convs]
            self.dx = self.dy.copy()
            plt.hist(self.dx[0].flatten())
            plt.show()
            if(self.checkx):
                self.truedx = [conv.convolve(self.truex) for conv in convs]
    
    def initialize(self):
        #Initialize shapes
        self.N1, self.N2 = self.x.shape[0], self.x.shape[1]
        self.N = self.N1 * self.N2
        
        #Noise
        self.signoise = PARAMS["freeEnergy"]["eta"]
        
        #Our prior is a mixture of J gaussian (MOG) with weights pi, mean 0, and standard deviation sigma
        self.pi = np.array(PARAMS["freeEnergy"]["pis"])
        self.pi /= self.pi.sum()
        self.sigma = 1 / np.array(PARAMS["freeEnergy"]["ivars"])**0.5
        self.J = len(self.sigma)
        
        #Weights
        self.C = np.sum(self.sigma**2) * np.ones(self.N)
        if(self.derivativeSpace):
            self.C = [self.C.copy() for i in range(self.nfilters)]

        #Initialize kernel
        self.M = 2 * int(PARAMS["freeEnergy"]["M"]/2) + 1
        self.k = np.zeros((self.M, self.M))
        self.k[int(self.M / 2.), int(self.M / 2.)] = 1
        self.k[int(self.M / 2.), int(self.M / 2.) + 1] = 1
        self.k /= self.k.sum()
        
        print()
        print("Expected number of pyramids", max(np.floor(np.log(5/self.M)/np.log(0.5**0.5)), 0))
        print()
            
    def deconv(self):
        self.initialize()
        for i in range(PARAMS["freeEnergy"]["Niter"]):
            if(PARAMS["verbose"]):print()
            self.computeIteration()
            
    def computeIteration(self):
        self.update_x()
        self.update_k()
        
    def update_x(self):
        if(self.derivativeSpace):
            for i in range(self.nfilters):
                self.dx[i], self.C[i] = self.update_specx(self.dx[i], self.C[i])
        else:
            self.x, self.c = self.update_specx(self.x, self.C)
            
        if(PARAMS["verbose"]):
            print("Updated x ...")
            if(self.checkx):
                if(self.derivativeSpace):
                    plt.figure()
                    for i in range(self.nfilters):
                        plt.subplot(201 + 10 * self.nfilters + 2 * i)
                        plt.imshow(self.dx[i], cmap = "gray")
                        plt.title("error " + str(np.sum((self.dx[i] - self.truedx[i])**2)**0.5))
                        plt.subplot(202 + 10 * self.nfilters + 2 * i)
                        plt.hist(self.dx[i].flatten())
                    plt.show()
                else:
                    plt.imshow(self.x, cmap = "gray")
                    plt.title("error " + str(np.sum((self.x - self.truex)**2)**0.5))
                    plt.show()
                
    def update_specx(self, x, c):
        E = x.flatten()**2 + c
        
        q = np.zeros((self.N, self.J))
        for i in range(self.N):
            logq_i = - 0.5 * E[i] / self.sigma**2 + np.ones(self.J) * (np.log(self.pi) - np.log(self.sigma))
            q_i = np.exp(logq_i - np.max(logq_i))
            q_i /= q_i.sum()
            """
            if(i < 3):
                print(E[i], x.flatten()[i]**2, c[i])
                print(- 0.5 * E[i] / self.sigma**2)
                print(np.log(self.pi))
                print(- np.log(self.sigma))
                print(logq_i)
                print(q_i)
                print()
            """
            q[i] = q_i.copy()
        w = ((q / self.sigma**2).sum(axis = -1)).reshape(self.N1, self.N2)
        
        x = AxequalsbSolver({
                "image": x,
                "kernel": self.k,
                "weightpen": self.signoise**2,
                "w": w
                }, option = "updatex").solve()
        
        da1 = (1 / self.signoise**2) * Convolution((self.N1, self.N2), self.k ** 2).convolve(np.ones((self.N1, self.N2)));
        xcov = 1. / (da1 + w);
        return x.reshape(self.N1, self.N2), xcov.flatten()
            
    def update_k(self):
        if(self.derivativeSpace):
            x = self.dx[0]
            y = self.dy[0]
            c = self.C[0]
        else:
            x = self.x
            y = self.y
            c = self.C
        
        Ak, bk = Kernel(self.k).getAkbk(y, x, np.diag(c))
        self.k = AxequalsbSolver({"A": Ak, "b": bk}).solve().reshape((self.M, self.M))
        self.k /= np.sum(np.abs(self.k))
        
        if(PARAMS["verbose"]):
            print("Updated k ...")
            print(self.k)
            if(self.checkk):
                print("k error         ", np.sum((self.k - self.truek)**2)**0.5)
        