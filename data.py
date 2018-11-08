# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:46:27 2018

@author: romai
"""

import numpy as np
from convolution import Convolution
from kernel import Kernel
import correlation as corr
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
            if(y.shape != truex.shape):
                print("WARNING, truex shape different from initial image y shape, setting self.checkx to False")
                self.checkx = False
            else:
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
        #Initialize kernel
        self.M = 2 * int(PARAMS["freeEnergy"]["M"]/2) + 1
        if(self.checkk and self.M != self.truek.shape[0]):
            print("WARNING, self.truek shape different from researched kernel shape self.M, setting self.checkk to False")
            self.checkk = False
            
        self.k = np.zeros((self.M, self.M))
        self.k[int(self.M / 2.), int(self.M / 2.)] = 1
        
        #Initialize shapes
        self.N1, self.N2 = self.x.shape[0], self.x.shape[1]
        self.dN1, self.dN2 = int(self.M / 2.), int(self.M / 2.)
        self.N1e, self.N2e = self.N1 + 2 * self.dN1, self.N2 + 2 * self.dN2
        self.N, self.Ne = self.N1 * self.N2, self.N1e * self.N2e
        
        
        #Noise
        self.signoise = PARAMS["freeEnergy"]["eta"]
        
        #Our prior is a mixture of J gaussian (MOG) with weights pi, mean 0, and standard deviation sigma
        self.pi = np.array(PARAMS["freeEnergy"]["pis"])
        self.pi /= self.pi.sum()
        self.sigma = 1 / np.array(PARAMS["freeEnergy"]["ivars"])**0.5
        self.J = len(self.sigma)
        
        #Weights
        #self.C = np.sum(1 / self.sigma**2) * np.ones(self.Ne)
        self.C = np.zeros(self.Ne)
        if(self.derivativeSpace):
            self.C = [self.C.copy() for i in range(self.nfilters)]

        print()
        print("Expected number of pyramids", max(np.floor(np.log(5/self.M)/np.log(0.5**0.5)), 0))
        print()
            
    def deconv(self):
        self.initialize()
        if(PARAMS["verbose"]):
            print("Initial variables")
            self.print_x()
            self.print_k()
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
            self.x, self.C = self.update_specx(self.x, self.C)
        
        if(PARAMS["verbose"]):
            print("Updated x ...")   
            self.print_x()
          
    def print_x(self):
        if(self.derivativeSpace):
            plt.figure(figsize = (15, 3 * self.nfilters))
            for i in range(self.nfilters):
                print("np.min(self.dx[i]), np.max(self.dx[i])", np.min(self.dx[i]), np.max(self.dx[i]))
                plt.subplot(100 * self.nfilters + 31 + 3 * i)
                plt.imshow(self.dx[i], cmap = "gray")
                if(self.checkx):
                    plt.title("x, error " + str(np.sum((self.dx[i] - self.truedx[i])**2)**0.5))
                else:
                    plt.title("x")
                plt.subplot(100 * self.nfilters + 32 + 3 * i)
                plt.hist(self.dx[i].flatten(), bins = 40)
                plt.title("self.dx[i]")
                plt.subplot(100 * self.nfilters + 33 + 3 * i)
                plt.hist(self.C[i], bins = 40)
                plt.title("self.C[i]")
            plt.show()
        else:
            print("np.min(self.x), np.max(self.x)", np.min(self.x), np.max(self.x))
            plt.figure(figsize = (15, 3))
            plt.subplot(131)
            plt.imshow(self.x, cmap = "gray")
            if(self.checkx):
                plt.title("x, error " + str(np.sum((self.x - self.truex)**2)**0.5))
            else:
                plt.title("x")
            plt.subplot(132)
            plt.hist(self.x.flatten(), bins = 40)
            plt.title("self.x")
            plt.subplot(133)
            plt.hist(self.C, bins = 40)
            plt.title("self.C")
            plt.show()
                
    def update_specx(self, x, c):
        
        x = np.lib.pad(x, ((self.dN1, self.dN1), (self.dN2, self.dN2)), 'constant', constant_values=(0))
        
        E = x.flatten()**2 + c
        
        print(np.mean(E), np.mean(x.flatten()**2), np.mean(c))
        
        logq = np.zeros((self.J, self.Ne))
        
        for i in range(self.J):
            sigma_i = self.sigma[i]
            pi_i = self.pi[i]
            logq_i = - 0.5 * E / sigma_i**2 + np.ones(self.Ne) * (np.log(pi_i) - np.log(sigma_i))
            logq[i] = logq_i.copy()
    
        q = np.exp(logq - np.max(logq, axis = 0))
        q /= np.sum(q, axis = 0)
        
        w = (q.transpose() / self.sigma**2).sum(axis = -1).reshape(self.N1e, self.N2e)
        
        x = AxequalsbSolver({
                "image": x,
                "kernel": self.k,
                "weightpen": self.signoise**2,
                "w": w
                }, option = "updatex").solve()
    
        
        x = x.reshape(self.N1e, self.N2e)
        
        convk = Convolution((self.N1e, self.N2e), self.k)
        da1 = convk.convolve(convk.convolve(np.ones((self.N1e, self.N2e))), mode = "adjoint")
        #da1 *= 1 / self.signoise**2
        
        xcov = 1. / (da1 + w)
            
        return x[self.dN1:-self.dN1, self.dN2:-self.dN2], xcov.flatten()
    
    def update_k(self):
        if(self.derivativeSpace):
            x = self.dx[0]
            y = self.dy[0]
            c = self.C[0]
        else:
            x = self.x
            y = self.y
            c = self.C
        
        #Ak, bk = Kernel(self.k).getAkbk(y, x, c)
        Ak, bk = corr.getAkbk(y,
                              x,
                              c.reshape(self.N1e, self.N2e)[self.dN1:-self.dN1, self.dN2:-self.dN2],
                              self.k.shape)
        self.k = AxequalsbSolver({"A": Ak, "b": bk}).solve(self.k).reshape((self.M, self.M))
        #self.k /= np.sum(np.abs(self.k))
        
        if(PARAMS["verbose"]):
            print("Updated k ...")
            self.print_k()
        
    def print_k(self):
        print(self.k)
        if(self.checkk):
            print("k error         ", np.sum((self.k - self.truek)**2)**0.5)
        