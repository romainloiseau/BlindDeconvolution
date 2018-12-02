# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:46:27 2018

@author: romai
"""

import numpy as np
from convolution import Convolution
import correlation as corr
from axequalsbSolver import AxequalsbSolver
import matplotlib.pyplot as plt
import yaml

from cvxopt import matrix, solvers


PARAMS = yaml.load(open("params.yaml"))

class Data:
    
    def __init__(self, y, derivativeSpace = False, truek = None, truex = None):
        
        np.set_printoptions(precision=4)
        
        self.y = y.copy() / 255.
        self.x = y.copy() / 255.
        
        self.checkx = not truex is None
        if(self.checkx):
            if(y.shape != truex.shape):
                print("WARNING, truex shape different from initial image y shape, setting self.checkx to False")
                self.checkx = False
            else:self.truex = truex / 256.
            
        self.checkk = not truek is None
        if(self.checkk):
            self.truek = truek
            self.nonblinddeconvx = Convolution(self.y.shape, self.truek).deconvolve(self.y)
        
        self.derivativeSpace = derivativeSpace
        if(self.derivativeSpace):self.computeInitialDerivatives()
        
    def computeInitialDerivatives(self):
        self.filters = [PARAMS["derivativefilters"][k] for k in PARAMS["derivativefilters"]]
        self.nfilters = len(self.filters)
        convs = [Convolution(self.y.shape, f) for f in self.filters]
        self.filty = [conv.convolve(self.y) for conv in convs]
        self.filtx = self.filty.copy()
        
        plt.hist(self.filtx[0].flatten())
        plt.show()
        
        if(self.checkx):self.truefiltx = [conv.convolve(self.truex) for conv in convs]
    
    def initialize(self):
        #Initialize kernel
        self.M = 2 * int(PARAMS["freeEnergy"]["M"]/2) + 1
        if(self.checkk and self.M != self.truek.shape[0]):
            print("WARNING, self.truek shape different from researched kernel shape self.M, setting self.checkk to False")
            self.checkk = False
            
        self.k = np.zeros((self.M, self.M))
        self.k[int(self.M / 2.), int(self.M / 2.)] = 1
        #self.k[int(self.M / 2.), int(self.M / 2.) + 1] = 1
        self.k /= np.sum(self.k)
        if(self.checkk):self.k_err = []
        
        #Initialize shapes
        self.N1, self.N2 = self.x.shape[0], self.x.shape[1]
        self.dN1, self.dN2 = int(self.M / 2.), int(self.M / 2.)
        self.N1e, self.N2e = self.N1 + 2 * self.dN1, self.N2 + 2 * self.dN2
        self.N, self.Ne = self.N1 * self.N2, self.N1e * self.N2e
        
        #Noise
        self.signoise_v = PARAMS["freeEnergy"]["eta"] * 1.15 ** np.arange(PARAMS["freeEnergy"]["Niter"] - 1, -1, -1)
        print(self.signoise_v)
        #Our prior is a mixture of J gaussian (MOG) with weights pi, mean 0, and standard deviation sigma
        self.pi = np.array(PARAMS["freeEnergy"]["pis"])
        self.pi /= self.pi.sum()
        self.sigma = 1 / np.array(PARAMS["freeEnergy"]["ivars"])**0.5
        self.J = len(self.sigma)
        
        #Weights
        self.C = np.zeros(self.Ne)
        if(self.derivativeSpace):
            self.C = [self.C.copy() for i in range(self.nfilters)]
        
        print()
        print("Expected number of pyramids", max(int(np.floor(np.log(5/self.M)/np.log(0.5**0.5))), 0))
        print()
            
    def deconv(self):
        self.initialize()
        if(PARAMS["verbose"]):
            print("Initial variables")
            self.print_x()
            self.print_k()
            self.showEvolution()
        for i in range(PARAMS["freeEnergy"]["Niter"]):
            self.signoise = self.signoise_v[i]
            self.computeIteration(i)
            
    def computeIteration(self, iteration):
        if(PARAMS["verbose"]):
            print()
            print("##################" + len(str(iteration)) * "#")
            print("### Iteration " + str(iteration) + " ###")
            print("##################" + len(str(iteration)) * "#")
        self.update_x(iteration)
        self.update_k(iteration)
        self.showEvolution()
        
    def showEvolution(self):
        if(self.checkx):
            plt.figure(figsize = (16, 4))
            plt.subplot(141)
            plt.imshow(self.y, cmap = "gray")
            plt.axis('off')
            plt.title("y")
            plt.subplot(142)
            plt.imshow(self.x, cmap = "gray")
            plt.axis('off')
            plt.title("x")
            plt.subplot(143)
            plt.imshow(self.nonblinddeconvx, cmap = "gray")
            plt.axis('off')
            plt.title("non blind deconv x")
            plt.subplot(144)
            plt.imshow(self.truex, cmap = "gray")
            plt.axis('off')
            plt.title("true x")
            plt.show()
        
    def update_x(self, iteration):
        
        paddedOnes = np.lib.pad(np.ones((self.N1, self.N2)), ((self.dN1, self.dN1), (self.dN2, self.dN2)), 'constant', constant_values=(0))
        convolved = Convolution((self.N1e, self.N2e), self.k**2).convolve(paddedOnes)
        #plt.imshow(convolved)
        #plt.axis("off")
        #plt.show()
        self.da1 = 1. / (convolved * self.signoise**2)
        print("self.signoise**2", self.signoise**2)
        
        if(self.derivativeSpace):
            for i in range(self.nfilters):
                if(PARAMS["freeEnergy"]["use_prev_x"]):
                    self.filtx[i], self.C[i] = self.update_specx(self.filtx[i], self.C[i], iteration)
                else:
                    self.filtx[i], self.C[i] = self.update_specx(self.filty[i], self.C[i], iteration)
            self.x = Convolution(self.y.shape, self.k).deconvolve(self.y)
        else:
            if(PARAMS["freeEnergy"]["use_prev_x"]):
                self.x, self.C = self.update_specx(self.x, self.C, iteration)
            else:
                self.x, self.C = self.update_specx(self.y, self.C, iteration)
                
        if(PARAMS["verbose"]):
            print("Updated x ...")   
            self.print_x()
          
    def print_x(self):
        if(self.derivativeSpace):
            plt.figure(figsize = (15, 3 * self.nfilters))
            for i in range(self.nfilters):
                print("np.min(self.filtx[i]), np.max(self.filtx[i])", np.min(self.filtx[i]), np.max(self.filtx[i]))
                plt.subplot(100 * self.nfilters + 31 + 3 * i)
                plt.imshow(self.filtx[i], cmap = "gray")
                if(self.checkx):
                    plt.title("x, error " + str(format(np.sum((self.filtx[i] - self.truefiltx[i])**2)**0.5, '.3f')))
                else:
                    plt.title("x")
                plt.subplot(100 * self.nfilters + 32 + 3 * i)
                plt.hist(self.filtx[i].flatten(), bins = 40)
                plt.title("self.filtx[i] " +
                          str(format(np.min(self.filtx[i]), '.3f')) + " " +
                          str(format(np.max(self.filtx[i]), '.3f')))
                plt.subplot(100 * self.nfilters + 33 + 3 * i)
                plt.hist(self.C[i], bins = 40)
                plt.title("self.C[i] " +
                          str(format(np.min(self.C[i]), '.4f')) + " " +
                          str(format(np.max(self.C[i]), '.4f')))
            plt.show()
        else:
            print("np.min(self.x), np.max(self.x)", np.min(self.x), np.max(self.x))
            plt.figure(figsize = (15, 3))
            plt.subplot(131)
            plt.imshow(self.x, cmap = "gray")
            if(self.checkx):
                plt.title("x, error " + str(format(np.sum((self.x - self.truex)**2)**0.5, '.3f')))
            else:
                plt.title("x")
            plt.subplot(132)
            plt.hist(self.x.flatten(), bins = 40)
            plt.title("self.x")
            plt.subplot(133)
            plt.hist(self.C, bins = 40)
            plt.title("self.C")
            plt.show()
                
    def update_specx(self, x, c, iteration):
            
        x = np.lib.pad(x, ((self.dN1, self.dN1), (self.dN2, self.dN2)), 'constant', constant_values=(0))
        
        if(iteration == 0):
            w = np.sum(self.pi / (self.sigma ** 2)) * np.ones((self.N1e, self.N2e))
            
        else:
            
            E = x.flatten()**2 + c
            
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
        
        print("np.min(da1), np.max(da1)  ", np.min(self.da1), np.max(self.da1))
        print("np.min(w),   np.max(w)    ", np.min(w), np.max(w))
        xcov = 1. / (self.da1 + w)
            
        return x[self.dN1:-self.dN1, self.dN2:-self.dN2], xcov.flatten()
    
    def update_k(self, iteration):
        
        if(self.derivativeSpace):
            Ak = np.zeros((self.M**2, self.M**2))
            bk = np.zeros(self.M**2)
            for i in range(self.nfilters):
                tmpAk, tmpbk = corr.getAkbk(self.filtx[i],
                                      self.filty[i],
                                      self.C[i].reshape(self.N1e, self.N2e)[self.dN1:-self.dN1, self.dN2:-self.dN2],
                                      self.k.shape)
                Ak += tmpAk
                bk += tmpbk
            
        else:
            Ak, bk = corr.getAkbk(self.x,
                                  self.y,
                                  self.C.reshape(self.N1e, self.N2e)[self.dN1:-self.dN1, self.dN2:-self.dN2],
                                  self.k.shape)
            
        Ak = .5 * (Ak + Ak.transpose())
        
        plt.figure(figsize = (6, 3))
        plt.subplot(121)
        plt.imshow(Ak, cmap = "gray")
        plt.axis("off")
        plt.title("Ak")
        plt.subplot(122)
        plt.imshow(bk.reshape((self.M, self.M)), cmap = "gray")
        plt.axis("off")
        plt.title("bk")
        plt.show()
        
        Mones = np.ones(self.M**2)
        G = matrix(- np.diag(Mones))
        h = matrix(np.zeros(self.M**2))
        A = matrix(Mones, (1, self.M**2))
        b = matrix(1.0)
        
        #primalstart = np.zeros((self.M, self.M))
        #primalstart[int(self.M / 2.), int(self.M / 2.)] = 1.
        #self.k = np.array(solvers.qp(matrix(Ak), matrix(-bk), G, h, A, b, primalstart = matrix(primalstart.flatten()))["x"]).reshape((self.M, self.M))
        self.k = np.array(solvers.qp(matrix(Ak), matrix(-bk), G, h, A, b)["x"]).reshape((self.M, self.M))
        self.k = self.k
        """
        self.k = AxequalsbSolver({"A": Ak, "b": bk}).solve(np.zeros(self.k.shape)).reshape((self.M, self.M)).copy()
        self.k = np.abs(self.k)
        self.k /= np.sum(np.abs(self.k))
        for i in range(3):
            print("SOLVEk")
            w = (np.maximum(np.abs(self.k), 0.0001) ** (.5-2)).flatten()
            self.k = AxequalsbSolver({"A": Ak + 0.01 * np.diag(w), "b": bk}).solve(np.zeros(self.k.shape)).reshape((self.M, self.M)).copy()
            self.k /= np.sum(np.abs(self.k))
        """
        
        if(PARAMS["verbose"]):
            print("Updated k ...")
            self.print_k()
        
    def print_k(self):
        if(self.checkk):
            self.k_err += [np.sum((self.k - self.truek)**2)**0.5]
            plt.figure(figsize = (12, 3))
            plt.subplot(131)
            plt.imshow(self.k, cmap = "gray")
            plt.axis("off")
            plt.title("{:.2f} - {:.2f}".format(np.min(self.k), np.max(self.k)))
            plt.subplot(132)
            plt.imshow(self.k ** .5, cmap = "gray")
            plt.axis("off")
            plt.subplot(133)
            plt.plot(self.k_err)
            plt.show()
            print("k error         ", self.k_err[-1])
        else:
            plt.figure(figsize = (6, 3))
            plt.subplot(121)
            plt.imshow(self.k, cmap = "gray")
            plt.axis("off")
            plt.title("{:.2f} - {:.2f}".format(np.min(self.k), np.max(self.k)))
            plt.subplot(122)
            plt.imshow(self.k ** .5, cmap = "gray")
            plt.axis("off")
            plt.show()
            
        