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
        
        self.y = y.copy()
        self.x = y.copy()
        
        self.checkx = not truex is None
        if(self.checkx):
            if(y.shape != truex.shape):
                print("WARNING, truex shape different from initial image y shape, setting self.checkx to False")
                self.checkx = False
            else:
                self.truex = truex
                self.x_psnr = []
            
        self.checkk = not truek is None
        if(self.checkk):
            self.k_err = []
            self.truek = truek
            self.nonblinddeconvx = Convolution(self.y.shape, self.truek).deconvolve(self.y)
        
        self.derivativeSpace = derivativeSpace
        if(self.derivativeSpace):self.computeInitialDerivatives()
        
    def computeInitialDerivatives(self):
        self.filters = [PARAMS["derivativefilters"][k] for k in PARAMS["derivativefilters"]]
        self.nfilters = len(self.filters)
        self.derivative_convs = [Convolution(self.y.shape, f) for f in self.filters]
        self.filty = [conv.convolve(self.y) for conv in self.derivative_convs]
        self.filtx = self.filty.copy()
        
        plt.hist(self.filtx[0].flatten())
        plt.show()
        
        if(self.checkx):self.truefiltx = [conv.convolve(self.truex) for conv in self.derivative_convs]
    
    def initialize(self):
        #Initialize kernel
        self.M = 2 * int(PARAMS["freeEnergy"]["M"]/2) + 1
        if(self.checkk and self.M != self.truek.shape[0]):
            print("WARNING, self.truek shape different from researched kernel shape self.M, setting self.checkk to False")
            self.checkk = False
        
        self.k = np.zeros((self.M, self.M))
        self.k[int(self.M / 2.), int(self.M / 2.)] = 1
        self.k /= np.sum(self.k)
        
        #Initialize shapes
        self.N1, self.N2 = self.x.shape[0], self.x.shape[1]
        self.dN1, self.dN2 = int(self.M / 2.), int(self.M / 2.)
        self.N1e, self.N2e = self.N1 + 2 * self.dN1, self.N2 + 2 * self.dN2
        self.N, self.Ne = self.N1 * self.N2, self.N1e * self.N2e
        
        #Noise
        self.divnoise = 1.01
        self.signoise_v = PARAMS["freeEnergy"]["eta"] * self.divnoise ** np.arange(PARAMS["freeEnergy"]["Niter"] - 1, -1, -1)
        print(self.signoise_v)
        #Our prior is a mixture of J gaussian (MOG) with weights pi, mean 0, and standard deviation sigma
        self.pi = np.array(PARAMS["freeEnergy"]["pis"])
        self.pi /= self.pi.sum()
        self.ivars = np.array(PARAMS["freeEnergy"]["ivars"])
        self.J = len(self.ivars)
        
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
            self.x_psnr += [10 * np.log10(1. / np.mean((self.x - self.truex)**2))]
            plt.figure(figsize = (20, 10))
            plt.subplot(245)
            plt.imshow(self.y, cmap = "gray")
            plt.axis('off')
            plt.title("blurred x  -  pSNR = {:.2f}".format(10 * np.log10(1. / np.mean((self.y - self.truex)**2))), fontdict = {'fontsize': 19})
            plt.subplot(246)
            plt.imshow(self.x, cmap = "gray")
            plt.axis('off')
            plt.title("deconv x  -  pSNR = {:.2f}".format(self.x_psnr[-1]), fontdict = {'fontsize': 19})
            plt.subplot(247)
            plt.imshow(self.nonblinddeconvx, cmap = "gray")
            plt.axis('off')
            plt.title("nonblind x  -  pSNR = {:.2f}".format(10 * np.log10(1. / np.mean((self.nonblinddeconvx - self.truex)**2))), fontdict = {'fontsize': 19})
            plt.subplot(248)
            plt.imshow(self.truex, cmap = "gray")
            plt.axis('off')
            plt.title("true x", fontdict = {'fontsize': 19})
            plt.subplot(244)
            plt.title("x  -  pSNR", fontdict = {'fontsize': 19})
            plt.plot(self.x_psnr)
            plt.ylabel("pSNR", fontdict = {'fontsize': 16})
            plt.xlabel("iteration", fontdict = {'fontsize': 16})
            plt.subplot(241)
            plt.imshow(self.k, cmap = "gray")
            plt.axis("off")
            plt.title("kernel  -  pSNR = {:.2f}".format(self.k_err[-1]), fontdict = {'fontsize': 19})
            plt.subplot(242)
            plt.imshow(self.truek, cmap = "gray")
            plt.title("true kernel", fontdict = {'fontsize': 19})
            plt.axis("off")
            plt.subplot(243)
            plt.plot(self.k_err)
            plt.title("kernel  -  pSNR", fontdict = {'fontsize': 19})
            plt.ylabel("pSNR", fontdict = {'fontsize': 16})
            plt.xlabel("iterations", fontdict = {'fontsize': 16})
            plt.tight_layout()
            plt.show()
        
    def update_x(self, iteration):
        
        paddedOnes = np.lib.pad(np.ones((self.N1, self.N2)), ((self.dN1, self.dN1), (self.dN2, self.dN2)), 'constant', constant_values=(0))
        convk = Convolution((self.N1e, self.N2e), self.k)
        convolved = convk.convolve(convk.convolve(paddedOnes, mode = "adjoint"))
        #plt.imshow(convolved)
        #plt.axis("off")
        #plt.show()
        self.da1 = convolved / self.signoise**2
        
        if(self.derivativeSpace):
            for i in range(self.nfilters):
                if(PARAMS["freeEnergy"]["use_prev_x"]):
                    self.filtx[i], self.C[i] = self.update_specx(self.filtx[i], self.C[i], iteration)
                else:
                    self.filtx[i], self.C[i] = self.update_specx(self.filty[i], self.C[i], iteration)
            """
            self.x = AxequalsbSolver({
                    "image": self.y,
                    "kernel": self.k,
                    "w": np.zeros(self.y.shape),
                    "weightpen" : 0.}, option = "updatex").solve().reshape(self.y.shape)
            """
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
            #results = [conv.deconvolve(filtxi) for conv, filtxi in zip(self.derivative_convs, self.filtx)]
            plt.figure(figsize = (15, 3 * self.nfilters))
            for i in range(self.nfilters):
                #print("np.min(self.filtx[i]), np.max(self.filtx[i])", np.min(self.filtx[i]), np.max(self.filtx[i]))
                plt.subplot(100 * self.nfilters + 41 + 4 * i)
                plt.imshow(self.filtx[i], cmap = "gray")
                plt.axis("off")
                if(self.checkx):
                    plt.title("dx, error " + str(format(10 * np.log10(1. / np.mean((self.filtx[i] - self.truefiltx[i])**2)), '.3f')))
                else:
                    plt.title("dx")
                plt.subplot(100 * self.nfilters + 42 + 4 * i)
                plt.imshow(self.C[i].reshape((self.N1e, self.N2e)), cmap = "gray")
                """
                plt.imshow(results[i], cmap = "gray")=
                if(self.checkx):
                    plt.title("x, error " + str(format(10 * np.log10(1. / np.mean((results[i] - self.truex[i])**2)), '.3f')))
                else:
                    plt.title("x")
                """
                plt.axis("off")
                plt.subplot(100 * self.nfilters + 43 + 4 * i)
                plt.hist(self.filtx[i].flatten(), bins = 40)
                plt.title("self.filtx[i] " +
                          str(format(np.min(self.filtx[i]), '.3f')) + " " +
                          str(format(np.max(self.filtx[i]), '.3f')))
                plt.subplot(100 * self.nfilters + 44 + 4 * i)
                plt.hist(self.C[i], bins = 40)
                plt.title("self.C[i] " +
                          str(format(np.min(self.C[i]), '.4f')) + " " +
                          str(format(np.max(self.C[i]), '.4f')))
            plt.show()
        else:
            #print("np.min(self.x), np.max(self.x)", np.min(self.x), np.max(self.x))
            plt.figure(figsize = (15, 3))
            plt.subplot(131)
            plt.imshow(self.x, cmap = "gray")
            if(self.checkx):
                plt.title("x, error " + str(format(10 * np.log10(1. / np.mean((self.x - self.truex)**2)), '.3f')))
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
            w = np.sum(self.pi * self.ivars) * np.ones((self.N1e, self.N2e))
            
        else:
            
            E = x.flatten()**2 + c
            plt.figure(figsize = (15, 5))
            plt.subplot(141)
            plt.imshow(E.reshape((self.N1e, self.N2e))[self.dN1:-self.dN1, self.dN2:-self.dN2], cmap = "gray")
            plt.axis("off")
            plt.subplot(142)
            plt.imshow((E - c).reshape((self.N1e, self.N2e))[self.dN1:-self.dN1, self.dN2:-self.dN2], cmap = "gray")
            plt.axis("off")
            plt.subplot(143)
            plt.imshow(c.reshape((self.N1e, self.N2e))[self.dN1:-self.dN1, self.dN2:-self.dN2], cmap = "gray")
            plt.axis("off")
            
            logq = np.array([E for j in range(self.J)]).transpose()
            logq =  - 0.5 * logq * self.ivars + np.ones((self.Ne, self.J)) * (np.log(self.pi) + .5 * np.log(self.ivars))
            logq_max = np.max(logq, axis = -1)
            q = np.exp(logq.transpose() - logq_max)
            q_sum = np.sum(q, axis = 0)
            q /= q_sum
            w = np.sum(q.transpose() * self.ivars, axis = -1).reshape(self.N1e, self.N2e)
            """
            logq = np.zeros((self.J, self.Ne))
            
            for i in range(self.J):
                ivar_i = self.ivars[i]
                pi_i = self.pi[i]
                logq_i = - 0.5 * E * ivar_i + np.ones(self.Ne) * (np.log(pi_i) + .5 * np.log(ivar_i))
                logq[i] = logq_i.copy()
        
            
            q = np.exp(logq - np.max(logq, axis = 0))
            q /= np.sum(q, axis = 0)
            
            w = (q.transpose() * self.ivars).sum(axis = -1).reshape(self.N1e, self.N2e)
            """
            plt.subplot(144)
            plt.imshow(w[self.dN1:-self.dN1, self.dN2:-self.dN2], cmap = "gray")
            plt.axis("off")
            plt.show()
            
        x = AxequalsbSolver({
                "image": x[self.dN1:-self.dN1, self.dN2:-self.dN2],
                "kernel": self.k,
                "weightpen": self.signoise**2,
                "w": w[self.dN1:-self.dN1, self.dN2:-self.dN2]
                }, option = "updatex").solve()
    
        x = x.reshape(self.N1, self.N2)
        
        print("np.min(da1), np.max(da1)  ", np.min(self.da1[self.dN1:-self.dN1, self.dN2:-self.dN2]), np.max(self.da1[self.dN1:-self.dN1, self.dN2:-self.dN2]))
        print("np.min(w),   np.max(w)    ", np.min(w[self.dN1:-self.dN1, self.dN2:-self.dN2]), np.max(w[self.dN1:-self.dN1, self.dN2:-self.dN2]))
        xcov = 1. / (self.da1 + w)
            
        return x, xcov.flatten()
    
    def update_k(self, iteration):
        
        if(self.derivativeSpace):
            Ak = np.zeros((self.M**2, self.M**2))
            bk = np.zeros(self.M**2)
            for i in range(self.nfilters):
                tmpAk, tmpbk = corr.getAkbk(self.filtx[i],
                                      self.filty[i],
                                      self.C[i].reshape(self.N1e, self.N2e),
                                      self.k.shape)
                Ak += tmpAk
                bk += tmpbk
            
        else:
            Ak, bk = corr.getAkbk(self.x,
                                  self.y,
                                  self.C.reshape(self.N1e, self.N2e),
                                  self.k.shape)
            
        #Ak = .5 * (Ak + Ak.transpose())
        
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
        
        self.k = np.array(solvers.qp(matrix(Ak), matrix(-bk), G, h, A, b)["x"]).reshape((self.M, self.M))
        #self.k = np.array(solvers.qp(matrix(Ak), matrix(-bk), G, h)["x"]).reshape((self.M, self.M))
        
        """
        k = np.abs(self.k).flatten()
        z = np.zeros(len(k))
        sort = np.argsort(k)[-int(len(k) * .25):]
        z[sort] = k[sort]
        z /= np.sum(z)
        self.k = z.reshape((self.M, self.M)).copy()
        """
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
            self.k_err += [10 * np.log10(1. / np.mean((self.k - self.truek)**2))]
            plt.figure(figsize = (15, 4))
            plt.subplot(131)
            plt.imshow(self.k, cmap = "gray")
            plt.axis("off")
            plt.title("k  -  pSNR = {:.2f}".format(self.k_err[-1]))
            plt.subplot(132)
            plt.imshow(self.truek, cmap = "gray")
            plt.title("true k")
            plt.axis("off")
            plt.subplot(133)
            plt.plot(self.k_err)
            plt.ylabel("pSNR")
            plt.xlabel("iterations")
            plt.show()
        else:
            plt.figure(figsize = (6, 3))
            plt.subplot(121)
            plt.imshow(self.k, cmap = "gray")
            plt.axis("off")
            plt.title("{:.2f} - {:.2f}".format(np.min(self.k), np.max(self.k)))
            plt.subplot(122)
            plt.imshow(np.abs(self.k) ** .5, cmap = "gray")
            plt.axis("off")
            plt.show()
            
        