# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:18:47 2018

@author: romai
"""

import numpy as np
from kernel import Kernel
from qhigamma import Qhigamma
import matplotlib.pyplot as plt
from axequalsbSolver import AxequalsbSolver
import opencv_utils as mycv2
import yaml

PARAMS = yaml.load(open("params.yaml"))

class FreeEnergy:
    
    def __init__(self, image):
        self.blurred = image.copy().flatten()
        self.image = image.copy()
        
    def initialize(self):
        #Our prior is a mixture of J gaussian (MOG) with weights pi, mean 0, and standard deviation sigma
        self.J = PARAMS["freeEnergy"]["J"]
        self.pi = np.random.random(size = self.J)    #Weigths
        self.pi /= np.sum(self.pi)
        self.sigma = np.random.random(size = self.J)
        
        self.M = PARAMS["freeEnergy"]["M"]
        self.k = Kernel(np.zeros((self.M, self.M)))
        
        self.N1, self.N2 = self.image.shape[0], self.image.shape[1]
        self.N = self.N1 * self.N2
        
        self.q = Qhigamma(self.N, self.J)
        
        self.mu = self.image.copy().flatten()
        self.Cdiag = np.random.random(self.N)
        
    def iterate(self):
        for i in range(PARAMS["freeEnergy"]["Niter"]):
            self.computeIteration()
            
    def computeIteration(self):
        self.updateQhigamma()
        self.updateMu()
        self.updateC()
        self.updateK()
    
    def updateQhigamma(self):
        self.q.update(self.pi, self.sigma, self.mu, self.Cdiag)
        if(PARAMS["verbose"]):
            print("Updated Qhigamma ...")
        
    def updateMu(self):
        #Ax = (1 / eta**2) * Tk.transpose() * Tk + np.sum([Tfgamma.transpose() * Wgamma * Tfgamma for gamma in Gamma], axis = 0)    (24)
        #bx = (1 / eta**2) * Tk.transpose() * y    (25)
        #solve Ax * mu = bx    (40)
        Tk = self.k.getToepliz(self.N1, self.N2)
        
        Wgammadiag = self.q.getQ().copy()
        for i in range(len(self.sigma)):
            Wgammadiag[:, i] /= self.sigma[i]
        Wgammadiag = Wgammadiag.sum(axis = -1)
        
        self.Ax = (1 / PARAMS["freeEnergy"]["eta"]**2) * Tk.transpose().dot(Tk) #+ Tfgamma.transpose() * np.diag(Wgammadiag) * Tfgamma
        self.bx = (1 / PARAMS["freeEnergy"]["eta"]**2) * Tk.transpose() * self.blurred
        
        print(Tk)
        print(self.bx)
        
        self.mu = AxequalsbSolver(self.Ax, self.bx).solve()
        print(self.mu)
        
        if(PARAMS["verbose"]):
            print("Updated Mu ...")
            
    def updateC(self):
        #C = Ax**(-1)    Impractical for large matrices
        #C[i, i] = 1 / Ax[i, i]    Best choice to accelerate computation
        self.Cdiag = 1 / (self.Ax.diagonal() + 10**(-6))
        if(PARAMS["verbose"]):
            print("Updated C ...")
            
    def updateK(self):
        #Ak(i1, i2) = np.sum([mu(i + i1) * mu(i + i2) + C(i + i1, i + i2) for i in pixels])    (26)
        #bk(i1) = np.sum([mu(i + i1) * y(i) for i in pixels])    (27)
        #solve minOverK(.5 * k.transpose() * Ak * k + bk.transpose() * k) s.t. k >= 0    (28)
        
        Ak, bk = self.k.getAkbk(self.image, self.mu.reshape(self.image.shape), np.diag(self.Cdiag))
        self.k = Kernel(AxequalsbSolver(Ak, bk).solve().reshape((self.M, self.M)))
        
        if(PARAMS["verbose"]):
            print("Updated k ...")
        
    def renderImage(self):
        plt.imshow(self.image, cmap = "gray")
        plt.show()