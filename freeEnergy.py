# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:18:47 2018

@author: romai
"""

import numpy as np
from kernel import Kernel
from qhigamma import Qhigamma
import matplotlib.pyplot as plt
from convolution import Convolution
from axequalsbSolver import AxequalsbSolver
import opencv_utils as mycv2
from scipy.fftpack import fft2, ifft2
import yaml

PARAMS = yaml.load(open("params.yaml"))

class FreeEnergy:
    
    def __init__(self, image, blurringkernel):
        self.blurred = image.copy()
        self.image = image.copy()
        self.blurringkernel = blurringkernel
        
    def initialize(self):
        #Our prior is a mixture of J gaussian (MOG) with weights pi, mean 0, and standard deviation sigma
        self.J = PARAMS["freeEnergy"]["J"]
        self.pi = np.random.random(size = self.J)    #Weigths
        self.pi /= np.sum(self.pi)
        self.sigma = 2 * self.image.shape[0] * np.random.random(size = self.J)
        
        self.deltak = PARAMS["filters"]["delta"]
        
        self.M = PARAMS["freeEnergy"]["M"]
        self.k = 10 + np.random.random((self.M, self.M))
        self.k /= np.sum(self.k)
        
        self.factorkernel = (1 / PARAMS["freeEnergy"]["eta"]**2)
        
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
            plt.hist(self.q.getQ().flatten())
            plt.title("Qhigamma")
            plt.show()
            print("Updated Qhigamma ...")
        
    def updateMu(self):
        #Ax = (1 / eta**2) * Tk.transpose() * Tk + np.sum([Tfgamma.transpose() * Wgamma * Tfgamma for gamma in Gamma], axis = 0)    (24)
        #bx = (1 / eta**2) * Tk.transpose() * y    (25)
        #solve Ax * mu = bx    (40)
        
        #Tk = self.k.getToepliz(self.N1, self.N2)
        
        Wgammadiag = self.q.getQ().copy()
        for i in range(len(self.sigma)):
            Wgammadiag[:, i] /= self.sigma[i]**2
        
        self.Wgamma = Wgammadiag.reshape((self.image.shape[0], self.image.shape[1], len(self.sigma))).transpose(2, 0, 1).sum(axis = 0)
        
        
        #self.Wgammadiag = Wgammadiag.sum(axis = -1)
        
        #self.Ax = (1 / PARAMS["freeEnergy"]["eta"]**2) * Tk.transpose().dot(Tk) + Tfgamma.transpose() * np.diag(Wgammadiag) * Tfgamma
        #self.bx = (1 / PARAMS["freeEnergy"]["eta"]**2) * Tk.transpose() * self.blurred
        
        #print(Tk)
        #print(self.bx)
        
        
        self.mu = AxequalsbSolver({
                "image": self.blurred,
                "kernel": self.k,
                "gammakernel": self.deltak,
                "gammasigma": self.sigma,
                "gammapi": self.pi,
                "Wgamma": self.Wgamma,
                "factorkernel" : self.factorkernel
                }, option = "updateMu").solve()
        
        self.mu = (self.mu - np.min(self.mu)) / (np.max(self.mu) - np.min(self.mu))
        """
        kconv = Convolution(self.image.shape, self.k)
        deltakconv = Convolution(self.image.shape, self.deltak)
        
        
        withKernel = self.factorkernel * (np.conjugate(kconv.kernelFT)).transpose() * kconv.kernelFT
        withGamma = (np.conjugate(deltakconv.kernelFT)).transpose() * self.Wgamma * deltakconv.kernelFT
        self.AxFT = withKernel + withGamma
        print(self.AxFT.shape)
        self.bxFT = self.factorkernel * (np.conjugate(kconv.kernelFT)).transpose() * fft2(self.blurred)
        print(self.bxFT.shape)
        self.mu = AxequalsbSolver({"A": self.AxFT, "b": self.bxFT})
        """
    
        if(PARAMS["verbose"]):
            plt.subplot(121)
            plt.hist(self.mu)
            plt.title("mu")
            plt.subplot(122)
            plt.imshow(self.mu.reshape(self.image.shape), cmap = "gray")
            plt.show()
            print("Updated Mu ...")

    def updateC(self):
        #C = Ax**(-1)    Impractical for large matrices
        #C[i, i] = 1 / Ax[i, i]    Best choice to accelerate computation
        
        
        convolution = Convolution(self.image.shape, self.k)
        gammaConvolution = Convolution(self.image.shape, self.deltak)
        
        for i in range(len(self.Cdiag)):
            origin = np.zeros(len(self.Cdiag))
            origin[i] = 1
            reshapedorigin = origin.reshape(self.image.shape)
            
            withKernel = self.factorkernel * convolution.convolve(convolution.convolve(reshapedorigin), mode = "adjoint")
            gammaconvolvedreshapedorigin = gammaConvolution.convolve(reshapedorigin)
            withGamma = gammaConvolution.convolve(self.Wgamma * gammaconvolvedreshapedorigin, mode = "adjoint")
            result = (withKernel + withGamma).flatten()
            self.Cdiag[i] = 1 / result[i]
            
        """
        kconv = Convolution(self.image.shape, self.k)
        deltakconv = Convolution(self.image.shape, self.deltak)
        
        withKernel = self.factorkernel * (np.conjugate(kconv.kernelFT)).transpose() * kconv.kernelFT
        withGamma = (np.conjugate(deltakconv.kernelFT)).transpose() * self.Wgamma * deltakconv.kernelFT
        self.Cdiag = 1 / np.real(ifft2(withKernel + withGamma).flatten())
        """
        
        if(PARAMS["verbose"]):
            plt.hist(self.Cdiag)
            plt.title("C")
            plt.show()
            print("Updated C ...")
            
    def updateK(self):
        #Ak(i1, i2) = np.sum([mu(i + i1) * mu(i + i2) + C(i + i1, i + i2) for i in pixels])    (26)
        #bk(i1) = np.sum([mu(i + i1) * y(i) for i in pixels])    (27)
        #solve minOverK(.5 * k.transpose() * Ak * k + bk.transpose() * k) s.t. k >= 0    (28)
        
        Ak, bk = Kernel(self.k).getAkbk(self.image, self.mu.reshape(self.image.shape), np.diag(self.Cdiag))
        #self.k = Kernel(AxequalsbSolver({"A": Ak, "b": bk}).solve().reshape((self.M, self.M)))
        self.k = AxequalsbSolver({"A": Ak, "b": bk}).solve().reshape((self.M, self.M))
        self.k /= np.sum(np.abs(self.k))
        if(PARAMS["verbose"]):
            print("Updated k ...")
            print(self.k)
            print("ERROR", np.sum((self.k - self.blurringkernel)**2)**0.5)
        
    def renderImage(self):
        plt.imshow(self.image, cmap = "gray")
        plt.title("Algo image")
        plt.show()