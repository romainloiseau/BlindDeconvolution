# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:18:47 2018

@author: romai
"""

import numpy as np
from kernel import Kernel
import opencv_utils as mycv2
import yaml

PARAMS = yaml.load(open("params.yaml"))

class FreeEnergy:
    
    def __init__(self, image):
        self.image = image
        
    def computeDerivatives(self):
        self.fh = Kernel([[-1, 1]]).convolve(self.image)
        self.fv = Kernel([[-1], [1]]).convolve(self.image)
        
    def iterate(self):
        for i in range(PARAMS["freeEnergy"]["Niter"]):
            self.computeIteration()
            
    def computeIteration(self):
        self.updateQhigamma()
        self.updateMu()
        self.updateC()
        self.updateK()
    
    def updateQhigamma(self):
        if(PARAMS["verbose"]):
            print("Updating Qhigamma ...")
        
    def updateMu(self):
        #Ax = (1 / eta**2) * Tk.transpose() * Tk + np.sum([Tfgamma.transpose() * Wgamma * Tfgamma for gamma in Gamma], axis = 0)    (24)
        #bx = (1 / eta**2) * Tk.transpose() * y    (25)
        #solve Ax * mu = bx    (40)
        if(PARAMS["verbose"]):
            print("Updating Mu ...")
            
    def updateC(self):
        #C = Ax**(-1)    Impractical for large matrices
        #C[i, i] = 1 / Ax[i, i]    Best choice to accelerate computation
        if(PARAMS["verbose"]):
            print("Updating C ...")
            
    def updateK(self):
        #Ak(i1, i2) = np.sum([mu(i + i1) * mu(i + i2) + C(i + i1, i + i2) for i in pixels])    (26)
        #bk(i1) = np.sum([mu(i + i1) * y(i) for i in pixels])    (27)
        #solve minOverK(.5 * k.transpose() * Ak * k + bk.transpose() * k) s.t. k >= 0    (28)
        if(PARAMS["verbose"]):
            print("Updating k ...")
        
    def renderImage(self):
        mycv2.show(self.image)
        
    def renderDerivatives(self):
        torender = np.concatenate((self.fh, self.fv), axis=1)
        mycv2.show(torender, maxwidth = 2*PARAMS["render"]["maxwidth"])