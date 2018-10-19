# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:24:47 2018

@author: romai
"""

import numpy as np
from scipy import ndimage, sparse
import time

class Kernel:
    
    def __init__(self, kernel):
        self.kernel = np.array(kernel)
        
    def getShape(self):
        return self.kernel.shape
    
    def getKernel(self):
        return self.kernel
        
    def convolveScipy(self, matrix, mode = "constant", cval = 0.0):
        return ndimage.convolve(matrix, self.kernel, mode = mode, cval = cval)
    
    def convolveToeplix(self, matrix):
        N0, N1 = matrix.shape
        #return np.dot(T, matrix.flatten()).reshape(N0 - self.getShape()[0] + 1, N1 - self.getShape()[1] + 1)
        return self.getToepliz(N0, N1).dot(matrix.flatten()).reshape(N0 - self.getShape()[0] + 1, N1 - self.getShape()[1] + 1)
    
    def getToepliz(self, N0, N1):
        N = N0*N1
        k0, k1 = self.getShape()
        k = self.kernel
        T = sparse.lil_matrix(((N0 - k0 + 1)* (N1 - k1 + 1), N))
        t = time.time()
        
        for k0i in range(k0):
            for k1j in range(k1):
                kk = k[k0i, k1j]
                for i in range(0, N0 - k0 + 1):
                    for j in range(0, N1 - k1 + 1):
                        T[(N1 - k1 + 1) * i + j, N1 * (i + k0i) + (j + k1j)] = kk
                        
        print(time.time() - t)
        return T
    
def runTests():
    k = Kernel([[1, .1]])
    print("Kernel = ")
    print(k.getKernel())
    N0, N1 = 3, 4
    matrix = np.arange(N0*N1).reshape(N0, N1)
    print("Matrix = ")
    print(matrix)
    T = k.getToepliz(N0, N1)
    print("Toepliz = ", T.shape)
    print(T)
    print("Result = ")
    print(k.convolveToeplix(matrix))
    del k
    print()
    k = Kernel([[10, 1], [.1, .01]])
    print("Kernel = ")
    print(k.kernel)
    N0, N1 = 3, 3
    matrix = np.arange(N0*N1).reshape(N0, N1)
    print("Matrix = ")
    print(matrix)
    T = k.getToepliz(N0, N1)
    print("Toepliz = ", T.shape)
    print(T)
    print("Result = ")
    print(k.convolveToeplix(matrix))
    
    print()    
    k = Kernel([[10, 1], [.1, .01]])
    N0, N1 = 1000, 300
    matrix = np.arange(N0*N1).reshape(N0, N1)
    
    ntimes = 1
    t = time.time()
    [k.convolveToeplix(matrix) for i in range(ntimes)]
    print("Toepliz :", (time.time() - t) / ntimes, "secs")
    
    t = time.time()
    [k.convolveScipy(matrix) for i in range(ntimes)]
    print("Scipy :", (time.time() - t) / ntimes, "secs")