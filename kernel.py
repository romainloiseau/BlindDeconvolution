# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:24:47 2018

@author: romai
"""

import numpy as np
from scipy import ndimage, sparse, signal
import time

class Kernel:
    
    def __init__(self, kernel):
        self.kernel = np.array(kernel)
        
    def getShape(self):
        return self.kernel.shape
    
    def getKernel(self):
        return self.kernel
        
    def convolveScipy(self, matrix, mode = "same"):
        return signal.convolve2d(matrix, self.kernel, mode = mode, boundary = "symm")
    
    def convolveToepliz(self, matrix):
        N0, N1 = matrix.shape
        #return np.dot(T, matrix.flatten()).reshape(N0 - self.getShape()[0] + 1, N1 - self.getShape()[1] + 1)
        return self.getToepliz(N0, N1).dot(matrix.flatten()).reshape(N0 - self.getShape()[0] + 1, N1 - self.getShape()[1] + 1)
    
    def getToepliz(self, N0, N1):
        N = N0*N1
        k0, k1 = self.getShape()
        k = self.kernel
        
        if(k0 == k1 and k0%2==1):
             T = sparse.lil_matrix((N, N))
             for i in range(N0):
                 for j in range(N1):
                     for k0i in range(int(-(k0 - 1)/2), int((k0 - 1)/2) + 1):
                         for k1j in range(int(-(k1 - 1)/2), int((k1 - 1)/2) + 1):
                             if(N1 * (i + k0i) + (j + k1j) < N):
                                 T[N1 * i + j, N1 * (i + k0i) + (j + k1j)] = k[k0i, k1j]
                                 
        else:
            T = sparse.lil_matrix(((N0 - k0 + 1)* (N1 - k1 + 1), N))
            
            for k0i in range(k0):
                for k1j in range(k1):
                    kk = k[k0i, k1j]
                    for i in range(0, N0 - k0 + 1):
                        for j in range(0, N1 - k1 + 1):
                            T[(N1 - k1 + 1) * i + j, N1 * (i + k0i) + (j + k1j)] = kk
                            
        return T
    
    def getAkbk(self, matrix, mu, C):
        M1, M2 = self.kernel.shape
        M = M1 * M2
        
        Ak = np.zeros((M, M))
        bk = np.zeros(M)
        
        for ix in range(matrix.shape[0]):
            for iy in range(matrix.shape[0]):
                for i1 in range(M):
                    i1x, i1y = i1 % M1, i1 // M1
                    if(ix + i1x < matrix.shape[0] and iy + i1y < matrix.shape[1]):
                        bk[i1] += mu[ix + i1x, iy + i1y] * matrix[ix, iy]
                        for i2 in range(i1, M):
                            i2x, i2y = i2 % M1, i2 // M1
                            if(ix + i2x < matrix.shape[0] and iy + i2y < matrix.shape[1]):
                                tempaki1i2 = mu[ix + i1x, iy + i1y] * matrix[ix + i2x, iy + i2y]
                                if(ix + i1x + M1 * (iy + i1y) == ix + i2x + M1 * (iy + i2y)):
                                    tempaki1i2 += C[ix + i1x + M1 * (iy + i1y)]
                                Ak[i1, i2] += tempaki1i2.copy()
                                Ak[i2, i1] += tempaki1i2.copy()
        return Ak, bk
    
def testsAkbk():
    k = Kernel([[10, 1], [.1, .01]])
    print("Kernel = ")
    print(k.kernel)
    N0, N1 = 4, 4
    matrix = np.arange(N0*N1).reshape(N0, N1)
    print("Matrix = ")
    print(matrix)
    
    C = np.diag(np.ones(4))
    print(C)
    
    k.getAkbk(matrix, matrix + np.random.random(matrix.shape), C)
    
    
def runTests():
    print("Kernel tests")
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
    print(k.convolveToepliz(matrix))
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
    print(k.convolveToepliz(matrix))
    
    print()    
    k = Kernel([[10, 1], [.1, .01]])
    N0, N1 = 1000, 300
    matrix = np.arange(N0*N1).reshape(N0, N1)
    
    ntimes = 1
    t = time.time()
    [k.convolveToepliz(matrix) for i in range(ntimes)]
    print("Toepliz :", (time.time() - t) / ntimes, "secs")
    
    t = time.time()
    [k.convolveScipy(matrix) for i in range(ntimes)]
    print("Scipy :", (time.time() - t) / ntimes, "secs")