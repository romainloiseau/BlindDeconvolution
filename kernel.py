# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:24:47 2018

@author: romai
"""

import numpy as np
from scipy import ndimage

class Kernel:
    
    def __init__(self, kernel):
        self.kernel = np.array(kernel)
        
    def getShape(self):
        return self.kernel.shape
    
    def getKernel(self):
        return self.kernel
        
    def convolve(self, matrix, mode = "constant", cval = 0.0):
        return ndimage.convolve(matrix, self.kernel, mode = mode, cval = cval)
    
    def convolve2(self, matrix):
        N0, N1 = matrix.shape
        T = self.getToepliz(N0, N1)
        return np.dot(T, matrix.flatten()).reshape(N0 - self.getShape()[0] + 1, N1 - self.getShape()[1] + 1)
    
    def getToepliz(self, N0, N1):
        N = N0*N1
        k0, k1 = self.kernel.shape
        T = np.zeros(((N0 - k0 + 1)* (N1 - k1 + 1), N))
        
        for i in range(0, N0 - k0 + 1):
            for j in range(0, N1 - k1 + 1):
                row = np.zeros(N)
                for k0i in range(k0):
                    for k1j in range(k1):
                        ind = N1 * (i + k0i) + (j + k1j)
                        
                        if(ind >= 0 and ind < N):
                            row[ind] = self.kernel[k0i, k1j]
                T[(N1 - k1 + 1) * i + j, :] = row
                
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
    print("Toepliz = ")
    print(T)
    print("Result = ")
    print(k.convolve2(matrix))
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
    print("Toepliz = ")
    print(T)
    print("Result = ")
    print(k.convolve2(matrix))
