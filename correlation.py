# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:06:36 2018

@author: romai
"""
import numpy as np

def getAkbk(y, x, c, kshape):
    M1, M2 = kshape
    M = M1 * M2
    
    Ak = np.zeros((M, M))
    bk = np.zeros(M)
    
    for i1 in range(M):
        i1x, i1y = i1 % M1, i1 // M1
        for i2 in range(M):
            i2x, i2y = i2 % M2, i2 // M2
            aki1i2 = 0
            for px in range(x.shape[0]):
                for py in range(x.shape[1]):
                    try:aki1i2 += y[px + i1x, py + i1y] * y[px + i2x, py + i2y]
                    except:pass
                    if(i1x == i2x and i1y == i2y):
                        try:aki1i2 += c[px + i1x, py + i1y]
                        except:pass
            Ak[i1, i2] += aki1i2.copy()
                
    for i1 in range(M):
        i1x, i1y = i1 % M1, i1 // M1
        bki1 = 0
        for px in range(x.shape[0]):
            for py in range(x.shape[1]):
                try:bki1 += y[px + i1x, py + i1y] * x[px, py]
                except:pass
        bk[i1] += bki1.copy()
                
                
    return Ak, bk