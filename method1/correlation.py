# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:06:36 2018

@author: romai
"""
import numpy as np

def getAkbk(x, y, c, kshape):
    M1, M2 = kshape
    M = M1 * M2
    
    PX, PY = x.shape
    Ak = np.zeros((M, M))
    bk = np.zeros(M)
            
    for i1 in range(M):
        i1x, i1y = i1 // M1, i1 % M1
        i1x -= int(M1 / 2.)
        i1y -= int(M1 / 2.)
        for i2 in range(i1, M):
            i2x, i2y = i2 // M2, i2 % M2
            i2x -= int(M1 / 2.)
            i2y -= int(M1 / 2.)
            aki1i2 = 0
            for px in range(PX):
                for py in range(PY):
                    if(px + i1x < PX and  py + i1y < PY and px + i1x >= 0 and  py + i1y >= 0):
                        if(px + i2x < PX and py + i2y < PY and px + i2x >= 0 and  py + i2y >= 0):
                            aki1i2 += x[px + i1x, py + i1y] * x[px + i2x, py + i2y]
                        if(i1x == i2x and i1y == i2y):
                            aki1i2 += c[px + i1x, py + i1y]
            Ak[i1, i2] += aki1i2.copy()
            Ak[i2, i1] += aki1i2.copy()
            
    for i1 in range(M):
        i1x, i1y = i1 // M1, i1 % M1
        i1x -= int(M1 / 2.)
        i1y -= int(M1 / 2.)
        bki1 = 0
        for px in range(x.shape[0]):
            for py in range(x.shape[1]):
                if(px + i1x >= 0 and px + i1x < PX and  py + i1y >= 0 and py + i1y < PY):
                    bki1 += x[px + i1x, py + i1y] * y[px, py]
        bk[i1] += bki1.copy()
                
                
    return Ak, bk