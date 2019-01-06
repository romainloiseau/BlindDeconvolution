# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:06:36 2018

@author: romai
"""
import matplotlib.pyplot as plt
import numpy as np


#Computes correlation matrices
def getAkbk(x, y, c, kshape):
    M1, M2 = kshape
    M = M1 * M2
    
    PX, PY = x.shape
    Ak = np.zeros((M, M))
    Ak_c = np.zeros((M, M))
    bk = np.zeros(M)
    
    #Auto correlation
    for i1 in range(M):
        i1x, i1y = i1 // M1, i1 % M1
        i1x -= int(M1 / 2.)
        i1y -= int(M1 / 2.)
        for i2 in range(i1, M):
            i2x, i2y = i2 // M2, i2 % M2
            i2x -= int(M1 / 2.)
            i2y -= int(M1 / 2.)
            aki1i2 = 0
            ak_c_i1i2 = 0
            for px in range(-M, PX + M):
                for py in range(-M, PY + M):
                    if(px + i1x < PX and  py + i1y < PY and px + i1x >= 0 and  py + i1y >= 0):
                        if(px + i2x < PX and py + i2y < PY and px + i2x >= 0 and  py + i2y >= 0):
                            aki1i2 += x[px + i1x, py + i1y] * x[px + i2x, py + i2y]
                    if(i1 == i2):
                        if(int(M1 / 2.) + px + i1x < PX + 2 * int(M1 / 2.) and  int(M2 / 2.) + py + i1y < PY + 2 * int(M2 / 2.)
                           and int(M1 / 2.) + px + i1x >= 0 and  int(M2 / 2.) + py + i1y >= 0):
                            ak_c_i1i2 += c[int(M1 / 2.) + px + i1x, int(M2 / 2.) + py + i1y]
            Ak[i1, i2] += aki1i2.copy()
            Ak_c[i1, i2] += ak_c_i1i2
            if(i1 != i2):
                Ak[i2, i1] += aki1i2.copy()
    
    plt.figure(figsize = (15, 5))
    plt.subplot(131)
    plt.imshow(Ak, cmap = "gray")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(Ak_c, cmap = "gray")
    plt.axis("off")
    plt.subplot(133)
    Ak += Ak_c
    plt.imshow(Ak, cmap = "gray")
    plt.axis("off")
    plt.show()
    
    for i1 in range(M):
        i1x, i1y = i1 // M1, i1 % M1
        i1x -= int(M1 / 2.)
        i1y -= int(M1 / 2.)
        bki1 = 0
        for px in range(PX):
            for py in range(PY):
                if(px + i1x >= 0 and px + i1x < PX and  py + i1y >= 0 and py + i1y < PY):
                    bki1 += x[px + i1x, py + i1y] * y[px, py]
        bk[i1] += bki1.copy()
                
                
    return Ak, bk