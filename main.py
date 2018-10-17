# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:01:57 2018

@author: romai
"""

import yaml
import numpy as np
import opencv_utils as mycv2
from freeEnergy import FreeEnergy
from kernel import Kernel

PARAMS = yaml.load(open("params.yaml"))

image = mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"]))

fe = FreeEnergy(image)
#fe.renderImage()
fe.computeDerivatives()
#fe.renderDerivatives()
fe.iterate()



#k = Kernel([[1, 2], [3, 4]])
#N0, N1 = 3, 3
#k.getToepliz(N0, N1)