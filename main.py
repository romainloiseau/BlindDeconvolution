# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:01:57 2018

@author: romai
"""

import yaml
import opencv_utils as mycv2
from freeEnergy import FreeEnergy

PARAMS = yaml.load(open("params.yaml"))

image = mycv2.cvtGray(mycv2.loadImage(PARAMS["paths"]["image"]))

fe = FreeEnergy(image)
#fe.renderImage()
fe.computeDerivatives()
#fe.renderDerivatives()
fe.iterate()
