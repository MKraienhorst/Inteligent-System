# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:14:21 2018

@author: alexa
"""

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import random
import decimal as d
import configparser

# open the configuration file and initialise variables
config = configparser.ConfigParser()
config.read('./Documents/init.ini')

kernelDilErSize = int(config['artefactRemoval']['kernel-diletation-erosion'])
kernelDilSize = int(config['artefactRemoval']['dilation-kernel-size'])
brightThresh = int(config['artefactRemoval']['bright-hair-threshold'])
brightGausSize = int(config['artefactRemoval']['bright-hair-gaus-size'])
cannyThresh1 = int(config['artefactRemoval']['canny-threshold-1'])
cannyThresh2 = int(config['artefactRemoval']['canny-threshold-2'])

def brightHairRemoval(img):
     ## input:
     #   an opencv image
     #   + the size of the kernel used for gaussian blur
     #   + a binary threshold
     #   + size for the dilitation and erosion
     ## output:
     #   image with removed blond hairs
     
     ## start with removing all bright/blond hairs
     # get image sizes
     kernel = np.ones( (kernelDilErSize, kernelDilErSize), np.uint8 )
     rows,cols,colors = img.shape
     # dilate and erose the img
     imgErDil = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
     # compute difference between dilated and erosed image
     imgErodeDif = img - imgErDil
     # convert to Gray Scale
     imgGrayErodeDif2 = cv2.cvtColor(imgErodeDif, cv2.COLOR_BGR2GRAY)
     # Add Gausian blur to get rid of small noise
     imgErodeDif2 = cv2.blur(imgGrayErodeDif2, (brightGausSize, brightGausSize))
     # threshold to binarize image, now we have a mask of the dilation+ersion area -> this is where hairs are
     ret, imgBinErodeDif2 = cv2.threshold(imgErodeDif2, brightThresh, 1, cv2.THRESH_BINARY)
     # just copy the image
     imgNew = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
     # add the dilatio pixels to the original image, if there is hair
     for ii in range(rows):
          for iii in range(cols):
               if imgBinErodeDif2[ii,iii] == 1:
                    for j in range(colors):
                         imgNew[ii,iii,j] = imgErDil[ii,iii,j]
     # show difference between both images
#     cv2.imshow('Preprocessed Image', imgNew)
#     cv2.imshow('Original Image', img)
     # press any key to close the figure
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
     return imgNew