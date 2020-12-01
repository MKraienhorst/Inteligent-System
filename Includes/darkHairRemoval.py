# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:15:27 2018

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


def darkHairRemoval(imgOrig,img):
     ## start with dark hair removal
     # not working very good, alot of times parts of the lesion are taken away, additionaly hair removal works only if there are only few hairs

     imgNew = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

     # gaussian blur to smooth the edges -> less of the lesion is taken away (and less hair)
     blurNew = cv2.blur(img,(kernelDilErSize,kernelDilErSize))
     # canny filter to detect edges
     edges = cv2.Canny(blurNew,cannyThresh1,cannyThresh2)
     
     ## get hair mask
     # define different kernels
     kernel = np.ones((kernelDilErSize, kernelDilErSize), np.uint8)
     kernelClose = np.ones((kernelDilSize, kernelDilSize), np.uint8)
     kernelClose2 = np.ones((kernelDilSize+2, kernelDilSize+2), np.uint8)
     # fill in canny edges to get 
     closing1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
     #closingDilEr = cv2.morphologyEx(closing1, cv2.MORPH_OPEN, kernel)
     # dilitation and erosion of canny edges
     closingDil = cv2.dilate(closing1,kernelClose2,1)
     mask = cv2.erode(closingDil,kernelClose,1)
     closingDilEr2 = cv2.dilate(mask,kernelClose2,1)
     mask = cv2.erode(closingDilEr2,kernelClose,1)
     
     imgNew = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
     #imgNew2 = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

     #cv2.imshow('Canny Filter',edges)
     #cv2.imshow('Canny Filter Closed areas',mask)
     #cv2.imshow('Original Image',imgOrig)
#     cv2.imshow('Preprocessed Image Inpainting Telea',imgNew)
     #cv2.imshow('Preprocessed Image Inpainting NS',imgNew2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
     
     return imgNew
## dark hairs are removed  