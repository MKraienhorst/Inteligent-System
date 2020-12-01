# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:36:32 2018

@author: finn - alex

parameters: 
    image: original image of the mole
    rot: rotation of final image: rot= 0 -> no rotation, rot = 1 -> rotation , 0 by default.
    shape: "hull" -> hull shape, "rect" -> smallest rectangle , "rect" by default.
    USE DEFAULT VALUES!! 
"""   
#######################################################################
#tuning parameter
#    evtl Threshold and gray..
#    clahe or no clahe? 
#    HSV or RGB?
#    in noise removal:      
#    kernelsize: 3,5,7,9.. 
#    in opening: Morph_open or Morph_close or both
#    iterations in sure.. back and foreground
#    in finding sure foreground area: erode or dist. transform

#######################################################################

import cv2 
import numpy as np
import skimage
from regseg1 import regseg1
import configparser
import os

# open the configuration file and initialise variables
config = configparser.ConfigParser()
config.read('./Documents/init.ini')

# load variables from configuration file init.ini
KernelSizeMedianFilter = int(config['RegionSegmentation']['KernelSizeMedianFilter'])
KernelSizeBlurFilter = int(config['RegionSegmentation']['KernelSizeBlurFilter'])
KerneSizeOeningClosing = int(config['RegionSegmentation']['KernelSizeOpeningClosing'])
OpeningIterations = int(config['RegionSegmentation']['OpeningIterations'])
DilateIterations = int(config['RegionSegmentation']['DilateIterations'])
ErodeIterations = int(config['RegionSegmentation']['ErodeIterations'])



def segmentmole(imgNew2,rot=None,shape=None):

    
    image = imgNew2.copy()
    
## get grayscale version of image ..different ways to do it.. 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    gray = gray[:,:,1]
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    gray = clahe.apply(gray)
    #gray = cv2.equalizeHist(gray)
    
## filtering and thresholding    
    gray= cv2.medianBlur(gray,KernelSizeMedianFilter)
    gray= cv2.medianBlur(gray,KernelSizeMedianFilter)
    gray = cv2.blur(gray,(KernelSizeBlurFilter,KernelSizeBlurFilter))
    
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
## plots for debugging    
#    cv2.imshow('thresh_image',thresh)
#    cv2.imshow('original',image)
#    cv2.waitKey(0)                          #press any key to close the figure
#    cv2.destroyAllWindows() 
   
## further filtering
#    kernel = np.ones((5,5),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KerneSizeOeningClosing, KerneSizeOeningClosing))
    #opening and closing
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = OpeningIterations)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
 
## sure background area
    sure_bg = cv2.dilate(thresh,kernel,DilateIterations)
    
## Finding sure foreground area
    sure_fg = cv2.erode(thresh,kernel,ErodeIterations) 
    # alternative: 
    #opening = cv2.distanceTransform(opening,cv2.DIST_C,5)
    #ret, sure_fg = cv2.threshold(dist_transform,.22*dist_transform.max(),255,0)
   
## Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    

## Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
     
     # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
     
     # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(image,markers)
    #markers[markers>0] =1
    
    #correcting markers
    markers[0,:] = 1
    markers[:,0] = 1
    markers[-1,:] = 1
    markers[:,-1] = 1
    
    image[markers == -1] = [255,0,0]

## ploting    
    #res = np.hstack((gray,image[:,:,0]))
    #cv2.imshow('Watershed|Median',res)
    #cv2.imshow(' watershed',image)   
    #cv2.imshow('dist_trans',dist_transform) 
    #cv2.imshow('background',sure_bg)
    #cv2.imshow('foreground',sure_fg)
    #cv2.imshow('opening',opening)
    #cv2.imshow('unknown',unknown)
    #cv2.imshow('median',gray)
    #cv2.waitKey(0)                          #press any key to close the figure
    #cv2.destroyAllWindows() 
    
    image[markers==1] = 0 
    image[markers>1] =255
    
    image = regseg1(image,imgNew2,rot,0,shape) 
    
    return image
