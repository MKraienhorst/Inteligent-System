# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:36:32 2018

@author: finn-alex
"""
import cv2 
import numpy as np


def regseg(image,threshold):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   # blur = cv2.GaussianBlur(image,(5,5),0)
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('image',thresh)
    cv2.imshow('original',image)
    cv2.waitKey(0)                          #press any key to close the figure
    cv2.destroyAllWindows() 
   
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
     
     # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
     
     # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
     
      # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
     
     # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
     
     # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]
    
    cv2.imshow('original',image)
    cv2.waitKey(0)                          #press any key to close the figure
    cv2.destroyAllWindows() 
    return ret
