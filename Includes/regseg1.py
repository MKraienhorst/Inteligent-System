# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:36:32 2018

@author: finn-alex
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from crop import crop
import math


def regseg1(image_bin, image_orig, rot = None, cropim = None, shape = None):
    
    rot = rot or 0
    shape = shape or "hull"
    cropim = cropim or 1
    #thresholding again
    image = image_bin
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   

        
    # contours of image are found and plotted (only for image with threshold)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  
##finds most close contour to image center
    dist = np.zeros(np.asarray(contours).shape)     
    mid = np.array([])
    
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        # calculate x,y coordinate of center + prevent division by zero
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        dist[i] = math.hypot(cX-225, cY-300) #calculates eucl. distance

    mindist = np.argmin(dist) 
    cnt = contours[mindist]


    #cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
    #cv2.drawContours(image, contours, -1, (0,255,0), 3) #all contours are showns
    
##contour properties, maybe used for calculation later..
    #Solidity is the ratio of contour area to its convex hull area.
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    #It is the ratio of width to height of bounding rect of the object
    x,y,w,h = cv2.boundingRect(cnt) # does not take into account the rotation of an object
    aspect_ratio = float(w)/h
    #Extent is the ratio of contour area to bounding rectangle area.
    rect_area = w*h
    extent = float(area)/rect_area
    #plot bounding box
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # bounding box with minimal area (Rotation of an object is taken into account)
    (x1,y1),(w1,h1),r = cv2.minAreaRect(cnt)
    rect = ((x1,y1),(w1,h1),r) 
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(image,[box],0,(0,0,255),2)
    rect_area1 = w1*h1
    extent1 = float(area)/rect_area1 #Extent is the ratio of contour area to bounding rectangle area.
    aspect_ratio1 = float(w1)/h1 #It is the ratio of width to height of bounding rect of the object
    # defines a line through the mole 
    # (such that sum(p(r))=min, for all points on contour.
    # p(r)= distance between point and fittet line. for CV2_DIST_L2: p(r)=r^2/2) 
    
    #look for symmetrie by that line! maybe with the area on each side? 
    
    #rows,cols = image.shape[:2]
    #[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    #lefty = int((-x*vy/vx) + y)
    #righty = int(((cols-x)*vy/vx)+y)
    #cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)
    # defines a circle with smallest radius around the mole 
    #(x,y),radius = cv2.minEnclosingCircle(cnt)
    #center = (int(x),int(y))
    #radius = int(radius)
    #cv2.circle(image,center,radius,(0,255,0),2)

## create desired mask    
    
    image[:,:,:] = [0,0,0]
    
    if shape == "hull":
        cv2.fillPoly(image, pts =[hull], color=(255,255,255))
    elif shape == "rect":
        cv2.fillPoly(image, pts = [box], color=(255,255,255))
    #rotate 
    if rot == 1: 
       rows = image.shape[0]
       cols = image.shape[1]
       M = cv2.getRotationMatrix2D((cols/2,rows/2),r,1)
       image = cv2.warpAffine(image,M,(cols,rows))
      
    else:
       pass
       
    #crop
    if cropim == 1: 
        mask_out=cv2.subtract(image,image_orig)
        mask_out=cv2.subtract(image,mask_out)
        mask_out = crop(mask_out)
    else:
        pass
        
    #plot
#    cv2.imshow('segmented and cropped mole',mask_out)
#    cv2.waitKey(0)                          #press any key to close the figure
#    cv2.destroyAllWindows() 

    #return (mask_out, ret, contours, area, hull, hull_area, solidity, aspect_ratio, extent, aspect_ratio1, extent1)
    return mask_out
