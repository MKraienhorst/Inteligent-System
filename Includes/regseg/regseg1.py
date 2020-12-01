# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:36:32 2018

@author: finn-alex
"""
import cv2 
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
#from myshow import myshow

def regseg1(image,threshold1):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    cv2.imshow('image',thresh)
    cv2.imshow('original',image)

#    # noise removal
#    kernel = np.ones((3,3),np.uint8)
#    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#     
#    # sure background area
#    sure_bg = cv2.dilate(opening,kernel,iterations=3)
#     
#    # Finding sure foreground area
#    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#    
#    # Finding unknown region
#    sure_fg = np.uint8(sure_fg)
#    unknown = cv2.subtract(sure_bg,sure_fg)
#     
#    # Marker labelling
#    ret, markers = cv2.connectedComponents(sure_fg)
#     
#    # Add one to all labels so that sure background is not 0, but 1
#    markers = markers+1
#     
#    # Now, mark the region of unknown with zero
#    markers[unknown==255] = 0
#    
#    markers = cv2.watershed(image,markers)
##    image[markers == -1] = [255,0,0]
#    
#    # make markers into contour to use contour properties 
#    markers1 = markers.astype(np.uint8)
#    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
#    cv2.imshow('m2', m2)
#    _, contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#    # defines contour of largest area 
#    contours = sorted(contours, key=cv2.contourArea, reverse=True) 
#    cnt = contours[1]
#    # draw contours = markers
#    cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
#    cv2.imshow('contours after watersehd',image)
        
#    # contours of image are found and plotted (only for image with threshold)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#    # defines contour of largest area 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)    
    cnt = contours[1]

#    # optional: defines wheather a contour is closed or not
#    closed_contours = []
#    for cnt in contours:
#       if cv2.isContourConvex(cnt) == True:
#          closed_contours.append(cnt)
#       else:
#          pass
    

    cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
#    cv2.drawContours(image, contours, -1, (0,255,0), 3) #all contours are showns
    
    #contour properties
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
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    
    # bounding box with minimal area (Rotation of an object is taken into account)
    (x1,y1),(w1,h1),r = cv2.minAreaRect(cnt)
    rect = ((x1,y1),(w1,h1),r) 
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)
    rect_area1 = w1*h1
    extent1 = float(area)/rect_area1 #Extent is the ratio of contour area to bounding rectangle area.
    aspect_ratio1 = float(w1)/h1 #It is the ratio of width to height of bounding rect of the object
    
    # defines a line through the mole 
    # (such that sum(p(r))=min, for all points on contour.
    # p(r)= distance between point and fittet line. for CV2_DIST_L2: p(r)=r^2/2) 
    look for symmetrie by that line! maybe with the area on each side? 
    
    
    rows,cols = image.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)
    # defines a circle with smallest radius around the mole 
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(image,center,radius,(0,255,0),2)
    
    cv2.drawContours(image,[hull],0,(255,0,0),2)
    cv2.imshow('o',image)
    cv2.waitKey(0)                          #press any key to close the figure
    cv2.destroyAllWindows() 

    return (ret, contours, area, hull, hull_area, solidity, aspect_ratio, extent, aspect_ratio1, extent1)
