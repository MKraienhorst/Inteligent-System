
import matplotlib.pyplot as plt
import os
import cv2 
import numpy as np
import pandas as pd 
import SimpleITK as sitk

path2 = (r"..\..\..\ISIC2018_Task3_Training_Input_test")

images=[]
for image in os.listdir(path2):
     if image.endswith('.jpg') or image.endswith('.JPG'):
          images.append(image)
          
os.chdir( path2 )

image = cv2.imread(images[24])  #8060 not the whole mole is on the picture 

## calculate moments
#moments = cv2.moments(im2)
#print(moments)

## Plot images
#cv2.imshow('images[3]',im2)
#cv2.waitKey(0)                          #press any key to close the figure
#cv2.destroyallwindows()   
  
(ret, contours, area, hull, hull_area, solidity, aspect_ratio, extent, aspect_ratio1, extent1) = regseg1(image,1)

