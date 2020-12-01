import sys
sys.path.append('./Includes')
from brightHairRemoval import brightHairRemoval
from darkHairRemoval import darkHairRemoval
from segmentmole import segmentmole
from feature import FeatureVec

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import configparser
import random
import decimal as d
from color_texture_feature import colorhsv

# reading image path from config file is not working yet, do not how to pass the "r" (see "path")
# path2 = ( config['DEFAULT']['testImgDir'])
path = (r"G:\ISM Projekt\ISIC2018_Task3_Training_Input_test")

def initialize(path):
     # load image list
     path = (r"G:\ISM Projekt\ISIC2018_Task3_Training_Input_test")
     os.chdir(path)
     images=[]
     for image in os.listdir(path):
          if image.endswith('.jpg') or image.endswith('.JPG'):
               images.append(image)
     
     return images 






def matrixgrap(hair,seg):
    images = initialize(path)
    im = []
    color = []
    for i in range(len(images)):
    #for i in range(3):
         # read image
         print(i/(len(images))*100)
         img = cv2.imread(images[i]) 
         
         # preprocessing function
         if hair == 1:
             imgNew1 = brightHairRemoval(img)               
             img = darkHairRemoval(img,imgNew1)  
         else: 
             pass
         # region segmentation function 
         # returns the from the original image cropped segmented region of the mole
         if seg == 1:
             img = segmentmole(img,0,'hull')       
         else: 
             pass
         #cv2.imshow('mit Haaren',image)       
         #image = segmentmole(imgNew2,rot=None,shape=None)    
         #cv2.imshow('ohne Haare',image)      
         #cv2.waitKey(0)          
         #cv2.destroyAllWindows() 
         color1 = colorhsv(img) 
         color.append(color1)
         
         im.append(img)
    
    color = np.asarray(color)
    color = color[:,:,0]
    com = FeatureVec(im)
    comcolor = np.hstack([com,color])
    return com, comcolor, color
    
 
#
com, comcolor, color = matrixgrap(1,1)
os.chdir(r'C:\Users\finn-\Documents\MEGA\TUHH\Master\ISM\GIT\Documents')
np.save('com_with_Hairremoval_and_with_Segmentmole_hull',com)
np.save('comcolor_with_Hairremoval_and_with_Segmentmole_hull',comcolor)
np.save('color_with_Hairremoval_and_with_Segmentmole_hull',color)
#
com, comcolor, color = matrixgrap(1,0)
os.chdir(r'C:\Users\finn-\Documents\MEGA\TUHH\Master\ISM\GIT\Documents')
np.save('com_with_Hairremoval_and_without_Segmentmole_hull',com)
np.save('comcolor_with_Hairremoval_and_without_Segmentmole_hull',comcolor)
np.save('color_with_Hairremoval_and_without_Segmentmole_hull',color)

com, comcolor, color = matrixgrap(0,1)
os.chdir(r'C:\Users\finn-\Documents\MEGA\TUHH\Master\ISM\GIT\Documents')
np.save('com_without_Hairremoval_and_with_Segmentmole_hull',com)
np.save('comcolor_without_Hairremoval_and_with_Segmentmole_hull',comcolor)
np.save('color_without_Hairremoval_and_with_Segmentmole_hull',color)
#
com, comcolor, color = matrixgrap(0,0)
os.chdir(r'C:\Users\finn-\Documents\MEGA\TUHH\Master\ISM\GIT\Documents')
np.save('com_without_Hairremoval_and_without_Segmentmole_hull',com)
np.save('comcolor_without_Hairremoval_and_without_Segmentmole_hull',comcolor)
np.save('color_without_Hairremoval_and_without_Segmentmole_hull',color)


