import sys
sys.path.append('./Includes')
from brightHairRemoval import brightHairRemoval
from darkHairRemoval import darkHairRemoval
from segmentmole import segmentmole

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import configparser
import random
import decimal as d
import csv
import configparser
# from color_texture_feature import colorhsv

# open the configuration file and initialise variables
config = configparser.ConfigParser()
config.read('./Documents/init.ini')

FeatureMatrixName = config['DEFAULT']['FeatureMatrixName']

# reading image path from config file is not working yet, do not how to pass the "r" (see "path")
path = (r"..\ISIC2018_Task3_Training_Input_test")
# path = (r"..\ISIC2018_Task3_Test_Input")
path2 = (r"..\intelligentsystems\Documents")

def initialize(path):
     # load image list
     os.chdir(path)
     allImages = []
     for image in os.listdir(path):
          if image.endswith('.jpg') or image.endswith('.JPG'):
               allImages.append(image)
     
     return allImages



allImages= initialize(path)

for i in range(len(allImages)):
     # read image
     os.chdir(path)
     img = cv2.imread(allImages[i]) 
     
     # preprocessing function
     imgNew1 = brightHairRemoval(img)      
     cv2.imwrite(("../ISIC2018_Task3_Training_Input_test_BrightHair/" + allImages[i]),imgNew1)           
     imgNew2 = darkHairRemoval(img,imgNew1)  
     cv2.imwrite(("../ISIC2018_Task3_Training_Input_test_BothHair/" + allImages[i]),imgNew2)
     imgNew2 = darkHairRemoval(img,imgNew2)

     # region segmentation function 
     # returns the from the original image cropped segmented region of the mole
     
     image = segmentmole(imgNew2,rot=None,shape='hull')
     cv2.imwrite(("../ISIC2018_Task3_Training_Input_test_BothHair_Segment/" + allImages[i]),image)
     image2 = segmentmole(img,rot=None,shape='hull')    
     cv2.imwrite(("../ISIC2018_Task3_Training_Input_test_Segment/" + allImages[i]),image2)
     if((i % 10) == 0):
        print(i)



