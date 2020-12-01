import sys
sys.path.append('./Includes')
from brightHairRemoval import brightHairRemoval
from darkHairRemoval import darkHairRemoval
from segmentmole import segmentmole
from feature import FeatureVec
#from color_texture_feature import colorhsv

import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import configparser
import random
import decimal as d

import pprint

path = (r"C:\Users\mkrai\Documents\ISIC_artefical hair") ### adjust path ###
os.chdir(path)
hairdata = []
for image in os.listdir(path):
    if image.endswith('.jpg') or image.endswith('.JPG'):
        hairdata.append(image)

textureNoRemoval = []
color = []
colorOriginal = []
im = []
imOriginal = []
for i in range(len(hairdata)):
     # read image
     imgOriginal = cv2.imread(hairdata[i])
     #store feature of the image without hair removal
     colorOriginal1 = [2+0.1*i, 4+0.3*i, 5+0.4*i, 6+0.6*i]
     #colorOriginal1 = colorhsv(imgOriginal)
     colorOriginal.append(colorOriginal1)
     imOriginal.append(imgOriginal)

     # preprocessing function
     imgNew1 = brightHairRemoval(imgOriginal)                
     image = darkHairRemoval(imgOriginal,imgNew1)   
     
     # get feature of the image with hair removal
     #color1 = colorhsv(image)
     color1 = [1+0.2*i,4+0.1*i,2+0.5*i,5+0.5*i]
     color.append(color1)
     im.append(image)
     print(i)

color = np.asarray(color)
#color = color[:,:,0]
colorOriginal = np.asarray(colorOriginal)
#colorOriginal = colorOriginal[:,:,0]
com = FeatureVec(im)
numImgs = 3 #amount of variation of 1 image 
comOriginal = FeatureVec(imOriginal)
diffcolor = np.zeros((4,numImgs,13))
#diffcolor = np.zeros((numImgs,4,len(color[0]))
#diffcolorOriginal = np.zeros(numImgs,4,len(color[0]))
diffcom = np.zeros((4,numImgs,10))
diffcomOriginal = np.zeros((4,numImgs,10))
for i in range(4):
    for ii in range(numImgs):
        for iii in range(len(color[0])):
            print(len(color))
            print(len(color[0]))
            print((color[i*numImgs, iii] - color[i*numImgs+(ii+1), iii])/ (color[i*numImgs, iii]) * 100)
            diffcolor[i,ii,iii] = (color[i*numImgs, iii] - color[i*numImgs+(ii+1), iii])/ (color[i*numImgs, iii]) * 100
        for iii in range(len(com[0])):
            #difference between the features of the original image and the one with arteficial hair 
            print(com[i*numImgs, iii])
            print(com[i*numImgs+(ii+1), iii])
            print((com[i*numImgs, iii] - com[i*numImgs+(ii+1), iii])/ (com[i*numImgs, iii]) * 100)
            diffcom[i,ii,iii] = (com[i*numImgs, iii] - com[i*numImgs+(ii+1), iii])/ (com[i*numImgs, iii]) * 100
            #difference between the features of the original image and the one with arteficial hair after preprocessing
            diffcomOriginal[i,ii,iii] = (comOriginal[i*numImgs,iii] - comOriginal[i*numImgs+ii,iii])/comOriginal[i*numImgs,iii] * 100
comcolor = np.hstack([com,color])
comcolorOriginal = np.hstack([comOriginal,colorOriginal])
for i in range(len(com[0])):
    output = []
    #add Headerline
    output.append(' , ' + str(hairdata[0]) + ', ' + str(hairdata[numImgs*1]) + ', ' + str(hairdata[numImgs*2]) + ', ' + str(hairdata[numImgs*3]))
    output.append('aenderung , ' + str(diffcom[0,0,i]) + ', ' + str(diffcom[1,0,i]) + ', ' + str(diffcom[2,0,i]))
    output.append('ohne , '+ str(diffcom[0,1,i]) + ', ' + str(diffcom[1,1,i]) + ', ' + str(diffcom[2,1,i]))
    output.append('preprocess , '+ str(diffcom[0,2,i]) + ', ' + str(diffcom[1,2,i]) + ', ' + str(diffcom[2,2,i]))
    output.append(' , , , ')
    output.append('aenderung , '+ str(diffcomOriginal[0,0,i]) + ', ' + str(diffcomOriginal[1,0,i]) + ', ' + str(diffcomOriginal[2,0,i]))
    output.append('mit , '+ str(diffcomOriginal[0,0,i]) + ', ' + str(diffcomOriginal[1,0,i]) + ', ' + str(diffcomOriginal[2,0,i]))
    output.append('preprocess , '+ str(diffcomOriginal[0,0,i]) + ', ' + str(diffcomOriginal[1,0,i]) + ', ' + str(diffcomOriginal[2,0,i]))
    #save
    np.savetxt('differenceOfComFeature' + str(i) + '.csv',output,fmt='%s')