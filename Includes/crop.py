# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:40:49 2018

@author: finn-
"""
import numpy as np
def crop(image):            
    mask = image > 0

# Coordinates of non-black pixels.
    coords = np.argwhere(mask[:,:,1])

# Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

# Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    return cropped


            