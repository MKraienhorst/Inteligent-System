# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:02:40 2018
Von der Website https://stackoverflow.com/questions/4292249/...
...automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
@author: finn-
"""

def auto_canny(image, sigma=0.33):
     # compute the median of the single channel pixel intensities
     v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged