import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import skimage.color
import matplotlib.pyplot as plt

#Q2.1.6

def rotTest(opts):

    #Read the image and convert to grayscale, if necessary
    
    image1 = cv2.imread('data/cv_cover.jpg')
    histogram = []
    x_axis = []
    for i in range(36):
        print(i)
        #Rotate Image; Don't include 360 angle
        if i == 0:
            rot_image = scipy.ndimage.rotate(image1,i)
        elif (i >0 and i*10 !=360):
            rot_image = scipy.ndimage.rotate(image1,i*10)
        
        #Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(image1, rot_image, opts)
        if i == 0:
            x_axis.append(i*10)
        elif (i >0 and i*10 !=360):
            x_axis.append(i*10)
        #Update histogram
        histogram.append(len(matches))

    # import pdb; pdb.set_trace()
    #Display histogram
    plt.bar(x_axis,histogram, align='center', alpha=0.5, width=9)
    # plt.hist(x_axis,histogram)
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
