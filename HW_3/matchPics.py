import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from PIL import Image

# Q2.1.4

def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

        # I1 = Image.open('../data/cv_cover.jpg')
        # I2 = Image.open('../data/cv_desk.jpg')

        # Convert Images to GrayScale
        img1 = skimage.color.rgb2gray(I1)
        img2 = skimage.color.rgb2gray(I2)

        # Detect Features in Both Images
        locs1 = corner_detection(img1,sigma)
        locs2 = corner_detection(img2,sigma)
        
        # Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(img1,locs1)
        desc2, locs2 = computeBrief(img2,locs2)
       
        # Match features using the descriptors
        matches = briefMatch(desc1,desc2,ratio)
        
        return matches, locs1, locs2
