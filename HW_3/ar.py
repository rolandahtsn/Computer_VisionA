import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import compositeH, computeH_ransac

from helper import loadVid
from opts import get_opts
from crop_center import center_crop
import multiprocessing
#import tenserflow as tf


def process():
    opts = get_opts()

    #Write script for Q3.1

    # Videos
    ar_source = loadVid('data/ar_source.mov')
    book = loadVid('data/book.mov')
    cv_cover = cv2.imread('data/cv_cover.jpg')
    #Total number of frames; helper returns shape (#,#,#,#) first index is number of frames
    nvideo_frames = min(ar_source.shape[0],book.shape[0])
    n_cpu = 8
    writer = cv2.VideoWriter('result/ar_RH.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (book.shape[2], book.shape[1]))
    
    for i in range(0,nvideo_frames):
    #for i in range(0,10):
        # continue running panda video if it ends

        # Get mactches, locs, locs2 from matchPics
        I1 = book[i]
        I2 = cv_cover

        #We need to take into account the difference in ratio of height and width between the two images and then multiply them

        #matches, locs1, locs2 = matchPics(I2, I1, opts)
        
        match_parameters = [(I1,I2,opts)]

        with multiprocessing.Pool(processes = n_cpu) as pool:
            results = pool.starmap(matchPics,match_parameters)
        pool.close()

        matches = results[0][0]
        locs1 = results[0][1]
        locs2 = results[0][2]
        
        #HarryPotterize for kungfu panda to cover of book
        #Need to flip the [row,col] --> (y,x) to be (x,y)
        f_locs1 = np.fliplr(locs1)
        f_locs2 = np.fliplr(locs2)
        
        #Get locations where there are matches in the image
        locs1_ = []
        locs2_ = []
        print("running",i)
        
        for k in matches:
            indx1 = k[0]
            indx2 = k[1]
            
            locs1_.append(f_locs1[indx1])
            locs2_.append(f_locs2[indx2])   
        # import pdb; pdb.set_trace()
        
        
        
    #Get x1 and x2
        x1 = np.array(locs1_)
        x2 = np.array(locs2_)
        
        
        #Compute Homography
        bestH2to1, inliers = computeH_ransac(x1, x2, opts)

        #Crop each frame of kungfu panda to fit onto the book cover
        dim = (cv_cover.shape[1],cv_cover.shape[0])
        cropped_ar = center_crop(ar_source[i],dim)
        
        #Resize frame
        resize_ar = cv2.resize(cropped_ar,dsize = [I2.shape[1],I2.shape[0]])
        
        bookfr = book[i]
        #print("AR shape: ", resize_ar.shape)
        #print("Book Frame Shape: ", bookfr.shape)
        #import pdb; pdb.set_trace()
        frames = compositeH(bestH2to1, resize_ar, bookfr)
        writer.write(frames)

    #Stop writing frames
    writer.release()  

if __name__ == "__main__":
    process()
