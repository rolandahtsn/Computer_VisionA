import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import compositeH,computeH_ransac
# Import necessary functions

# Q2.2.4

def warpImage(opts):
    cover = cv2.imread('data/cv_cover.jpg')
    desk = cv2.imread('data/cv_desk.png')
    hp_cover = cv2.imread('data/hp_cover.jpg')
    
    #To ensure that the dimensions for the harry potter image and the cp_cover are the same size we should resize
    resized_hp_cover = cv2.resize(hp_cover,dsize = [cover.shape[1],cover.shape[0]])
    
    #Setting Images
    I1 = desk
    I2 = cover
    I2_2 = hp_cover
    
    #Compute homography using matchPics and computeH_ransac
    matches, locs1, locs2 = matchPics(I2, I1, opts)
    
   
    
    #Need to flip the [row,col] --> (y,x) to be (x,y)
    f_locs1 = np.fliplr(locs1)
    f_locs2 = np.fliplr(locs2)
    
    #Get locations where there are matches in the image
    locs1_ = []
    locs2_ = []

    for i in matches:
        indx1 = i[0]
        indx2 = i[1]
        
        locs1_.append(f_locs1[indx1])
        locs2_.append(f_locs2[indx2])   
    # import pdb; pdb.set_trace()
    
   #Get x1 and x2
    x1 = np.array(locs1_)
    x2 = np.array(locs2_)
    
    #Compute Homography
    bestH2to1, inliers = computeH_ransac(x1, x2, opts)
    
    #Use the homography to warp hp_cover to dimensions of cv_disk image
    returned_image = compositeH(bestH2to1, resized_hp_cover, desk)
    
    cv2.imshow('returned_image',returned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


