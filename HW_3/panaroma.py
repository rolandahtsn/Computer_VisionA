import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac,compositeH

# Import necessary functions
I1 =  cv2.imread('data/pl.jpg')
I2 = cv2.imread('data/pr.jpg')
opts = get_opts()

#Image shape to compress to: 1080.1457
# I1 = cv2.resize(I1,dsize = [1457,1080])
# I2 = cv2.resize(I2,dsize = [1457,1080])

#Get corresponding points and matches
matches, locs1,locs2 = matchPics(I1,I2,opts)

#Compute the homography
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
print("HEre")
#Use composite H? Transform geometrically one image by using the matrix H
warped_pan = compositeH(bestH2to1, I1, I2)

#Align and blend the images
cv2.imshow("Img 1",warped_pan)
warped_pan[0:I2.shape[0], 0:I2.shape[1]] = I1
cv2.imshow("Warped image",warped_pan)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Q4
