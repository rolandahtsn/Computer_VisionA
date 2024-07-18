import numpy as np
import cv2
import math


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    #Need the Ai matrix derived in question 1.4, x1 = x' and x2 = X
    X = x2
    x_prime = x1
    
    A = np.zeros((2*x1.shape[0],9)) #Count was 9 even though DOF is 8
    for i in range(x1.shape[0]):
        # A[i*2,:] = [-X[i,0],-X[i,1], -1, 0, 0, 0,X[i,0]*x_prime[i,0],X[i,1]*x_prime[i,0],x_prime[i,0]]
        # A[i*2+1,:] = [0,0,0,-X[i,0], -X[i,1],-1,X[i,0]*x_prime[i,1],X[i,1]*x_prime[i,1],x_prime[i,1]]
        
        A[i*2,:] = [X[i,0],X[i,1], 1, 0, 0, 0,-X[i,0]*x_prime[i,0],-X[i,1]*x_prime[i,0],-x_prime[i,0]]
        A[i*2+1,:] = [0,0,0,X[i,0], X[i,1],1,-X[i,0]*x_prime[i,1],-X[i,1]*x_prime[i,1],-x_prime[i,1]]


    #A = np.dot(A.T,A)
    #eig_val, eig_vect  = np.linalg.eig(A)
    u,s,v = np.linalg.svd(np.dot(A.T,A))
    H2to1 = np.reshape((v[-1,:]),(3,3))
    # import pdb; pdb.set_trace()
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    x1_centroid = x1.mean(axis = 0)
    x2_centroid = x2.mean(axis = 0)
    
    #Shift the origin of the points to the centroid
    x1_shift = x1-x1_centroid
    x2_shift = x2-x2_centroid

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_max = np.sqrt(2)/np.max(np.linalg.norm(x1_shift, axis = 1))
    x2_max = np.sqrt(2)/np.max(np.linalg.norm(x2_shift,axis = 1))
    
    # T = s*[[1, 0, -u,],[0, 1, -v], [0, 0, 1/s]]
    # (this is x1/2_max) s = sqrt(2)*n/sum((ui-u)^2+(vi-v)^2)^(1/2)

    u1 = -x1_max*x1_centroid[0]
    v1 = -x1_max*x1_centroid[1]
    u2 = -x2_max*x2_centroid[0]
    v2 = -x2_max*x2_centroid[1]

    #Similarity transform 1
    T1 = np.array(([[x1_max, 0, u1], [0, x1_max, v1], [0, 0, 1]]))
   
    #Similarity transform 2
    T2 = np.array(([[x2_max, 0, u2], [0, x2_max, v2], [0, 0, 1]]))
    
    #Compute homography of points after norm.
    h = computeH(x1_shift*x1_max,x2_shift*x2_max)
    
    H = np.dot(np.linalg.inv(T1),h)
    
    #Denormalization
    H2to1 = np.dot(H,T2)

    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    
    #H will be a homography such that if x2 is a point in locs2 and x1 is a corresponding
    # point in locs1 then x1 = Hx2
    
    bestH2to1 = []
    # x_coord = locs1[:,0]
    # Use when computing distance bewtween points
    
    #Keep track of inliers on line
    current_inliers = 0
    inliers = []
    
    numb_inliers = 0
    ct_inliers = []
    p_prime = np.zeros((locs1.shape[0],3))
    
    
    for i in range(max_iters):
        #Select a random sample of 4 pairs of corresponding points
        
        indexes = np.random.choice(locs1.shape[0],4, False)
        x1 = locs1[indexes]
        x2 = locs2[indexes]
        
        #Compute H norm
        H = computeH_norm(x1,x2)    
        numb_inliers = 0
        # print(i)
        
        #Calculate distance for each correspondence
        for k in range(locs1.shape[0]):
            # p_prime[0].insert(np.append(locs2[0],1))
            p_prime = np.append(locs2[k],1)
            error = np.dot(H,p_prime)
            
            #These are my x and y coords for x1
            x2_x = error[0]
            x2_y = error[1]
            x2_z = error[2]
            
            #These are my x and y coords for x2
            x1_x = locs1[k][0]
            x1_y = locs1[k][1]
            
            # x1_z = locs1[2]
            
            #Now compute the distance between the points: #Should be np.sum(np.sqrt((x2 - x1)^2/Z + (y2-y1)^2)/Z)
            distance = np.sqrt((x2_x/x2_z - x1_x)**2+(x2_y/x2_z - x1_y)**2) #Points have to be normalized divide by Z

            #Check if distance satifies the tolerance for each point pair
            if distance < inlier_tol:
                numb_inliers +=1
                ct_inliers.append(1)
            else:
                ct_inliers.append(0)
                numb_inliers = 0
                
            #Choose the H with the largest number of inliers
            if numb_inliers > current_inliers:
                current_inliers = numb_inliers
                inliers = ct_inliers
                # save H
                bestH2to1 = H
                
            #Recompute H with the inliers
            
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    #Template - Harry potter cover
    # Img is the cover on the desk
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template
    
    #Warp mask by appropriate homography
    warped =  cv2.warpPerspective(template,np.linalg.inv(H2to1),(img.shape[1],img.shape[0]))
    mask = np.ones(shape = (template.shape[2]))

    warped_mask = cv2.warpPerspective(mask,np.linalg.inv(H2to1),(img.shape[1],img.shape[0]))
    
    #Now check where the warped image is equal to zero
    warped_index = np.where(warped == 0)
    img_values_index = img[warped_index]
    warped[warped_index] = img_values_index

    composite_img = warped
    # cv2.imshow('returned_image',composite_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return composite_img





