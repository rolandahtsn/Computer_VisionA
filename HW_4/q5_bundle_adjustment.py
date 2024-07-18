import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_1_essential_matrix import essentialMatrix
from q3_2_triangulate import findM2,triangulate

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''

def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # Replace pass by your implementation
    #First compute the fundamental matrix F 

    #Keep track of inliers
    current_inliers = 0
    inliers = []
    numb_inliers = 0
    ct_inliers = []

    p_prime = np.zeros((pts1.shape[0],3))
    for i in range(nIters):
        print("Here:         ",i)
        index = np.random.choice(pts1.shape[0],8,False)
        
        x1 = pts1[index]
        x2 = pts2[index]

        F_ = eightpoint(x1,x2,M) #My sevenpoint is not that accurate..

        #Calculate distance for each correspondence
        for k in range(pts1.shape[0]):
            p_prime = np.append(pts2[k],1)
            error = np.dot(F_,p_prime)

            x2_x = error[0]
            x2_y = error[1]
            distance_1 = np.sqrt(x2_x**2+x2_y**2)
            distance = abs(np.dot(p_prime.T,error)/distance_1)

            #Find the inliers
            # error_ = error/np.sqrt(np.sum(error[:,2:]**2,axis = 0))
            # dist = abs(np.sum(homo_2*error_,axis = 0))
            if (distance<= tol).all():
                numb_inliers+=1
                ct_inliers.append(1)
            # else:
            #     ct_inliers.append(0)
            #     numb_inliers = 0
            
            if numb_inliers > current_inliers:
                current_inliers = numb_inliers
                inliers = ct_inliers
                F = F_

    if F[2,2] != 1:
        F = F/F[2,2]

    return F, inliers
'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r) #theta = ||r||
    u = r/theta
    if theta <= np.pi:
        if theta == 0:
            R = np.diag([1,1,1])
        else:
            ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
            
            u_t = (u.reshape(3,1) @ u.reshape(3,1).T)
            R = np.diag([1,1,1])*np.cos(theta)+(1-np.cos(theta))*u_t+ux*np.sin(theta)
    else:
        R = np.eye(3)
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R-R.T)//2
    rho = np.array([A[2][1],A[0][2],A[1][0]]).T
    s = np.linalg.norm(rho)
    c = (R[0][0]+R[1][1]+R[2][2]-1)//2

    # r = None
    if (s == 0 and c == 1):
        r = np.zeros((3,1))

    elif (s == 0 and c == -1):
        # index = np.where((R+np.diag[1,1,1]) != 0)
        # v = (R+np.diag[1,1,1])[:,index]
        I = R + np.diag([1,1,1])
        for i in range(3):
            if (np.count_nonzero(I[:,i]))>0 and (np.count_nonzero(I[:,i]))!=0:
                v = I[:, i]            
        u = v/np.linalg.norm(v)
        r_ = u*np.pi

        if ((np.linalg.norm(r) == np.pi) and (r[0][0] == 0 and r[1][0] == 0 and r[3][0]<0) 
        or (r[0][0] == 0 and r[1][0]<0) or (r[0][0]<0)):
            
            r = -r_ #r negative
        else:
            r = r_

    else:
        u = rho/s
        theta = np.arctan2(s,c)
        r = u*theta
    return r



'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    C1 = np.dot(K1,M1)
    
    w = np.reshape(x[:-6],shape = (p1.shape[0],3))
    t2 = np.reshape(x[-6:-3],shape = (3,1))
    r2 = np.reshape(x[-3:],shape = (3,1))
    R2 = rodrigues(r2)
    
    M2 = np.htsack(R2,x[-3:].reshape(3,1))
    P_homo = np.hstack(w,np.ones(p1.shape[0],1))
    C2 = np.dot(K2,M2)

    p1_= np.dot(C1,P_homo.T).T
    p2_ = np.dot(C2,P_homo.T).T

    p1_hat = p1_[:2,:]/p1_[2,:]
    p2_hat =  p1_[:2,:]/p1_[2,:]

    
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])])
    return residuals
'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass

    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE

    R2_ = M2_init[:,:3]
    t2_ = M2_init[:,:3]
    x_0 = np.concatenate(P_init.flatten(),invRodrigues(R2_),t2_)

    fsu = lambda x: np.sum((rodriguesResidual(K1,M1,p1,K2,p1,x)))






    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('code/data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('code/data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('code/data/im1.png')
    im2 = plt.imread('code/data/im2.png')
    
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))


    # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    # assert(F.shape == (3, 3))
    # assert(F[2, 2] == 1)
    # assert(np.linalg.matrix_rank(F) == 2)
    

    # YOUR CODE HERE


    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())
    # print(invRodrigues(mat))

    

    # assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    # assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)



    # YOUR CODE HERE
    F = eightpoint(noisy_pts1[inliers, :], noisy_pts2[inliers, :], M=np.max([*im1.shape, *im2.shape]))
    E = essentialMatrix(F, K1, K2)

    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = np.dot(K1,M1)
    # M2, C2, P = findM2(F, noisy_pts1, noisy_pts2, intrinsics, 'code/data/q5_3.npz')