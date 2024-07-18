import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''

def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    x1,y1 = pts1[:,0],pts1[:,1]
    x1_prime,y1_prime = pts2[:,0],pts2[:,1]
    
    p1 = C1[0,:].T
    p2 = C1[1,:].T
    p3 = C1[2,:].T

    p1_prime = C2[0,:].T
    p2_prime = C2[1,:].T
    p3_prime = C2[2,:].T

    #x = P*X where P is the camera matrix
    non_homo = np.zeros((pts1.shape[0],4))
    homogenous = np.zeros((pts1.shape[0],3))
    repr_error = 0

    proj_norm = np.zeros((3,pts1.shape[0]))
    for i in range(0,pts1.shape[0]):
        row1 = np.array([y1[i]*p3-p2])
        row2 = np.array([x1[i]*p3-p1])
        row3 = np.array([y1_prime[i]*p3_prime-p2_prime])
        row4 = np.array([x1_prime[i]*p3_prime-p1_prime])
        A = np.vstack((row1,row2,row3,row4)) #4x4
    # import pdb; pdb.set_trace()
    # (3) Solve for the least square solution using SVD.
        u,s,vT = np.linalg.svd(A)
        F = vT[-1,:]
    #Non-homogenous is form (X,Y,Z,1) and Homogenous is form (X,Y,Z) 
        homogenous[i] = F[0:3]/F[-1] 
        non_homo[i,:] = np.array([homogenous[i][0],homogenous[i][1],homogenous[i][2],np.ones(1)],dtype=object)#Non-homogenous matrix

    # # (4) Calculate the reprojection error using the calculated 3D points and C1 & C2
    #(pts1[i]-proj1)**2 proj1 = np.dot(C1,non_homo[i,:].T)
    #proj_matrix_1 = C1*[X,Y,Z,1]

    proj_matrix_1 = np.dot(C1,non_homo.T).T
    proj_matrix_2 = np.dot(C2,non_homo.T).T
    
    #Reprojection error is the squared error between the true image coordinaties of a point and
    #the projected coordinations of hypothesized 3D points
    # for i in range(0,pts1.shape[0]):
    #x1 = P11(x)+P12(y)+P13(Z)+P14/Z --> x1 - P(1)/P(3); same for y1
    # repr_error = np.sum((pts1[i]- np.dot(C1,non_homo[i])**2))+np.sum((pts2[i]-np.dot(C2,non_homo[i])))

    ##Keeping track of 3D Points; Homogenize coordinates again (divide by Z)
    for i in range (0, pts1.shape[0]):
        proj_norm = proj_matrix_1[i]/proj_matrix_1[i][-1] #3XN
        proj_norm2 = proj_matrix_2[i]/proj_matrix_2[i][-1]

    #Reprojection error; Keep track of error
        repr_error += np.sum((pts1[i]- proj_norm.T[0:2].T)**2)+np.sum((pts2[i]-proj_norm2.T[0:2].T)**2)
    # import pdb; pdb.set_trace()

    return non_homo, repr_error

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M2 = camera2(E)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    
    #Get C1: 3x4 camera matrix; M1 is just identity matrix
    C1 = np.dot(K1,M1)
    error_ = 0 #tracking best_error
    p_save = 0 #saving P
    # C2_save = np.zeros((len(M2),3))

    for i in range (0,M2.shape[2]):
        C2 = np.dot(K2,M2[:,:,i]) #Getting C2 values and check the last index of M2 since that is where the values are stored
        p,error_2 = triangulate(C1,pts1,C2,pts2)
        #The z needs to be a positive value since negayive z means point is behind the camera
        
        if (p[:,-1] >= 0).all():
           # Keep track of the projection error through best_error and retain the best one. 
            error_ = error_2
            p_save = p
            M2_ = M2[:,:,i]
            break #Keep M2s; If there is no break then the correct M2 is not found
        
    np.savez('code/data/q3_3.npz', M2_,C2, p_save)
    return M2_, C2, p_save



if __name__ == "__main__":

    correspondence = np.load('code/data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('code/data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('code/data/im1.png')
    im2 = plt.imread('code/data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    
    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert(err < 500)