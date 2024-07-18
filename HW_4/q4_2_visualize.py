import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    #Get x2, y2; x1 and y1 passed in to epipolar will be at a single index
    x2 = np.zeros(shape = (1,temple_pts1['x1'].shape[0]))
    y2 = np.zeros(shape = (1,temple_pts1['y1'].shape[0]))

    for i in range(0,temple_pts1['x1'].shape[0]):
        x2_, y2_ = epipolarCorrespondence(im1, im2, F, int(temple_pts1['x1'][i]), int(temple_pts1['y1'][i]))
        x2[0][i] = x2_
        y2[0][i] = y2_

    #Compute M2 matrix and use triangulate to find 3D points
    M2,C2,P = findM2(F, pts1, pts2, intrinsics, filename = 'q4_2.npz') #Triangulate is called in the M2
    
    # homograpy, error_ = triangulate(C1, temple_pts1[0], C2, temple_pts1[1])

    return P



'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('code/data/templeCoords.npz')
    correspondence = np.load('code/data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('code/data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('code/data/im1.png')
    im2 = plt.imread('code/data/im2.png')


    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    P = compute3D_pts(temple_coords_path, intrinsics, F, im1, im2)
    
    xs = P[:,0]
    ys = P[:,1]
    zs = P[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs,ys,zs)

    plt.show()