import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF,_singularize

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''

def eightpoint(pts1, pts2, M):
    # Get the diagonal matrix T
    T = np.diag([1/M,1/M,1])
    #Now normalize the points using T Xnorm = Tx
    x1,y1 = pts1[:,0]*T[0,0],pts1[:,1]*T[1,1]
    x1_prime,y1_prime = pts2[:,0]*T[0,0],pts2[:,1]*T[1,1]


    A = np.zeros((pts1.shape[0],9))
    for i in range(0,pts1.shape[0]):
        A[i] = np.array([x1[i]*x1_prime[i],x1[i]*y1_prime[i],x1[i],y1[i]*x1_prime[i],y1[i]*y1_prime[i],y1[i],x1_prime[i],y1_prime[i],1])
        # print(x1[i])

    # (3) Solve for the least square solution using SVD
    # u,s,v = np.linalg.svd(np.dot(A.T,A))
    # print(A)
    u,s,v = np.linalg.svd(A)

    F = np.reshape((v[-1:,]),(3,3))

    # (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    F = _singularize(F)

    # (5) Use the function `refineF` (provided) to refine the computed fundamental matrix.)
    # import pdb; pdb.set_trace()
    F = refineF(F, pts1*T[0,0], pts2*T[1,1])
    

    # (6) Unscale the fundamental matrix Funnorm = T.T*F*T
    F = np.dot(np.dot(T.T,F),T)
    return F


if __name__ == "__main__":
        
    correspondence = np.load('code/data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('code/data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('code/data/im1.png')
    im2 = plt.imread('code/data/im2.png')
    M=np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print(F)

    np.savez('code/data/q2_1.npz', F=F,M=M)

    # Q2.1
    displayEpipolarF(im1, im2, F)


    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)