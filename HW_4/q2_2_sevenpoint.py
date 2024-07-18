import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):
    # (1) Normalize the input pts1 and pts2 scale paramter M.
    x1,y1 = pts1[:,0]/M,pts1[:,1]/M
    x1_prime,y1_prime = pts2[:,0]/M,pts2[:,1]/M
    # import pdb; pdb.set_trace()
    # (2) Setup the seven point algorithm's equation.
    A = np.zeros((7,9))
    T = np.diag([1/M,1/M,1])

    for i in range(0,7):
        A[i] = np.array([x1[i]*y1[i],x1[i]*y1_prime[i],x1[i],y1[i]*x1_prime[i],y1[i]*y1_prime[i],y1[i],x1_prime[i],y1_prime[i],1])
    # (3) Solve for the least square solution using SVD.
    u,s,vT = np.linalg.svd(A)

    # (4) Pick the last two column vectors of vT.T (the two null space solution f1 and f2)
    F1 = np.reshape((vT[-1,:]),(3,3))
    F2 = np.reshape((vT[-2,:]),(3,3))

    # (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        # det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        # Use np.polynomial.polynomial.polyroots to solve for the roots
    # det(F1+lambda*F2) = a3*lambda**3+a2*lambda**2+a1*lambda+a0 = 0
    
    #Find alpha such that Determinant(αF1 + (1 − α)F2) = 0; use symbolic variable sympy
    # a = sym.symbols('a')
    F3 = lambda a: a*F1+(1-a)*F2
    # f_det = sym.Matrix(F3)

    #Returns a cubic polynomial
    # F = f_det.det()

    poly = lambda x: np.linalg.det(x*F1+(1-x)*F2)
    coeff = extract_coefficients(poly,3)
    import pdb; pdb.set_trace()
    #After getting the coefficients need to plug them back into equation and solve for roots
    
    F_roots = np.polynomial.polynomial.polyroots(coeff)
    Y = np.roots(coeff)
    Farray = []

    
    #Now get the real parts
    for s in F_roots:
        if np.isreal(s):
            F = s*F1 + (1-s)*F2
            F = (np.dot(T.T,F)).dot(T)  
            Farray.append(F)
    # import pdb; pdb.set_trace()

    # raise NotImplementedError()
    return Farray

 
def extract_coefficients(p, degree):
    n = degree + 1
    sample_x = [ x for x in range(n) ]
    sample_y = [ p(x) for x in sample_x ]

    A = [ [ 0 for _ in range(n) ] for _ in range(n) ]
    for line in range(n):   
        for column in range(n):
            A[line][column] = sample_x[line] ** column
    c = np.linalg.solve(A, sample_y)
    return c[::-1]


if __name__ == "__main__":
        
    correspondence = np.load('code/data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('code/data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('code/data/im1.png')
    im2 = plt.imread('code/data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[2]

    np.savez('code/data/q2_2.npz', F, M, pts1, pts2)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)