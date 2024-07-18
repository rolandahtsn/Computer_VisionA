import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    #Precomputation
    #1. Evalutate the gradient of template image

    x1x2 = np.arange(0,(It.shape[1]-1)) #Instead of rectangle we are now looking at the entire image
    y1y2 = np.arange(0,(It.shape[0]-1))
    
    
    splineIt1 = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1) # Rectangular Spline for current   
    splineIt = RectBivariateSpline(np.arange(0,It.shape[0]),np.arange(0,It.shape[1]),It)
    
    
    x1_x2,y1_y2 = np.meshgrid(x1x2, y1y2)

    p = M0.flatten()
    
    #pre-compute gradient
    y1_y2_flattened = y1_y2.flatten()
    x1_x2_flattened = x1_x2.flatten()
    
    #Pre-compute gradient of Template
    grad_x_T = splineIt.ev(y1_y2,x1_x2,dx=0,dy=1).flatten()
    grad_y_T = splineIt.ev(y1_y2,x1_x2, dx=1,dy =0).flatten()
    
    #Pre-Compute A and evaluate jacobian at (x;0)
    A=np.zeros((It.shape[0]*It.shape[1],6))
    A = np.vstack((grad_x_T*x1_x2_flattened,grad_x_T*y1_y2_flattened,grad_x_T,grad_y_T*x1_x2_flattened,grad_y_T*y1_y2_flattened,grad_y_T)).T
    #Compute the Hessian Matrix
    hessian = np.dot(A.T,A)

    for i in range(int(num_iters)):
        M0 = np.array([[(1+p[0]),p[1],p[2]],[p[3],(1+p[4]),p[5]],[0,0,1]])
        
        # xp = x1_x2*(M0[0,0])+y1_y2*M0[0,1]+M0[0,2]
        # yp = x1_x2*(M0[1,0])+y1_y2*(M0[1,1])+M0[1,2]
        
        xp = x1_x2*p[0]+y1_y2*p[1]+p[2]
        yp = x1_x2*p[3]+y1_y2*p[4]+p[5]
        
        x_comparison = np.logical_and(0<xp,xp<It1.shape[1])
        y_comparison = np.logical_and(0<yp,yp<It1.shape[0])
        
        valid_pts = np.logical_and(x_comparison,y_comparison)
        xp,yp = xp[valid_pts], yp[valid_pts]
    
        warped_I = splineIt1.ev(yp, xp)
        # t_x = splineIt.ev(yp,xp)
        # warped_I = affine_transform(It1,M0)
        # t_x = affine_transform(It,M0)
        error = (It[valid_pts].flatten()-warped_I.flatten())
        
        valid_pts_shape = np.reshape(valid_pts,(It.shape[0]*It.shape[1],))
        #Calculate the sum
        x_incoA = np.dot(A[valid_pts_shape].T,error)

        #Compute deltap
        #8. Compute deltap
        hessian_inv = np.linalg.inv(hessian)
        # counter+=1
        # print(hessian_inv)
        # print(counter, "Hessian Loops")
        deltap = np.dot(hessian_inv,x_incoA)
        
        p+=deltap

        #9. Update M
        deltaM = np.array([[p[0],p[1],p[2]],[p[3],p[4],p[5]],[0,0,0]])
        
        # print(np.linalg.norm(deltap),threshold)
        # threshold = .75
        if(np.sum(np.square(deltap)) < threshold):
            break  

        M0 += deltaM
        
    return M0
