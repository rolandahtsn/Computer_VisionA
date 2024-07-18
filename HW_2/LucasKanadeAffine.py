import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image (It)
    :param It1: Current image (It+1)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    #Current image
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    x1x2 = np.arange(0,(It.shape[1]-1)) #Instead of rectangle we are now looking at the entire image
    y1y2 = np.arange(0,(It.shape[0]-1))
    
    
    splineIt1 = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1) # Rectangular Spline for current   
    splineIt = RectBivariateSpline(np.arange(0,It.shape[0]),np.arange(0,It.shape[1]),It)
    
    
    x1_x2,y1_y2 = np.meshgrid(x1x2, y1y2)
  
    # put your implementation here
    p = M.flatten()
    
    deltap = [[1], [0], [0], [0], [1], [0]]
    for i in range(int(num_iters)):
        
        
        xp = x1_x2*(M[0,0])+y1_y2*M[0,1]+M[0,2]
        yp = x1_x2*(M[1,0])+y1_y2*(M[1,1])+M[1,2]
        
        M = np.array([[(1+p[0]),p[1],p[2]],[p[3],(1+p[4]),p[5]],[0,0,1]])
        
        #We only need to consider the common points between the current image and the warped current image

        x_comparison = np.logical_and(0<xp,xp<It.shape[1])
        y_comparison = np.logical_and(0<yp,yp<It.shape[0])
        valid_pts = np.logical_and(x_comparison,y_comparison)
    
        # Check deltap against threshold
        
        xp,yp = xp[valid_pts],yp[valid_pts]
        
        #y_p,x_p = np.meshgrid(yp,xp)

        #1. Warp I with (x;p)
        
        # warped_I = affine_transform(It1,M)
        warped_I = splineIt1.ev(yp, xp)
        #2. Compute error
        error =(It[valid_pts].flatten()-warped_I.flatten())
        
        #3. warp the gradient
        grad_x = splineIt1.ev(yp,xp, dx=0,dy=1)
        grad_y = splineIt1.ev(yp,xp, dx=1,dy=0)

        #4.Evaluate the jacobian at (x;p); #What is x and y in the jacobian? - pixel values
        A=np.zeros((It.shape[0]*It.shape[1],6)) #Nx6, with N being number of pixels
        
        #for y in range(It.shape[0]):
           # for x in range(It.shape[1]):
            #    jacobian = np.array([[x, 0, y, 0, 1, 0],[0, x, 0, y, 0, 1]])
                #5. Compute steepest_descent
              #  steepest_descent = np.dot(grad_x_y,jacobian)
               # sd_save[x*y] = steepest_descent[x*y]  
                #print(x)

        A = np.vstack((grad_x*xp,grad_x*yp,grad_x,grad_y*xp,grad_y*yp,grad_y)).T
        
        # import pdb; pdb.set_trace()
        #A = np.reshape(A,(It.shape[1]*It.shape[0],6))
        
        
        #6. Compute Hessian Matrix
        #6a. evaluate steepest_descent.T *steepest_descent
        hessian = np.dot(A.T,A)

        #7. Compute the sum
        x = np.dot(A.T,error)
        
        #8. Compute deltap
        hessian_inv = np.linalg.inv(hessian)
        
        deltap = np.dot(hessian_inv,x)

        p+=deltap

        #9. Update M
        deltaM = np.array([[p[0],p[1],p[2]],[p[3],p[4],p[5]],[0,0,0]])
        
        # print(np.linalg.norm(deltap),threshold)
        # threshold = .75
        if(np.sum(np.square(deltap)) < threshold):
            break  

        M += deltaM
    ################### TODO Implement Lucas Kanade Affine ###################
    
    return M
