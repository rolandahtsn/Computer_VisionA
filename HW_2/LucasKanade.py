import numpy as np
import scipy
from scipy.interpolate import RectBivariateSpline
import cv2


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # set up the threshold
    # Compute the optimal local motion from frame It to from It+1 that minimizes Equation 3
    # It is the image frame; rect is the 4x1 vector rep a rectangle with all pixel conditions in N
    
    #where to get x and y values? From the frame in It!
    
    #Current image
    x1 = rect[0]
    x2 = rect[2]
    y1 = rect[1] 
    y2 = rect[3]

    #Top left, Bottom Right
    x1x2 = np.arange(rect[0],rect[2]+1)
    y1y2 = np.arange(rect[1],rect[3]+1)
    
    y1_y2,x1_x2 = np.meshgrid(y1y2,x1x2)
    
    
    # x_warp,y_warp = np.meshgrid(x2,y2)
    
    # Spline allows us to extract the gradient at any point/pixel in the image
    
    #Compute the spline for the image
    splineIt1 = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1) # Rectangular Spline for current   
    splineIt = RectBivariateSpline(np.arange(0,It.shape[0]),np.arange(0,It.shape[1]),It)
    # Compute delta_p in order to shift the bounding box (Reiew lecture slides)
    p = p0
    t_x = splineIt.ev(y1_y2,x1_x2) #template image for error
    jacobian = [[1,0],[0,1]]
    
    for i in range(int(num_iters)):
        #Warping I to compute I(W(x;p)) requires interpolating the image I at the sub-pixel locations
        #W(x:p)
        xp = np.arange(x1,x2)+p[0]
        yp = np.arange(y1,y2)+p[1]
        y_p,x_p = np.meshgrid(yp,xp)
        
        #1. Warp I with (x;p)
        warped_I_rect = splineIt1.ev(y_p,x_p)
        
        #2. Compute the error
        error = (t_x-warped_I_rect).flatten()
        
        #3. warp the gradient
        grad_x = splineIt1.ev(y_p,x_p, dx=0,dy=1)
        grad_y = splineIt1.ev(y_p,x_p, dx=1,dy=0)
        
        #Substep: Concatenate gradients
        grad_x_y = np.stack((grad_x,grad_y),axis=2)
        grad_x_y = np.reshape(grad_x_y,(-1,2)) #Nx2
        
        #4. Jacobian in translation is identity matrix so no need to evaluate at W(x;p)
        
        #5. Compute steepest descent images
        steepest_descent = np.dot(grad_x_y,jacobian) #Nx2
        #6. Compute Hessian Matrix
        #6a. evaluate steepest_descent.T *steepest_descent
        hessian = np.dot(steepest_descent.T,steepest_descent)
        
        #7. Compute the sum
        x = np.dot(steepest_descent.T,error)
        
        #8. Compute deltap
        hessian_inv = np.linalg.inv(hessian)
        
        deltap = np.dot(hessian_inv,x)
        
        p += deltap
        
        if(np.sum(np.square(deltap)) < threshold):
            break
    
    return p

