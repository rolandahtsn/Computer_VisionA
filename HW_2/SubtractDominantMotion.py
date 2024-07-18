from cv2 import threshold
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    # t = tolerance #what is the tolerance for?
    # #Binary_erosion (uses a structuring element for shrinking the shapes in an image) and dilation - 
    # # (a structuring element used for expanding the shapes in an image
    # # The binary dilation of an image by a structuring element is the locus of the points covered by
    # # the structuring element, when its center lies within the non-zero points of the image.)

    #M = np.linalg.inv(LucasKanadeAffine(image1, image2, threshold, num_iters))
    M = np.linalg.inv(InverseCompositionAffine(image1, image2, threshold, num_iters))
    
    # #Warp the image using M
    warp_image1 = affine_transform(image1,M)

    array_structure = np.array(([0,1,0],[1,1,1],[0,1,0]))
    
    #Getting the difference
    abs_diff = np.abs(image2 - warp_image1)
    mask = abs_diff > tolerance
    
    #Erosion and dilation
    
    mask = binary_erosion(mask, array_structure)
    mask = binary_dilation(mask, array_structure)
    
    mask = abs_diff > tolerance

    return mask.astype(bool)
