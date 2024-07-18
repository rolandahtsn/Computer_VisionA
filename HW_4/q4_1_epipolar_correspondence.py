import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, y1], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x = np.array([x1,y1,1])
    ep_line = np.dot(F,x) #Epipolar line on image 1
    # ep_line_x2 = np.dot(F,x2)
    #Search window
    window = 11
    window_def = im1[y1-window//2:y1+window//2+1,x1-window//2:x1+window//2+1] #This is a fixed window for image 1

    y2_search_array = np.array(range(y1-50,y1+50))
    x2_search_array = np.round(-(ep_line[1]*y2_search_array+ep_line[2])/ep_line[0]).astype(int)#Form. epipolar line

    e_dist = np.inf
    x2 = None
    y2 = None

    #Apply Gaussian weight to window 1
    blurred_window_1 = gaussian_filter(window_def, sigma = 3)

    # import pdb; pdb.set_trace()
    #Search along this line to check nearby pixel intensity
    for i in range(0,len(x2_search_array)):
        k = (x2_search_array>= window//2) & (y2_search_array >= window//2) & (x2_search_array <im2.shape[1]-window//2) & (y2_search_array <im2.shape[0]-window//2)

        if k.all():
            x2_search_array = x2_search_array[k]
            y2_search_array = y2_search_array[k]
        else:
            print("Points not on line")

        #Second window, this is changing 
        shift = y2_search_array[i]
        shift_x = x2_search_array[i]
        
        window_def_2 = im2[shift-int(window//2):shift+int(window//2)+1,shift_x-window//2:shift_x+int(window//2)+1]
        blurred_window_2 = gaussian_filter(window_def_2, sigma = 3)
        # scipy.ndimage.gaussian_filter(im2, sigma)
        erro_d = np.sum(np.linalg.norm(blurred_window_1-blurred_window_2))
        if erro_d < e_dist:
            e_dist = erro_d
            x2,y2 = x2_search_array[i],y2_search_array[i]
    return x2,y2


if __name__ == "__main__":

    correspondence = np.load('code/data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('code/data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('code/data/im1.png')
    im2 = plt.imread('code/data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    np.savez('code/data/q4_1.npz', F,pts1,pts2)
    # epipolarMatchGUI(im1, im2, F) #To plot points

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)