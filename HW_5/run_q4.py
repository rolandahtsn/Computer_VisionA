import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('./images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('./images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    def getCenterYbox(bbox):
        y1 = bbox[0]
        y2 = bbox[2]
        center = (y1+y2)/2
        return center
    def getXCenterbox(bbox):
        x1 = bbox[1]
        x2 = bbox[3]
        center_x = (x1+x2)/2
        return center_x
    # find the rows using..RANSAC, counting, clustering, etc.
    #Heights
    box_height = []
    for i in bboxes:
        height = i[2]-i[0]
        box_height.append(height)
    avg_ = np.mean(box_height) 
    bboxes.sort(key=getCenterYbox)

    y = 0
    row = 0
    x_rows =[]
    _rows=[]
    sorted_rows = []

    #Get height of the first box; initial height
    height_y = getCenterYbox(bboxes[0])
    for i in range(0,len(bboxes)):
        box = bboxes[i]
        center_x = getXCenterbox(box)
        center_y = getCenterYbox(box)

        # y_1 = center_y-y
        # y_2_1 = bboxes[i][2]-bboxes[i][0]
        if center_y <height_y+avg_:
            if ((box[3]-box[1])>0 and (box[2]-box[0])>0):
                x_rows.append(box) 
                row += 1
            height_y = center_y
            _rows.append(row)
        else:
            height_y = center_y
            x_rows.sort(key=getCenterYbox) 
            sorted_rows.append(x_rows)
            x_rows = []
            x_rows.append(box)

    x_rows.sort(key=getXCenterbox)
    sorted_rows.append(x_rows)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    letter_tl = []
    for row in sorted_rows:
        # center_x = sorted_rows[i][2]-sorted_rows[i][0]
        # center_y = sorted_rows[i][3]-sorted_rows[i][1]
        track_data = []
        sorted_rows.sort(key = lambda p:p[1])
        for bbox in row:
            y1 = bbox[0]
            x1 = bbox[1]
            y2 = bbox[2]
            x2 = bbox[3]
            padding = np.ones((2,2),dtype=int)
            #Bounding boxes back into the image
            char = bw[y1:y2,x1:x2]
            char = np.pad(char,20*padding,'constant',constant_values=1)

            #Letters thicken or erode; test this
            char = skimage.morphology.binary_erosion(char)
            
            # char = skimage.morphology.binary_dilation(char)
            # char = skimage.morphology.binary_dilation(char)
            # char = skimage.morphology.binary_dilation(char)

            #Reshape into 32x32
            char = skimage.transform.resize(char,(32,32))

            #flatten and transpose
            char = char.T.reshape(-1)
            track_data.append(char)
            all_characters = np.array(track_data)
        letter_tl.append(track_data)
    
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    print("Image: "+img)
    row_text = ""
    for data in letter_tl:
        h_ = forward(data, params, 'layer1')
        probs = forward(h_, params, 'output', softmax)
        N = probs.shape[0]
        row_text = ""
        index = np.argmax(probs,axis=1)
        for prediction in index:
            row_text += letters[prediction]
        print(row_text)