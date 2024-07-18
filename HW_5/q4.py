import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
# import cv2

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    #Draw rectangles on the image to show the bounding boxes
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    bboxes = []
    bw_img = None
    area_box = 0
    area_box_list = []

    #denoise image
    img_noise = skimage.restoration.denoise_bilateral(image, multichannel=True)
    # convert image to grayscale
    img_gray = skimage.color.rgb2gray(img_noise)
    # get a binary image first
    threshold = skimage.filters.threshold_otsu(img_gray)

    #Process the binary image
    mask = img_gray>threshold

    #Use morphology: # Thicken letters; skimage.morphology dilation step 
    mask = skimage.morphology.erosion(mask)
    mask = skimage.morphology.erosion(mask)
    mask = skimage.morphology.dilation(mask)

    bw_img = mask
    # mask = skimage.morphology.remove_small_objects(mask, 50)
    # mask = skimage.morphology.remove_small_holes(mask, 50)

    # skimage to find boundaries of the letters; skimage.measure.regionprops
    labels = skimage.measure.label(bw_img, background=1, connectivity=2)
    bb = skimage.measure.regionprops(labels,bw_img)

    # check if boundary box is larger than average box size
    # skip the small boxes
    # determine threshold (100-200) for comparing the top edge of the bounding boxes to the next bbox
    # compare against the letter next to it
    
    for box in bb:
        area_box = area_box+ box.area
        area_box_list.append(area_box)
    avg_box_size = area_box/len(bb)
    for i in bb:
        if i.area >avg_box_size/4:
            bboxes.append(i.bbox)
            # print("here")
    # bw_img = (~bw_img).astype(np.float)
    return bboxes, bw_img