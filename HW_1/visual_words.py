# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:28:19 2022

@author: rhutson
"""

import os
import multiprocessing
from os.path import join, isfile
import opts
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans

import tempfile
from multiprocessing import Process, Pool, Queue
import time
import sklearn

q = Queue()

def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    # filter_scales = [1,2,4,8]
    
    #Gray-scale images output as 3F channel image
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    # channels = img.shape[2]
    
    if len(dimensions) < 3:
        # What does it mean to duplicate into three channels?
        gray_img = np.stack((img,)*3,axis =2)
        img = gray_img
    else:
        pass
    
    img = skimage.color.rgb2lab(img)
    
    #How am I supposed to add to the filter responses if it is not an array?
    # I don't understand what filter responses should initially be?
    # filter_responses = np.zeros(shape = (height,width,3*4))
    filter_scales = [1,2,4,8]
    
    responses = []
    list_filter = ["Gaussian", "Laplacian", "Dogx", "Dogy"]
        
    for i in filter_scales:
        #Gaussian Filter
        #How to implement zero padding in the image?
        for filter in list_filter:
            if filter == "Gaussian":
                for rgb in range(0,3): 
                    gaussian = scipy.ndimage.gaussian_filter(img[:,:,rgb], i, mode ="reflect")
                    responses.append(gaussian)
    
        # for rgb in range(0,3):    
            #Laplacian filter
            if filter == "Laplacian":
                for rgb in range(0,3): 
                    laplacian = scipy.ndimage.gaussian_laplace(img[:,:,rgb], i, mode ="reflect")
                    responses.append(laplacian)
        # for rgb in range(0,3):    
            #X derivative Gaussian
            if filter == "Dogx":
                for rgb in range(0,3): 
                    gaussian_x = scipy.ndimage.gaussian_filter(img[:,:,rgb], i, order=[0,1], mode = "reflect")
                    responses.append(gaussian_x)
            # for rgb in range(0,3):   
            #y derivative gaussian
            if filter == "Dogy":
                for rgb in range(0,3): 
                    gaussian_y = scipy.ndimage.gaussian_filter(img[:,:,rgb], i, order=[1,0], mode = "reflect")
                    responses.append(gaussian_y)
    filter_responses = np.dstack(responses)

    return filter_responses



def compute_dictionary_one_image(img_path):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    
    #Read one image
    img_path = '../data/' + str(img_path)
    img_path = img_path.replace('[','')
    img_path = img_path.replace(']','') #Adding directory to path
    img_path = img_path.replace("'",'')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    height = img.shape[0]
    width = img.shape[1]
    
    if len(img.shape) < 3:
        # What does it mean to duplicate into three channels?
        gray_img = np.stack((img,)*3, axis = 2)
        img = gray_img
    #Extracts the responses
    filter_responses = extract_filter_responses(opts, img)
    
    #Use alphaT to select the filter responses; opts.alpha is 25 (int)
    #Random pixels(How am I supposed to get the values at the alpha random pixels?)
    random_val2 = np.random.randint(0, width,25)
    random_val = np.random.randint(0, height,25)

    #Collect matrix of responses (size alphaTx3F)
    # matrix_of_responses = np.empty((opts.alpha*len(train_files),3*4))
    holder = filter_responses[random_val, random_val2,:]
    
    #Save to a temporary file
    # temp_file = tempfile.TemporaryFile()
    # temp_file.write(holder)
    
    # x = []
    # x.append(holder)
    
    return holder

def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
#Load the training data
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    
    
    #Iterate through paths to the image files to read the images
    img_path = []
    x = []
    
    pool = Pool(processes = (n_worker))
  
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    
    for i in range (0,1177):
        img_path.append(train_files[i])
    with multiprocessing.Pool(processes = 4) as pool:
        x = pool.map(compute_dictionary_one_image, img_path)
        
        #If I do the process this way then this means that there is a lot of processes trying to run in parallel
        # p = multiprocessing.Process(target = compute_dictionary_one_image, args=(img_path, return_dict))
        # x.append(p)
        # pool.apply_async(compute_dictionary_one_image, args=(img_path, return_dict))
        # x.append(p)   
        # p.start()
        
    # pool.close()
    # pool.join()
    # x.append(q.get())    
    # print(x)    
    # Load the temporary files back
    
    # matrix_of_responses = np.empty((opts.alpha*len(train_files),3*4))
    matrix_of_responses = np.concatenate(x,axis=0)
    
    #call k-means
    kmeans = sklearn.cluster.KMeans(n_clusters = K).fit(matrix_of_responses)
    d = kmeans.cluster_centers_

    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), d)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """
    height = img.shape[0]
    width = img.shape[1]
    #Each pixel in wordmap is assigned the closest visual response
    # at the respective pixel in img
    filter_responses = extract_filter_responses(opts,img) 
    filter_responses = np.reshape(filter_responses,((height*width),48)) # Resahpe the filter responses format: (a - array, (new shape dimensions))
    
    # import pdb; pdb.set_trace()
    
    
    wordmap = np.zeros(img.shape[0:2])
    
    
    # responses_size = np.array([])
    # responses_size = np.reshape(responses_size,[filter_responses.shape[0],filter_responses.shape[1]])
    
    # import pdb; pdb.set_trace()
    smallest_distance_comp = scipy.spatial.distance.cdist(filter_responses,dictionary, metric = "euclidean")
    wordmap = np.argmin(smallest_distance_comp, 1)
    wordmap = np.reshape(wordmap,(height,width))
    
    
    return wordmap
    
