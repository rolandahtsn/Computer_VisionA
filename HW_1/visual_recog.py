# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:14:04 2022

@author: rhutson
"""
  
import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    # print(K)
    
    # np.histogram represents rectangles of the same horizontal size corresponding to class intervals called bins
    # Variable height corresponds to frequency
    
    size = 11 #Array is 1 size bigger than K as histogram size is returning K-1 
    histogram = np.histogram(wordmap,range(0,K+1))[0] #This returns two arrays so we want the first
    
    # normalization
    
    row_sum = histogram.sum(axis=0) 
    
    new_matrix = []
    for i in range(0,len(histogram)):
        new_matrix.append(histogram[i]/row_sum)
        # print(histogram[i])
        # print(row_sum)
    histogram = new_matrix
    
    if np.isfinite(histogram).all() == True:
        np.nan_to_num(histogram, copy = True, nan = 0.0, posinf = None, neginf = None)
    else:
        pass

    return histogram

def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    # divide the image into a small number of cells, and concatenate the histogram of each
# of these cells to the histogram of the original image, with a suitable weight
    K = opts.K #K is the visual words
    L = opts.L #There are L+ 1 Layers
    
    # hist_all = np.ndarray(shape = ((K*4**(L+1)-1)/3))
    # Computation of histogram for the finest layers

    first_histogram = np.array([])
    weighted_gram = []

    hist_all = []
    
    # #First we need to make patchs of the image
    # #Computing first layer
    
    for i in range(0,2**L):
        for s in range(0,2**L): #If I only seperate based on columns then it returns entire column and not a patch
            patch_one_height = int(wordmap.shape[0]/(2**L))
            patch_two_width = int(wordmap.shape[1]/(2**L))
            first_histogram = np.append(first_histogram,
                                        get_feature_from_wordmap(opts,wordmap[patch_two_width*s:((s+1)*patch_two_width),i*patch_one_height:(i+1)*patch_one_height]))
    row_sum = first_histogram.sum(axis=0) 
    
    # Need to look at first_histogram[0,0];[0,1];[1,0];[1,1]
    
    
    # #Histogram is now in array of first_histogram
        
    #Normalizing
    
    new_matrix = []
    for i in range(0,len(first_histogram)):
        new_matrix.append(first_histogram[i]/row_sum)
    first_histogram = new_matrix
    
    #Weighting
    #Set the weight of 0 and 1 to 2^(-L); others 2^(l-L-1)
    for l in range(L): #l is layer number
        if l == 0 or l == 1:
            weight = 2**(-L)
        if l != 0 or l != 1:
            weight = 2**(l-L-1)
        for i in range(0,len(first_histogram)):
            weighted_gram.append(first_histogram[i]*weight)
    first_histogram = weighted_gram
    hist_all = np.append(first_histogram,hist_all) #Save first value
     
    first_histogram = np.reshape(first_histogram, [2,2,10]) #reshape into array to get [i,j]

    # print(first_histogram[0,0])
    
    #Remaining layers; now we are going through top and bottom of patch
    new_weight = [] 
    next_histogram = []
    for k in range(0,L): # To compute next layers
        c = np.array([1,1,K])
        for i in range(0,1):
            for j in range(0,1):
                row_z_c_z = first_histogram[i*(2**L),j*(2**L)]
                row_z_c_t = first_histogram[i*(2**L),j*(2**L)+1]
                row_t_c_z = first_histogram[i*(2**L)+1,j*(2**L)]
                row_t_c_t = first_histogram[i*(2**L)+1,j*(2**L)+1] 
                c = row_z_c_z+row_z_c_t+row_t_c_z+row_t_c_t
                # y = np.concatenate((row_z_c_z+row_z_c_t+row_t_c_z,row_t_c_t))
                next_histogram.append(c) #Next patch to compute
                
                #Normalize again
                new_matrix = []
                for i in range(0,len(next_histogram)):
                    new_matrix.append(next_histogram[i]/row_sum)
                next_histogram = new_matrix
                
                #     #Weight again
                for l in range(0,L): #l is layer number
                    if l == 0 or l == 1:
                        weight = 2**(-L)
                    if l != 0 or l != 1:
                        weight = 2**(l-L-1)
                    for i in range(0,len(next_histogram)):
                        new_weight.append(next_histogram[i]*weight)
                next_histogram = new_weight
        first_histogram = np.reshape(next_histogram,[1,1,K])
        hist_all = np.append(next_histogram,hist_all)
                
        # first_histogram = 
        # final_hist = next_histogram #Compute next layer
        # #After weighing again
        # hist_all = np.append(final_hist)
   
    #     #Weight again
    #     for l in range(0,L,-1): #l is layer number
    #         if l == 0 or l == 1:
    #             weight = 2**(-L)
    #         if l != 0 or l != 1:
    #             weight = 2**(l-L-1)
    #         for i in range(0,len(next_histogram)):
    #             new_weight.append(next_histogram[i]*weight)
    #     next_histogram = new_weight
    #     next_histogram_flat = next_histogram.flatten()
    #     hist_all = np.append(next_histogram_flat,hist_all)
    # print(hist_all.shape)
    return hist_all
def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    #Loads an image
    load_img = Image.open(img_path)
    load_img = np.array(load_img).astype(np.float32) / 255
    #Extracts word from the wordmap in the image
    wordmap = visual_words.get_visual_words(opts, load_img, dictionary)
    # Computes the SPM 
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    #Returns the computed feature
    return feature
    
def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K
    L = opts.L
    #Loading the training images
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))
    M = int((K*(4**(L+1)-1))/3)
    #Need to get dictionary (already loaded)
    #Get features (matrix with all of the histograms of training images)
    features = np.zeros([len(train_files),M])
    # features = np.zeros((len(train_files),M))
    # Labels with the labels of each training images features[i] = label_labels[i]
    argument_parameters = []
    
    for i in enumerate(train_files):
        #print(type(train_files)) - list
        img_path = str(i[1:])
        img_path = img_path.replace('(','')
        img_path = img_path.replace(')','') #Adding directory to path
        img_path = img_path.replace("'",'')
        img_path = img_path.replace(",",'')
        img_path = join(data_dir, img_path)
        argument_parameters.append((opts,img_path,dictionary))
        
    with multiprocessing.Pool(processes = 4) as pool:
        # x = pool.map(get_image_feature, zip((opts,img_paths,dictionary)))
        x = pool.starmap(get_image_feature, argument_parameters)
        # x.close()
        # x.join()
    # for i in range(0,len(x)):
    #     features = np.append(features[i][0])
   
    for i in range(0,len(x)):
        features.append(x[i][0])
        # for j in range(0,len(x)):
            
    SPM_layer_num = opts.L
    # SPM_layer_num?
    
    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    

def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    
    # q = Computes histogram intersection similarity and each training sample as a vector of length T
    #returns 1 - q as a distance
    q = np.minimum(word_hist,histograms) #Compute distance between word_hist and histgram arrays
    sim = 1 - np.sum(q,1) #Add together for images; avoid for loops
    
    return sim


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    ''''Compute the test image's distance to every image
    in the training set;
    2.Return the label of the closest training image
    3. Quantify accuracy with confusion matrix C(ij)
    '''
    C = np.zeros((8,8))
    img_paths =[]
    # Confusion matrix: C = 8x8
    for i in range(0,len(test_files)):
        features = get_image_feature(opts, img_paths[i],dictionary) #Get features
        similarities = similarity_to_set(features,trained_system['features']) #Get distances
        img_paths.append(join(data_dir,str(test_files[i]))) #Get image path
        
        # print(np.argmin(similarities))
        
        estimated = trained_system['labels']
        pro = estimated[np.argmin(similarities)]
    
        real_values = test_labels[i]
        
        C[real_values,pro] = C[real_values,pro] +1
    
    accuracy = np.trace(C)/np.sum(C)
    return C,accuracy
   
# def compute_IDF(opts, n_worker=1):
#     # YOUR CODE HERE
#     pass

# def evaluate_recognition_System_IDF(opts, n_worker=1):
#     # YOUR CODE HERE
#     pass