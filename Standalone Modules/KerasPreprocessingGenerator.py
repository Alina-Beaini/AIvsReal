#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This code here is intended to create a custom keras layer.
# This layer takes as input the path to the image
# It outputs an array that can be given as an input to a convolutional neural network


# In[6]:


#Create constants and import libraries
X_DIM = 256
Y_DIM = 256
import os
import numpy as np
from PIL import Image as im_lib
from keras.layers import Layer
from keras import models
from keras import layers
from keras.utils import Sequence
from math import ceil
import time


# In[8]:


def categorical_output(image_path):
    keywords = ["biggan", "crn", "cyclegan","deepfake","gaugan","imle","progan","san","seeingdark","stargan","stylegan2", "stylegan",
           "whichfaceisreal"]
    category = [0] * (len(keywords) + 1)
    for kwd_index in range(len(keywords)):
        kwd = keywords[kwd_index]
        if kwd in image_path:
            category[kwd_index] = 1
            return category
    category[-1] = 1
    return category


# In[9]:


class ImageBatchSequence(Sequence):
    def __init__(self, x_set, batch_size, x_dim = X_DIM, y_dim = Y_DIM, filterFunction = None, output_function = None,
                normalization_factor = 256):
        self.x = x_set
        if output_function is None:
            y_set = [(1 if "fake" in image_path else 0) for image_path in x_set] #This makes the output to be 1 (AI-generated) if FAKE is in pathname.
            #                                                  # Otherwise, it makes the output 0 (not AI-generated)
        else:
            y_set = [output_function(image_path) for image_path in x_set]
        self.y = y_set
        self.batch_size = batch_size
        self.x_dim = x_dim
        self.y_dim = y_dim
    def __len__(self):
        return ceil(len(self.x)/self.batch_size)
    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low+self.batch_size, len(self.x))
        batch_x = self.x[low : high]
        batch_y = self.y[low : high]
        image_x_batch = []
        for image_path in batch_x:
            image = im_lib.open(image_path)
            if filterFunction is not None:
                image = filterFunction(image)
            image_array = np.asarray(image).reshape(self.x_dim,self.y_dim,3)
            image_x_batch.append(image_array/normalization_factor)
        return np.array(image_x_batch), np.array(batch_y)


# In[ ]:




