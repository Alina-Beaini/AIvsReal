#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Create constants and import libraries
UNPROCESSED_DIRECTORY = "CNN_synth_testset"
PROCESSED_DIRECTORY = "processed_images"
CROP = False
import os
import numpy as np
from PIL import Image as im_lib
from random import randint


# In[2]:


# This function obtains unprocessed image paths and returns the list of those paths.
def obtain_image_paths(unprocessed_directory = UNPROCESSED_DIRECTORY):
    data_directory = os.listdir("./"+unprocessed_directory)
    list_of_image_paths = []
    for dirpath, dirname, filename in os.walk("./"+unprocessed_directory):
        if not dirname:
            for dirpath2, dirname2, filename2 in os.walk(dirpath):
                for filedir in filename2:
                    list_of_image_paths.append(dirpath+"/"+filedir)
    return list_of_image_paths


# In[3]:


# This function crops/compresses unprocessed images to 256x256 then saves them in a new directory.
keywords = ["biggan", "crn", "cyclegan","deepfake","gaugan","imle","progan","san","seeingdark","stargan", "stylegan2","stylegan",
           "whichfaceisreal"]
X_MIN_NEEDED = 256
Y_MIN_NEEDED = 256

def process_images(list_of_image_paths, keyword_list = keywords, x_processed = X_MIN_NEEDED, y_processed = Y_MIN_NEEDED,
                   processed_directory = PROCESSED_DIRECTORY, crop = CROP):
    if not os.path.exists("./"+processed_directory):
        os.mkdir(processed_directory)
    nReal = 0
    nFake = 0
    n = nReal + nFake
    m = 0
    keyword = ""
    N = len(list_of_image_paths)
    left, upper, right, lower = 0,0,0,0
    for image_path in list_of_image_paths: 
        image = im_lib.open(image_path)
        m += 1
        x, y = image.size
        if x >= x_processed and y >= y_processed:
            for kwd in keywords:
                if kwd in image_path:
                    keyword = kwd
                    break
            if "real" in image_path and "fake" not in image_path:
                nReal += 1
                processed_path = "./"+processed_directory+"/real_"+str(nReal)+".png"
            else:
                nFake += 1
                processed_path = "./"+processed_directory+"/fake_"+str(nFake)+"_"+keyword+".png"
            n = nReal + nFake
            if not crop:
                image = image.resize((x_processed,y_processed), im_lib.Resampling.LANCZOS)
            elif crop: 
                left, upper = randint(0, x-x_processed), randint(0, y-y_processed)
                right, lower = x_processed+left, y_processed+upper
                image = image.crop((left,upper,right,lower))
            if len(np.asarray(image).shape)==3: #This is to not save the grayscale images, which actually exist in our dataset..
                image.save(processed_path, quality = 100)
        print(str(m)+"/"+str(N), end="\r")


# In[4]:


#This here tests the above two functions.
#list_of_image_paths_test = obtain_image_paths("dataset_initial") #dataset_initial is the name of the folder.
#process_images(list_of_image_paths_test, processed_directory = "Test1_modular", crop = True)


# In[ ]:





# In[ ]:




