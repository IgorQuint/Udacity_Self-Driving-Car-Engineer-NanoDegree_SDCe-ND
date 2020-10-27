#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import glob
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[3]:


#Function to load image is it's within the IMG folder
def load_img(filename):
    filename = filename.strip()
    if filename.split('/')[0] == 'IMG':
        filename = '/opt/carnd_p3/data/{}'.format(filename)
    return plt.imread(filename)


# In[ ]:


# Function to build network
# I've based this on the Nvidia network mentioned in the theory sections. This 
def net():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0)-0.5))
    model.add(Conv2d(24, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(36, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(48, (3,3), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(64, (3,3), strides= (2,2), activation = 'relu'))
    model.add(Conv2d(64, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))

