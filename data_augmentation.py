#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:07:42 2022

@author: abelson
"""
import numpy as np

from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=(True),
    fill_mode='nearest')

image=load_img('cat.jpeg')
x=img_to_array(image)
x=x.reshape((1,)+x.shape)# this is numpy array [1,3,150,150]

# the .flow() command generates batches of randomly transformed images
#and saves it in the directory

i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir=('data_augmented'),
                          save_prefix='cat',save_format='jpeg'):
    i+=1
    if i>20:
        break