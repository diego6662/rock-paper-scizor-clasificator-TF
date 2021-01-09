#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:19:19 2020

@author: diego
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import RMSprop

train_dir = '/home/diego/universidad/coursera/tensorflow introduction/rock-papper-scizor/train'
test_dir = '/home/diego/universidad/coursera/tensorflow introduction/rock-papper-scizor/test'

#creating a model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dense(3,activation='softmax'))
#compile a model

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(0.001),metrics=['acc'])

#preprocesin images

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (300,300),
                                                    batch_size = 20,
                                                    class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size = (300,300),
                                                    batch_size = 20,
                                                    class_mode = 'categorical')
model.fit(train_generator,
          steps_per_epoch=126, # step*bath = number of image
          epochs = 2,
          validation_data=test_generator,
          validation_steps=18,#step* bacth = number of image
          verbose=1)