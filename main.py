#import keras
import nibabel
import numpy as np
import re
import os
import keras

from BatchGenerator import *
"""
cnn = keras.Sequential()
cnn.add(keras.layers.Conv3D(20, (4, 4, 4), input_shape=(64, 64, 24, 1)))
cnn.add(keras.layers.MaxPooling3D(pool_size=(2,2,2)))
cnn.add(keras.layers.Conv3D(20, (2, 2, 2)))
cnn.add(keras.layers.MaxPooling3D(pool_size=(2,2,2)))
cnn.add(keras.layers.Flatten())

cnn.compile(loss='mean_squared_error', optimizer='sgd')

model = keras.Sequential()
model.add(keras.layers.TimeDistributed(cnn, input_shape=(None, 192,64,64,24,1)))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(4, activation="softmax"))
"""
generator = BatchGenerator("/Users/louislenief/Desktop/Projet IRM/ds000108-00002")
"""
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.fit_generator(generator=generator)
"""