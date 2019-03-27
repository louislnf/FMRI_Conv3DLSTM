import keras
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from BatchGenerator import *



cnn = keras.Sequential()
cnn.add(keras.layers.Conv3D(20, (4, 4, 4), input_shape=(64, 64, 24, 1)))
cnn.add(keras.layers.MaxPooling3D(pool_size=(2,2,2)))
cnn.add(keras.layers.Conv3D(20, (2, 2, 2)))
cnn.add(keras.layers.MaxPooling3D(pool_size=(2,2,2)))
cnn.add(keras.layers.Flatten())

cnn.compile(loss='mean_squared_error', optimizer='sgd')

model = keras.Sequential()
model.add(keras.layers.TimeDistributed(cnn, input_shape=(15,64,64,24,1)))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(4, activation="softmax"))

generator = BatchGenerator("/home/benoit/Documents/Code/Projets/OpenFMRIDS")

print(np.shape(generator[0][0][0]))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit_generator(generator=generator)
