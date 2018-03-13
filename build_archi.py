#! /usr/bin/python3

import json
import keras
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, GRU, RepeatVector
from keras.layers import Dropout, Flatten, Reshape, Embedding, LSTM
from keras.layers import TimeDistributed
from keras.models import Model
from keras.optimizers import SGD

input_img = Input((32, 100, 3))

# CNN
x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# FC_layer
x = Dense(1024, activation = 'relu')(x)
x = RepeatVector(18)(x)

# RNN
x = LSTM(512, return_sequences = True)(x)
x = LSTM(512, return_sequences = True)(x)

output_word = TimeDistributed(Dense(64, activation = 'softmax'), input_shape = (18, 64))(x)
recognizer= Model(input_img, output_word)

with open('./models/output_char/Reco_archi.json', 'w') as f:
	f.write(recognizer.to_json())

print(recognizer.summary())
