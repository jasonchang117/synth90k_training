#! /usr/bin/python3

import h5py
import random
import numpy as np
import tensorflow as tf
import multiprocessing
import keras.backend.tensorflow_backend as KTF
from tools import threadsafe_generator
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint

def get_session():
	gpu_options = tf.GPUOptions(allow_growth = True)
	return tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
# KTF.set_session(get_session())


# load and compile model
with open('./models/Reco_archi.json', 'r') as f:
	reco_model = model_from_json(f.read())
print(reco_model.summary())
reco_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# read data
x_t = h5py.File('./x_train.h5', 'r')
y_t = h5py.File('./y_train_char.h5', 'r')
x_v = h5py.File('./x_val.h5', 'r')
y_v = h5py.File('./y_val_char.h5', 'r')
bs = 500	# batch_size


# total 7224612
@threadsafe_generator
def batch_generator():
	i = 0
	while True:
		i += 1
		yield x_t.get('x_train')[(i%14400)*bs:(i%14400)*bs+bs], y_t.get('y_train')[(i%14400)*bs:(i%14400)*bs+bs]

@threadsafe_generator
def val_data_generator():
	i = 0
	while True:
		i += 1
		yield (x_v.get('x_val')[(i%1600)*bs:(i%1600)*bs+bs], y_v.get('y_val')[(i%1600)*bs:(i%1600)*bs+bs])


earlystopping = EarlyStopping(monitor = 'val_acc', patience = 2)
checkpoint = ModelCheckpoint('./models/weights.h5', monitor = 'val_loss', save_best_only = True, save_weights_only = True, period = 1)
hist = reco_model.fit_generator(batch_generator(), 
								steps_per_epoch = 7224000//bs, 
								epochs = 8, 
								validation_data = val_data_generator(), 
								validation_steps = 800000//bs, 
								workers = 6, 
								max_queue_size = 10,
								callbacks = [checkpoint])
