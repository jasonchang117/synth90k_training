#! /usr/bin/python3

import os
import h5py
import numpy as np
from PIL import Image

# download synth90k dataset
os.system("wget http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz")
os.system("tar -xvzf mjsynth.tar.gz")
data_path = "./mnt/ramdisk/max/90kDICT32px/"

# read training data
f = open(data_path + 'annotation_val.txt', 'r')
x_data = f.readlines()

data_size = len(x_data)
word_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
			'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
			'u', 'v', 'w', 'x', 'y', 'z',
			'A', 'B', 'C', 'D', 'E', 'F', 'G' ,'H', 'I', 'J',
			'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
			'U', 'V', 'W', 'X', 'Y', 'Z', ' ']

# create file
x = h5py.File('./x_val.h5', 'w')
y = h5py.File('./y_val_char.h5', 'w')

x_train = x.create_dataset('x_val', (data_size, 32, 100, 3), dtype = 'float16', chunks = (10000, 32, 100, 3))
y_train = y.create_dataset('y_val', (data_size, 18, 64), dtype = 'int8', chunks = (10000, 18, 64))

# build dataset
def build_data(arg):
	try:
		img = Image.open(data_path + arg[1].split(' ')[0], 'r')
		img = img.resize((100, 32), Image.BICUBIC)
		img = np.asarray(img).astype('float16')/255
		x_train[arg[0], :, :, :] = img

		word = []
		temp = list(arg[1].split(' ')[0].split('_')[1])
		while(len(temp) < 18):
			temp = [' '] + temp
		for k in range(18):
			one_hot = np.zeros((1, 64))
			one_hot[0, word_list.index(temp[k])] = 1
			y_train[arg[0], k, :] = one_hot

	except OSError:
		pass
	except TypeError:
		pass
	except IndexError:
		pass
	
def data_gen():
	for i in range(data_size):
		build_data([i, x_data[i]])
		print('\r%.2f %%  %d' % (i/data_size*100, i), end = '')

data_gen()
print()
os.system("./propressing_train.py")
os.system("mkdir models")
