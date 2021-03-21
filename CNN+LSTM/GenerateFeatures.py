import keras
import tensorflow as tf
from keras.models import Model
import math
import matplotlib.pyplot as plt
import skvideo.io
import PIL
import numbers
import random
from tqdm import tqdm
import numpy as np
import csv
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('include50')
import os
import pandas as pd
from io import StringIO
from io import BytesIO
from tensorflow.python.lib.io import file_io
import cv2
import pandas as pd
import os.path
# from skimage import segmentation, measure

model = tf.keras.applications.MobileNetV2()
model2 = Model(model.input, model.layers[-2].output)

blob = bucket.get_blob('INCLUDE_TRAIN.txt')
blob = blob.download_as_string()
blob = blob.decode('utf-8')
blob = StringIO(blob)

blob2 = bucket.get_blob('INCLUDE_TEST.txt')
blob2 = blob2.download_as_string()
blob2 = blob2.decode('utf-8')
blob2 = StringIO(blob2)

names = csv.reader(blob)
filenames = []
for name in names:
	filenames.append(name[0])
filenames = [x[0:-3] + 'csv' for x in filenames]

names = csv.reader(blob2)
test_filenames = []
for name in names:
	test_filenames.append(name[0])
test_filenames = [x[0:-3] + 'csv' for x in test_filenames]

all_paths = filenames + test_filenames


#this function takes the path of sparse video and generates MobileNetV2 features for each frame of the video

def make_cnn(path):
	videoat = path[0:-3] + 'mp4'
	videoat = os.path.join('SPARSE', videoat)
	cap = cv2.VideoCapture(videoat)#path to video
	if cap.isOpened():
		hasFrame, frame = cap.read()
	else:
		hasFrame = False
	train_array = np.empty((0,1280))
	while hasFrame:
		frame = cv2.resize(frame, (224, 224))
		img_array_expanded_dims = np.expand_dims(frame, axis=0)
		processed =  keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
		predictions = model2.predict(processed)
		train_array = np.vstack([train_array, predictions])
		hasFrame, frame = cap.read()
	dire = path[0:path.rfind('/')]
	dire = os.path.join('CNN_NPY', dire)
	try:
		os.makedirs(dire)
	except:
		pass
	file_path = path[0:-3] + 'npy'
	file_path = os.path.join('CNN_NPY', file_path)
	with open(file_path, 'wb') as f:
		np.save(f, train_array)
	# print(train_array.shape)
	cap.release()

not_available = []
for path in tqdm(all_paths):
	file_path = path[0:-3] + 'mp4'
	file_path = os.path.join('SPARSE', file_path)
	cap = cv2.VideoCapture(file_path)#path to video
	if cap.isOpened():
		hasFrame, frame = cap.read()
	else:
		hasFrame = False
	if hasFrame:
		make_cnn(path)
	else:
		not_available.append(file_path)

with open('failed-files.txt', 'w') as f:
	for item in not_available:
		f.write("%s\n" % item)
