import os.path
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('include50')
from tqdm import tqdm
import numpy as np
from io import StringIO
import csv

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
filenames = [x[0:-3] + 'npy' for x in filenames]

names = csv.reader(blob2)
test_filenames = []
for name in names:
	test_filenames.append(name[0])
test_filenames = [x[0:-3] + 'npy' for x in test_filenames]


#This function is used to create the final data files that are fed into the lstm
#It ensures that all the files have 154 rows by padding rows of zeros and then stacks all the data from files together to create the final data file
def make_Data(filenames, path_to_save):
	train_data = []
	for path in tqdm(filenames):
		path = os.path.join('CNN_NPY', path)
		with open(path, 'rb') as f:
			arr = np.load(f)
		train_data.append(arr)
	uni_train_data = [np.vstack([arr, np.zeros(((154-arr.shape[0]), 1280))]) for arr in train_data]
	train_array = np.empty((0,1280))
	for array in uni_train_data:
		train_array = np.vstack([train_array, array])
	X_train = train_array.reshape((int(train_array.shape[0]/154), 154, 1280))
	print("data made")
	with open(os.path.join('DATA', path_to_save), 'wb') as f:
		np.save(f, X_train)

make_Data(filenames, 'LSTM-TRAIN.npy')
make_Data(test_filenames, 'LSTM-TEST.npy')
