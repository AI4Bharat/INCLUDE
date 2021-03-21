import numpy as np
import tensorflow as tf
import os
import csv
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('include50')
import os.path
import os
import pandas as pd
from io import StringIO
from io import BytesIO
from tensorflow.python.lib.io import file_io
import cv2
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
import os.path
import skvideo.io
import numpy as np
import os.path
import skvideo.io
from tqdm import tqdm


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

POINT_COLOR = (255,0,0)
CONNECTION_COLOR = (0,255,0)
THICKNESS = 2

connections = [
	(0, 1),
	(1, 2),
	(2, 3),
	(3, 4),
	(5, 6),
	(6, 7),
	(7, 8),
	(9, 10),
	(10, 11),
	(11, 12),
	(13, 14),
	(14, 15),
	(15, 16),
	(17, 18),
	(18, 19),
	(19, 20),
	(0, 5),
	(5, 9),
	(9, 13),
	(13, 17),
	(0, 17),
]

links = [(11, 12), (11, 23), (12, 24),
(23, 24), (11, 13), (13, 15), (12, 14), (14, 16), (15, 21),
(15, 17), (17, 19), (19, 15), (22, 16), (16, 18),
(18, 20), (16, 20)]

#This function creates SPARE videos from the input videos of INCLUDE by plotting all the points and the connections between them on a black screen

def process_video(df):
    videodata = []
    for i in range(df.shape[0]):
        image = np.zeros((1080, 1920, 3), np.uint8)
        row_points = df.loc[i, :].to_list()
        row_points = row_points[:-1]
        xlist_pose = row_points[0:25]
        ylist_pose = row_points[25:50]
        xlist_h1 = row_points[50:71]
        ylist_h1 = row_points[71:92]
        xlist_h2 = row_points[92:113]
        ylist_h2 = row_points[113:134]
        if xlist_h1[0] != 0:
            for connection in connections:
                x0 = xlist_h1[connection[0]]
                y0 = ylist_h1[connection[0]]
                x1 = xlist_h1[connection[1]]
                y1 = ylist_h1[connection[1]]

                cv2.line(
                    image,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    CONNECTION_COLOR,
                    THICKNESS,
                )
            for j in range(len(xlist_h1)):
                x = xlist_h1[j]
                y = ylist_h1[j]
                cv2.circle(image, (int(x), int(y)), THICKNESS, POINT_COLOR, THICKNESS)

        if xlist_h2[0] != 0:
            for connection in connections:
                x0 = xlist_h2[connection[0]]
                y0 = ylist_h2[connection[0]]
                x1 = xlist_h2[connection[1]]
                y1 = ylist_h2[connection[1]]

                cv2.line(
                    image,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    CONNECTION_COLOR,
                    THICKNESS,
                )
            for k in range(len(xlist_h2)):
                x = xlist_h2[k]
                y = ylist_h2[k]
                cv2.circle(image, (int(x), int(y)), THICKNESS, POINT_COLOR, THICKNESS)

        for link in links:
            x0 = xlist_pose[link[0]]
            x1 = xlist_pose[link[1]]
            y0 = ylist_pose[link[0]]
            y1 = ylist_pose[link[1]]
            cv2.line(
                image,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                CONNECTION_COLOR,
                THICKNESS,
            )
            cv2.circle(image, (int(x0), int(y0)), THICKNESS, POINT_COLOR, THICKNESS)
            cv2.circle(image, (int(x1), int(y1)), THICKNESS, POINT_COLOR, THICKNESS)

        videodata.append(image)
    sparse_rep = np.array(videodata)
    dire = path[0:path.rfind('/')]
    dire = os.path.join('SPARSE', dire)
    try:
        os.makedirs(dire)
    except:
        pass
    file_path = path[0:-3] + 'mp4'
    file_path = os.path.join('SPARSE', file_path)
    skvideo.io.vwrite(file_path, sparse_rep)


root = 'gs://include50/INCLUDE_csv'

for path in tqdm(all_paths):
	  path_new = os.path.join(root, path)
	  df = pd.read_csv(path_new)
	  df = df.fillna(0)
	  del df['Unnamed: 0']
	  process_video(df)
