import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# MP params
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
pose = mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

train_files = pd.read_csv("Train_Test_Split/Include_train_test/train.csv")
test_files = pd.read_csv("Train_Test_Split/Include_train_test/test.csv")
#train_matrix = pd.DataFrame()


filePaths = list(train_files['FilePath'].values) + list(test_files['FilePath'].values)

print(len(filePaths))
for f in filePaths:
    if not os.path.isfile(f):
        print(f)


def getHandedness(pose, hands):
    if hands == None:
        return None
    elif len(hands) == 1:
        leftWrist_x = pose[15]['x']
        leftWrist_y = pose[15]['y']

        rightWrist_x = pose[16]['x']
        rightWrist_y = pose[16]['y']

        hands_x = hands[0][0]['x']
        hands_y = hands[0][0]['y']

        left_len = (leftWrist_x - hands_x) ** 2 + (leftWrist_y - hands_y) ** 2
        right_len = (rightWrist_x - hands_x) ** 2 + (rightWrist_y - hands_y) ** 2

        if left_len < right_len:
            return 'h2'
        else:
            return 'h1'

    elif len(hands) == 2:
        leftWrist_x = pose[15]['x']
        leftWrist_y = pose[15]['y']

        rightWrist_x = pose[16]['x']
        rightWrist_y = pose[16]['y']

        hands_x = hands[0][0]['x']
        hands_y = hands[0][0]['y']

        left_len = (leftWrist_x - hands_x) ** 2 + (leftWrist_y - hands_y) ** 2
        right_len = (rightWrist_x - hands_x) ** 2 + (rightWrist_y - hands_y) ** 2

        if left_len < right_len:
            return 'h2', 'h1'
        else:
            return 'h1', 'h2'


for i in tqdm(range(2474 , len(filePaths))):
    filePath = filePaths[i]
    cap = cv2.VideoCapture(filePath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    handLandmark = []
    poseLandmark = []

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results_hand = hands.process(image)
        results_pose = pose.process(image)
        handLandmark.append(results_hand.multi_hand_landmarks)
        poseLandmark.append(results_pose.pose_landmarks)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    for i in range(len(poseLandmark)):
        poseLandmark[i] = MessageToDict(poseLandmark[i])
        poseLandmark[i] = poseLandmark[i]['landmark']

    for i in range(len(handLandmark)):
        if handLandmark[i] != None:
            if len(handLandmark[i]) == 1:
                handLandmark[i][0] = MessageToDict(handLandmark[i][0])['landmark']
            elif len(handLandmark[i]) == 2:
                handLandmark[i][0] = MessageToDict(handLandmark[i][0])['landmark']
                handLandmark[i][1] = MessageToDict(handLandmark[i][1])['landmark']

    cols = ['pose_x' + str(i) for i in range(25)] + ['pose_y' + str(i) for i in range(25)]
    cols += ['h1_x' + str(i) for i in range(21)] + ['h1_y' + str(i) for i in range(21)]
    cols += ['h2_x' + str(i) for i in range(21)] + ['h2_y' + str(i) for i in range(21)]
    df = pd.DataFrame(columns=cols, index=list([i for i in range(frameCount)]))

    for i in range(frameCount):
        if poseLandmark[i] != None:
            lndmrk = poseLandmark[i]
            xCoord = np.array(list([lndmrk[j]['x'] for j in range(25)])) * 1920
            yCoord = np.array(list([lndmrk[j]['y'] for j in range(25)])) * 1080
            df.at[i, 'pose_x0':'pose_x24'] = xCoord
            df.at[i, 'pose_y0':'pose_y24'] = yCoord

        if handLandmark[i] != None:
            handCount = len(handLandmark[i])
            if handCount == 1:
                handedness = getHandedness(poseLandmark[i], handLandmark[i])

                lndmrk0 = handLandmark[i][0]
                xCoord = np.array(list([lndmrk0[j]['x'] for j in range(21)])) * 1920
                yCoord = np.array(list([lndmrk0[j]['y'] for j in range(21)])) * 1080

                if handedness == 'h1':
                    df.at[i, 'h1_x0':'h1_x20'] = xCoord
                    df.at[i, 'h1_y0':'h1_y20'] = yCoord
                elif handedness == 'h2':
                    df.at[i, 'h2_x0':'h2_x20'] = xCoord
                    df.at[i, 'h2_y0':'h2_y20'] = yCoord

            elif handCount == 2:
                handedness = getHandedness(poseLandmark[i], handLandmark[i])

                lndmrk0 = handLandmark[i][0]
                lndmrk1 = handLandmark[i][1]
                xCoord = {}
                yCoord = {}
                xCoord[0] = np.array(list([lndmrk0[j]['x'] for j in range(21)])) * 1920
                yCoord[0] = np.array(list([lndmrk0[j]['y'] for j in range(21)])) * 1080
                xCoord[1] = np.array(list([lndmrk1[j]['x'] for j in range(21)])) * 1920
                yCoord[1] = np.array(list([lndmrk1[j]['y'] for j in range(21)])) * 1080

                if list(handedness) == ['h1', 'h2']:
                    df.at[i, 'h1_x0':'h1_x20'] = xCoord[0]
                    df.at[i, 'h1_y0':'h1_y20'] = yCoord[0]
                    df.at[i, 'h2_x0':'h2_x20'] = xCoord[1]
                    df.at[i, 'h2_y0':'h2_y20'] = yCoord[1]

                elif list(handedness) == ['h2', 'h1']:
                    df.at[i, 'h1_x0':'h1_x20'] = xCoord[1]
                    df.at[i, 'h1_y0':'h1_y20'] = yCoord[1]
                    df.at[i, 'h2_x0':'h2_x20'] = xCoord[0]
                    df.at[i, 'h2_y0':'h2_y20'] = yCoord[0]

    cap.release()
    df['label'] = filePath
    filePath = filePath[:-3] + 'csv'
    df.to_csv('INCLUDE_csv/' + filePath)