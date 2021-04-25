import os
import json
import multiprocessing
import argparse

import cv2
import mediapipe as mp
from tqdm.auto import tqdm
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description="Generate keypoints from Mediapipe")
parser.add_argument(
    "--include_dir",
    default="",
    type=str,
    required=True,
    help="path to the location of INCLUDE/INCLUDE50 videos",
)
parser.add_argument(
    "--save_dir",
    default="",
    type=str,
    required=True,
    help="location to output json file",
)
parser.add_argument(
    "--dataset", default="include", type=str, help="options: include or include50"
)
args = parser.parse_args()


def process_landmarks(landmarks):
    x_list, y_list = [], []
    for landmark in landmarks.landmark:
        x_list.append(landmark.x)
        y_list.append(landmark.y)
    return x_list, y_list


def process_hand_keypoints(results):
    hand1_x, hand1_y, hand2_x, hand2_y = [], [], [], []

    if results.multi_hand_landmarks is not None:
        if len(results.multi_hand_landmarks) > 0:
            hand1 = results.multi_hand_landmarks[0]
            hand1_x, hand1_y = process_landmarks(hand1)

        if len(results.multi_hand_landmarks) > 1:
            hand2 = results.multi_hand_landmarks[1]
            hand2_x, hand2_y = process_landmarks(hand2)

    return hand1_x, hand1_y, hand2_x, hand2_y


def process_pose_keypoints(results):
    pose = results.pose_landmarks
    pose_x, pose_y = process_landmarks(pose)
    return pose_x, pose_y


def swap_hands(left_wrist, right_wrist, hand, input_hand):
    left_wrist_x, left_wrist_y = left_wrist
    right_wrist_x, right_wrist_y = right_wrist
    hand_x, hand_y = hand

    left_dist = (left_wrist_x - hand_x) ** 2 + (left_wrist_y - hand_y) ** 2
    right_dist = (right_wrist_x - hand_x) ** 2 + (right_wrist_y - hand_y) ** 2

    if left_dist < right_dist and input_hand == "h2":
        return True

    if right_dist < left_dist and input_hand == "h1":
        return True

    return False


def process_video(path):
    hands = mp.solutions.hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=True
    )

    pose_points_x, pose_points_y = [], []
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []

    label = path.split("/")[3]
    label = "".join([i for i in label if i.isalpha()]).lower()
    uid = os.path.splitext(os.path.basename(path))[0]
    uid = "_".join([label, uid])
    n_frames = 0

    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(image)
        pose_results = pose.process(image)

        hand1_x, hand1_y, hand2_x, hand2_y = process_hand_keypoints(hand_results)
        pose_x, pose_y = process_pose_keypoints(pose_results)

        ## Assign hands to correct positions
        if len(hand1_x) > 0 and len(hand2_x) == 0:
            if swap_hands(
                left_wrist=(pose_x[15], pose_y[15]),
                right_wrist=(pose_x[16], pose_y[16]),
                hand=(hand1_x[0], hand1_y[0]),
                input_hand="h1",
            ):
                hand1_x, hand1_y, hand2_x, hand2_y = hand2_x, hand2_y, hand1_x, hand1_y

        elif len(hand1_x) == 0 and len(hand2_x) > 0:
            if swap_hands(
                left_wrist=(pose_x[15], pose_y[15]),
                right_wrist=(pose_x[16], pose_y[16]),
                hand=(hand2_x[0], hand2_y[0]),
                input_hand="h2",
            ):
                hand1_x, hand1_y, hand2_x, hand2_y = hand2_x, hand2_y, hand1_x, hand1_y

        pose_x = pose_x if pose_x else [0.0] * 25
        pose_y = pose_y if pose_y else [0.0] * 25

        hand1_x = hand1_x if hand1_x else [0.0] * 21
        hand1_y = hand1_y if hand1_y else [0.0] * 21
        hand2_x = hand2_x if hand2_x else [0.0] * 21
        hand2_y = hand2_y if hand2_y else [0.0] * 21

        pose_points_x.append(pose_x)
        pose_points_y.append(pose_y)
        hand1_points_x.append(hand1_x)
        hand1_points_y.append(hand1_y)
        hand2_points_x.append(hand2_x)
        hand2_points_y.append(hand2_y)

        n_frames += 1

    cap.release()

    ## Unable to open video
    pose_points_x = pose_points_x if pose_points_x else [[0.0] * 25]
    pose_points_y = pose_points_y if pose_points_y else [[0.0] * 25]

    hand1_points_x = hand1_points_x if hand1_points_x else [[0.0] * 21]
    hand1_points_y = hand1_points_y if hand1_points_y else [[0.0] * 21]
    hand2_points_x = hand2_points_x if hand2_points_x else [[0.0] * 21]
    hand2_points_y = hand2_points_y if hand2_points_y else [[0.0] * 21]

    return {
        "uid": uid,
        "label": label,
        "pose_x": pose_points_x,
        "pose_y": pose_points_y,
        "hand1_x": hand1_points_x,
        "hand1_y": hand1_points_y,
        "hand2_x": hand2_points_x,
        "hand2_y": hand2_points_y,
        "n_frames": n_frames,
    }


def load_file(path, include_dir):
    with open(path, "r") as fp:
        data = fp.read()
        data = data.split("\n")
    data = list(map(lambda x: os.path.join(include_dir, x), data))
    return data


def load_train_test_val_paths(args):
    train_paths = load_file(
        f"train_test_paths/{args.dataset}_train.txt", args.include_dir
    )
    val_paths = load_file(f"train_test_paths/{args.dataset}_val.txt", args.include_dir)
    test_paths = load_file(
        f"train_test_paths/{args.dataset}_test.txt", args.include_dir
    )
    return train_paths, val_paths, test_paths


def save_keypoints(dataset, file_paths, mode):
    outputs = Parallel(n_jobs=n_cores, backend="multiprocessing")(
        delayed(process_video)(path)
        for path in tqdm(file_paths, desc=f"processing {mode} videos")
    )

    print(f"Saving {mode} points.....")
    save_path = os.path.join(args.save_dir, f"{dataset}_{mode}_keypoints.json")
    with open(save_path, "w") as f:
        json.dump(outputs, f)


n_cores = multiprocessing.cpu_count()
train_paths, val_paths, test_paths = load_train_test_val_paths(args)

save_keypoints(args.dataset, train_paths, "train")
save_keypoints(args.dataset, val_paths, "val")
save_keypoints(args.dataset, test_paths, "test")
