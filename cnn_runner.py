import os
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from models import CNN
from configs import CnnConfig
from utils import load_json
import cv2
import glob


def replace_nan(x, y):
    x = 0.0 if np.isnan(x) else x
    y = 0.0 if np.isnan(y) else y
    return x, y


def draw_hands(
    image,
    hand_x,
    hand_y,
    connections,
    connection_color,
    thickness,
    point_color,
    frame_length,
    frame_width,
):
    for connection in connections:
        x0 = hand_x[connection[0]] * frame_width
        y0 = hand_y[connection[0]] * frame_length
        x1 = hand_x[connection[1]] * frame_width
        y1 = hand_y[connection[1]] * frame_length

        x0, y0 = replace_nan(x0, y0)
        x1, y1 = replace_nan(x1, y1)

        cv2.line(
            image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            connection_color,
            thickness,
        )

    for x, y in zip(hand_x, hand_y):
        x, y = replace_nan(x, y)
        cv2.circle(image, (int(x), int(y)), thickness, point_color, thickness)

    return image


def draw_pose(
    image,
    pose_x,
    pose_y,
    links,
    connection_color,
    thickness,
    point_color,
    frame_length,
    frame_width,
):
    for link in links:
        x0 = pose_x[link[0]] * frame_width
        y0 = pose_y[link[0]] * frame_length
        x1 = pose_x[link[1]] * frame_width
        y1 = pose_y[link[1]] * frame_length

        x0, y0 = replace_nan(x0, y0)
        x1, y1 = replace_nan(x1, y1)

        cv2.line(
            image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            connection_color,
            thickness,
        )
        cv2.circle(image, (int(x0), int(y0)), thickness, point_color, thickness)
        cv2.circle(image, (int(x1), int(y1)), thickness, point_color, thickness)

    return image


def cnn_feat(file_path, save_dir):
    video_record = load_json(file_path)

    FRAME_LENGTH = 1080
    FRAME_WIDTH = 1920
    POINT_COLOR = (255, 0, 0)
    CONNECTION_COLOR = (0, 255, 0)
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

    links = [
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 21),
        (15, 17),
        (17, 19),
        (19, 15),
        (22, 16),
        (16, 18),
        (18, 20),
        (16, 20),
    ]

    model = CNN(CnnConfig)
    features = np.empty((0, CnnConfig.output_dim))

    assert video_record["n_frames"] > 0, "Number of frames should be greater than zero"
    for i in range(video_record["n_frames"]):
        image = np.zeros((FRAME_LENGTH, FRAME_WIDTH, 3), np.uint8)
        pose_x = video_record["pose_x"][i]
        pose_y = video_record["pose_y"][i]
        hand1_x = video_record["hand1_x"][i]
        hand1_y = video_record["hand1_y"][i]
        hand2_x = video_record["hand2_x"][i]
        hand2_y = video_record["hand2_y"][i]

        if hand1_x[0] != 0:
            image = draw_hands(
                image,
                hand1_x,
                hand1_y,
                connections,
                CONNECTION_COLOR,
                THICKNESS,
                POINT_COLOR,
                FRAME_LENGTH,
                FRAME_WIDTH,
            )

        if hand2_x[0] != 0:
            image = draw_hands(
                image,
                hand2_x,
                hand2_y,
                connections,
                CONNECTION_COLOR,
                THICKNESS,
                POINT_COLOR,
                FRAME_LENGTH,
                FRAME_WIDTH,
            )

        image = draw_pose(
            image,
            pose_x,
            pose_y,
            links,
            CONNECTION_COLOR,
            THICKNESS,
            POINT_COLOR,
            FRAME_LENGTH,
            FRAME_WIDTH,
        )
        image = image.astype(np.float32) / 255
        image = cv2.resize(image, (224, 224))
        feat = model(torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0))
        features = np.vstack([features, feat.numpy()])

    save_path = os.path.join(save_dir, video_record["uid"] + ".npy")
    np.save(save_path, features)


def runner(args, mode):
    json_files = sorted(
        glob.glob(
            os.path.join(args.data_dir, f"{args.dataset}_{mode}_keypoints", "*.json")
        )
    )

    save_dir = os.path.join(args.data_dir, f"{args.dataset}_{mode}_features")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    saved_files = glob.glob(os.path.join(save_dir, "*.npy"))
    if len(saved_files) != len(json_files):
        for file_path in tqdm(json_files, desc=f"Saving CNN features - {mode} files"):
            cnn_feat(file_path, save_dir)
    else:
        print(mode, "CNN features already exist!")


def save_cnn_features(args):
    runner(args, mode="train")
    runner(args, mode="val")
    runner(args, mode="test")
