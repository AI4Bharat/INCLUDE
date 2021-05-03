import json
import glob
import os

import torch
from torch.utils import data
import numpy as np
import pandas as pd
from augment import (
    Augmentation,
    OneOf,
    plus7rotation,
    minus7rotation,
    gaussSample,
    cutout,
    upsample,
    downsample,
)


class KeypointsDataset(data.Dataset):
    def __init__(
        self, keypoints_path, use_augs, label_map, mode="train", max_frame_len=169
    ):
        #print('###  keypoints_path = ' , keypoints_path)
        self.df = pd.read_json(keypoints_path)
        #with open(keypoints_path) as f:
        #    self.df = pd.DataFrame([json.loads(l) for l in f.readlines()])
        self.mode = mode
        self.use_augs = use_augs
        self.label_map = label_map
        self.max_frame_len = max_frame_len
        self.augs = None

        if mode == "train" and use_augs:
            self.augs = [
                Augmentation(OneOf(plus7rotation, minus7rotation), p=0.4),
                Augmentation(gaussSample, p=0.4),
                Augmentation(cutout, p=0.4),
                Augmentation(OneOf(upsample, downsample), p=0.4),
            ]

    def augment(self, df):
        for aug in self.augs:
            df = aug(df)
        return df

    def interpolate(self, arr):

        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        return np.stack([arr_x, arr_y], axis=-1)

    def combine_xy(self, x, y):
        x, y = np.array(x), np.array(y)
        _, length = x.shape
        x = x.reshape((-1, length, 1))
        y = y.reshape((-1, length, 1))
        return np.concatenate((x, y), -1).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row.label
        label = "".join([i for i in label if i.isalpha()]).lower()

        pose = self.combine_xy(row.pose_x, row.pose_y)
        h1 = self.combine_xy(row.hand1_x, row.hand1_y)
        h2 = self.combine_xy(row.hand2_x, row.hand2_y)

        pose = self.interpolate(pose)
        h1 = self.interpolate(h1)
        h2 = self.interpolate(h2)

        df = pd.DataFrame.from_dict(
            {
                "uid": row.uid,
                "pose": pose.tolist(),
                "hand1": h1.tolist(),
                "hand2": h2.tolist(),
                "label": label,
            }
        )
        if self.mode == "train" and self.use_augs:
            df = self.augment(df)

        pose = (
            np.array(list(map(np.array, df.pose.values)))
            .reshape(-1, 50)
            .astype(np.float32)
        )
        h1 = (
            np.array(list(map(np.array, df.hand1.values)))
            .reshape(-1, 42)
            .astype(np.float32)
        )
        h2 = (
            np.array(list(map(np.array, df.hand2.values)))
            .reshape(-1, 42)
            .astype(np.float32)
        )
        final_data = np.concatenate((pose, h1, h2), -1)
        final_data = np.pad(
            final_data,
            ((0, self.max_frame_len - final_data.shape[0]), (0, 0)),
            "constant",
        )
        return {
            "uid": row.uid,
            "data": torch.FloatTensor(final_data),
            "label": self.label_map[label],
            "lablel_string": label,
        }

    def __len__(self):
        return len(self.df)


class FeaturesDatset(data.Dataset):
    def __init__(self, features_dir, label_map, mode="train"):
        self.features_dir = features_dir
        self.file_paths = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
        self.label_map = label_map
        self.mode = mode

    def __getitem__(self, i):
        file_path = self.file_paths[i]
        data = np.load(file_path)
        label = os.path.basename(file_path).split("_")[0]
        return {
            "uid": os.path.basename(file_path).split(".")[0],
            "data": torch.FloatTensor(data),
            "label": self.label_map[label],
            "lablel_string": label,
        }

    def __len__(self):
        return len(self.file_paths)
