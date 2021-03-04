import torch
from torch.utils import data
import numpy as np
import json
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


class IncludeDataset(data.Dataset):
    def __init__(
        self, df_path, use_augs, label_map_path="utils/label_map.json", mode="train"
    ):
        self.df = pd.read_csv(df_path)
        self.mode = mode
        self.use_augs = use_augs
        self.label_map = self._load_file(label_map_path)
        self.augs = None
        if mode == "train" and use_augs:
            self.augs = [
                Augmentation(OneOf(plus7rotation, minus7rotation), p=0.4),
                Augmentation(gaussSample, p=0.4),
                Augmentation(cutout, p=0.4),
                Augmentation(OneOf(upsample, downsample), p=0.4),
            ]

    def _load_file(self, path):
        with open(path, "r") as fp:
            data = json.load(fp)
        return data

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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row[-1]
        label = "".join([i for i in label if i.isalpha()]).lower()
        values = row[1:-1].values.reshape(154, -1)

        pose = np.concatenate(
            (values[:, :25].reshape(154, -1, 1), values[:, 25:50].reshape(154, -1, 1)),
            -1,
        ).astype(np.float32)
        h1 = np.concatenate(
            (
                values[:, 50:71].reshape(154, -1, 1),
                values[:, 71:92].reshape(154, -1, 1),
            ),
            -1,
        ).astype(np.float32)
        h2 = np.concatenate(
            (
                values[:, 92:113].reshape(154, -1, 1),
                values[:, 113:134].reshape(154, -1, 1),
            ),
            -1,
        ).astype(np.float32)
        pose = self.interpolate(pose)
        h1 = self.interpolate(h1)
        h2 = self.interpolate(h2)

        df = pd.DataFrame.from_dict(
            {
                "uid": row.filePath,
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
            final_data, ((0, 169 - final_data.shape[0]), (0, 0)), "constant"
        )
        return {
            "uid": row.filePath,
            "data": final_data,
            "label": self.label_map[label],
            "lablel_string": label,
        }

    def __len__(self):
        return len(self.df)
