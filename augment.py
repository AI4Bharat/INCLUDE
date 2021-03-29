import pandas as pd
import numpy as np


class Augmentation:
    def __init__(self, aug_func, p=1):
        self.aug_func = aug_func
        self.p = p

    def __call__(self, df):
        if np.random.rand() <= self.p:
            return self.aug_func(df)
        return df


def OneOf(aug_a, aug_b):
    if np.random.rand() < 0.5:
        return aug_a
    return aug_b


def plus7rotation(df):
    # +7 degree rotation
    df_augmented = pd.DataFrame()
    df_augmented["uid"] = df["uid"]
    df_augmented["pose"] = ""
    df_augmented["hand1"] = ""
    df_augmented["hand2"] = ""
    df_augmented["label"] = df["label"]

    theta = 7 * (np.pi / 180)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    for i in range(df.shape[0]):
        for col in ["pose", "hand1", "hand2"]:
            matrix = np.array(df.loc[i, col], dtype=np.float)
            matrix = np.matmul(matrix, rotation_matrix)
            matrix = np.where(np.isnan(matrix), None, matrix).tolist()
            df_augmented.at[i, col] = matrix

    return df_augmented


def minus7rotation(df):
    # -7 degree rotation
    df_augmented = pd.DataFrame()
    df_augmented["uid"] = df["uid"]
    df_augmented["pose"] = ""
    df_augmented["hand1"] = ""
    df_augmented["hand2"] = ""
    df_augmented["label"] = df["label"]

    theta = -7 * (np.pi / 180)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    for i in range(df.shape[0]):
        for col in ["pose", "hand1", "hand2"]:
            matrix = np.array(df.loc[i, col], dtype=np.float)
            matrix = np.matmul(matrix, rotation_matrix)
            matrix = np.where(np.isnan(matrix), None, matrix).tolist()
            df_augmented.at[i, col] = matrix

    return df_augmented


def gaussSample(df):
    # Random Gaussian sampling
    df_augmented = df.copy()
    dv = 0.05 * 10 ** -2
    sv = 0.08 * 10 ** -2
    lv = 0.08 * 10 ** -1
    sigma = [
        sv,
        dv,
        dv,
        dv,
        dv,
        dv,
        dv,
        sv,
        sv,
        sv,
        sv,
        lv,
        lv,
        lv,
        lv,
        sv,
        sv,
        sv,
        sv,
        sv,
        sv,
        sv,
        sv,
        lv,
        lv,
    ]

    ## Check if keypoints is range [0, 1]
    x_width = 1920
    y_height = 1080
    for i in range(df.shape[0]):
        if np.count_nonzero(df.loc[i, "pose"]) == 0:
            break

        pose = np.array(df.loc[i, "pose"], dtype=np.float)
        pose[:, 0] /= x_width
        pose[:, 1] /= y_height
        pose_variance = np.column_stack((sigma, sigma))
        pose = np.random.normal(pose, pose_variance)
        pose[:, 0] *= x_width
        pose[:, 1] *= y_height
        pose = np.where(np.isnan(pose), None, pose).tolist()

        hand1 = np.array(df.loc[i, "hand1"], dtype=np.float)
        hand1[:, 0] /= x_width
        hand1[:, 1] /= y_height
        hand1 = np.random.normal(hand1, dv)
        hand1[:, 0] *= x_width
        hand1[:, 1] *= y_height
        hand1 = np.where(np.isnan(hand1), None, hand1).tolist()

        hand2 = np.array(df.loc[i, "hand2"], dtype=np.float)
        hand2[:, 0] /= x_width
        hand2[:, 1] /= y_height
        hand2 = np.random.normal(hand2, dv)
        hand2[:, 0] *= x_width
        hand2[:, 1] *= y_height
        hand2 = np.where(np.isnan(hand2), None, hand2).tolist()

        df_augmented.at[i, "pose"] = pose
        df_augmented.at[i, "hand1"] = hand1
        df_augmented.at[i, "hand2"] = hand2

    return df_augmented


def cutout(df):
    # cutout
    df_augmented = df.copy()

    pad_idx = 0
    for i in range(df.shape[0]):
        if np.count_nonzero(df.loc[i, "pose"]) == 0:
            pad_idx = i
            break

    for i in range(df.shape[0]):
        if np.count_nonzero(df.loc[i, "pose"]) == 0:
            break

        if i < pad_idx:
            pose = np.array(df.loc[i, "pose"])
            hand1 = np.array(df.loc[i, "hand1"])
            hand2 = np.array(df.loc[i, "hand2"])
            pose_zero_idx = np.random.choice(25, 3, replace=False)
            hand1_zero_idx = np.random.choice(21, 3, replace=False)
            hand2_zero_idx = np.random.choice(21, 3, replace=False)

            for i in pose_zero_idx:
                pose[i] = [0, 0]
            for i in hand1_zero_idx:
                hand1[i] = [0, 0]
            for i in hand2_zero_idx:
                hand2[i] = [0, 0]

            pose = pose.tolist()
            hand1 = hand1.tolist()
            hand2 = hand2.tolist()

            df_augmented.at[i, "pose"] = pose
            df_augmented.at[i, "hand1"] = hand1
            df_augmented.at[i, "hand2"] = hand2

    return df_augmented


def downsample(df):
    # downsample
    df_augmented = df.copy()
    drop_idx = np.random.choice(154, 15)  # 154 frames , 15 frames
    df_augmented = df_augmented.drop(index=drop_idx)
    return df_augmented


def upsample(df):
    # upsample
    def get_avg(df, idx, col):
        aug_points = (
            (
                np.array(df.loc[idx - 1, col], dtype=np.float)
                + np.array(df.loc[idx, col], dtype=np.float)
            )
            / 2
        ).tolist()
        return np.where(np.isnan(aug_points), None, aug_points).tolist()

    df_augmented = pd.DataFrame(
        index=np.arange(169), columns=["uid", "pose", "hand1", "hand2", "label"]
    )  # 154 + 15 extra frames
    df_augmented["uid"] = df.iloc[0].loc["uid"]

    j = 0
    for i in range(df_augmented.shape[0]):
        if i % 10 != 0 or i == 0:
            df_augmented.at[i, "pose"] = df.loc[j, "pose"]
            df_augmented.at[i, "hand1"] = df.loc[j, "hand1"]
            df_augmented.at[i, "hand2"] = df.loc[j, "hand2"]
            j += 1
            continue

        df_augmented.at[i, "pose"] = get_avg(df, j, "pose")
        df_augmented.at[i, "hand1"] = get_avg(df, j, "hand1")
        df_augmented.at[i, "hand2"] = get_avg(df, j, "hand2")

    df_augmented["label"] = df.iloc[0].loc["label"]
    return df_augmented
