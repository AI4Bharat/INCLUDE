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


def plus7rotation(df0):
    # +7 degree rotation
    df0_plus7 = pd.DataFrame()
    df0_plus7["uid"] = df0["uid"]
    df0_plus7["pose"] = ""
    df0_plus7["hand1"] = ""
    df0_plus7["hand2"] = ""
    df0_plus7["label"] = df0["label"]

    theta = 7 * (np.pi / 180)
    c, s = np.cos(theta), np.sin(theta)
    augmentMatrix = np.array([[c, -s], [s, c]])

    for i in range(df0.shape[0]):
        for col in ["pose", "hand1", "hand2"]:
            try:
                matrix = np.array(df0.iloc[i].loc[col], dtype=np.float)
                matrix = np.matmul(matrix, augmentMatrix)
                matrix = np.where(np.isnan(matrix), None, matrix).tolist()
                df0_plus7.at[i, col] = matrix
            except:
                print(i)

    return df0_plus7


def minus7rotation(df0):
    # -7 degree rotation
    df0_minus7 = pd.DataFrame()
    df0_minus7["uid"] = df0["uid"]
    df0_minus7["pose"] = ""
    df0_minus7["hand1"] = ""
    df0_minus7["hand2"] = ""
    df0_minus7["label"] = df0["label"]

    theta = -7 * (np.pi / 180)
    c, s = np.cos(theta), np.sin(theta)
    augmentMatrix = np.array([[c, -s], [s, c]])

    for i in range(df0.shape[0]):
        for col in ["pose", "hand1", "hand2"]:
            try:
                matrix = np.array(df0.iloc[i].loc[col], dtype=np.float)
                matrix = np.matmul(matrix, augmentMatrix)
                matrix = np.where(np.isnan(matrix), None, matrix).tolist()
                df0_minus7.at[i, col] = matrix
            except:
                print(i)

    return df0_minus7


def gaussSample(df0):
    # Random Gaussian sampling
    df0_gaussSample = df0.copy()
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

    x_width = 960
    y_height = 540
    for i in range(df0.shape[0]):
        if np.count_nonzero(df0.iloc[i].loc["pose"]) == 0:
            padIdx = i
            break
        try:
            pose = np.array(df0.iloc[i].loc["pose"], dtype=np.float)
            pose[:, 0] /= x_width
            pose[:, 1] /= y_height
            poseVariance = np.column_stack((sigma, sigma))
            pose = np.random.normal(pose, poseVariance)
            pose[:, 0] *= x_width
            pose[:, 1] *= y_height
            pose = np.where(np.isnan(pose), None, pose).tolist()

            hand1 = np.array(df0.iloc[i].loc["hand1"], dtype=np.float)
            hand1[:, 0] /= x_width
            hand1[:, 1] /= y_height
            hand1 = np.random.normal(hand1, dv)
            hand1[:, 0] *= x_width
            hand1[:, 1] *= y_height
            hand1 = np.where(np.isnan(hand1), None, hand1).tolist()

            hand2 = np.array(df0.iloc[i].loc["hand1"], dtype=np.float)
            hand2[:, 0] /= x_width
            hand2[:, 1] /= y_height
            hand2 = np.random.normal(hand2, dv)
            hand2[:, 0] *= x_width
            hand2[:, 1] *= y_height
            hand2 = np.where(np.isnan(hand2), None, hand2).tolist()

            df0_gaussSample.at[i, "pose"] = pose
            df0_gaussSample.at[i, "hand1"] = hand1
            df0_gaussSample.at[i, "hand2"] = hand2
        except:
            print(i)
            pass

    return df0_gaussSample


def cutout(df0):
    # cutout
    df0_cutout = df0.copy()

    padIdx = 0
    for i in range(df0.shape[0]):
        if np.count_nonzero(df0.iloc[i].loc["pose"]) == 0:
            padIdx = i
            break

    for i in range(df0.shape[0]):
        if np.count_nonzero(df0.iloc[i].loc["pose"]) == 0:
            break
        if i < padIdx:
            pose = np.array(df0.iloc[i].loc["pose"])
            hand1 = np.array(df0.iloc[i].loc["hand1"])
            hand2 = np.array(df0.iloc[i].loc["hand1"])
            poseZeroIdx = np.random.choice(25, 3, replace=False)
            hand1ZeroIdx = np.random.choice(21, 3, replace=False)
            hand2ZeroIdx = np.random.choice(21, 3, replace=False)

            for i in poseZeroIdx:
                pose[i] = [0, 0]
            for i in hand1ZeroIdx:
                hand1[i] = [0, 0]
            for i in hand2ZeroIdx:
                hand2[i] = [0, 0]

            pose = pose.tolist()
            hand1 = hand1.tolist()
            hand2 = hand2.tolist()

            df0_cutout.at[i, "pose"] = pose
            df0_cutout.at[i, "hand1"] = hand1
            df0_cutout.at[i, "hand2"] = hand2

    return df0_cutout


def downsample(df0):
    # downsample
    df0_downsample = df0.copy()
    drop_idx = np.random.choice(154, 15)  # 154 frames , 15 frames
    df0_downsample = df0_downsample.drop(index=drop_idx)
    return df0_downsample


def upsample(df0):
    # upsample
    df0_upsample = pd.DataFrame(
        index=np.arange(169), columns=["uid", "pose", "hand1", "hand2", "label"]
    )  # 154 + 15 extra frames
    df0_upsample["uid"] = df0.iloc[0].loc["uid"]

    j = 0
    for i in range(df0_upsample.shape[0]):
        if i % 10 != 0 or i == 0:
            df0_upsample.at[i, "pose"] = df0.iloc[j].loc["pose"]
            df0_upsample.at[i, "hand1"] = df0.iloc[j].loc["hand1"]
            df0_upsample.at[i, "hand2"] = df0.iloc[j].loc["hand2"]
            j += 1
        else:
            try:
                pose = (
                    np.array(df0.iloc[j - 1].loc["pose"], dtype=np.float)
                    + np.array(df0.iloc[j].loc["pose"], dtype=np.float)
                ) / 2  # get average of two frames
                # why dtype = np.float: convert None to np.nan, so that a complete frame isn't missing
                df0_upsample.at[i, "pose"] = np.where(
                    np.isnan(pose), None, pose
                ).tolist()  # make np.nan to None

                hand1 = (
                    (
                        np.array(df0.iloc[j - 1].loc["hand1"], dtype=np.float)
                        + np.array(df0.iloc[j].loc["hand1"], dtype=np.float)
                    )
                    / 2
                ).tolist()
                df0_upsample.at[i, "hand1"] = np.where(
                    np.isnan(hand1), None, hand1
                ).tolist()

                hand2 = (
                    (
                        np.array(df0.iloc[j - 1].loc["hand2"], dtype=np.float)
                        + np.array(df0.iloc[j].loc["hand2"], dtype=np.float)
                    )
                    / 2
                ).tolist()
                df0_upsample.at[i, "hand2"] = np.where(
                    np.isnan(hand2), None, hand2
                ).tolist()

            except:
                print(i)

    df0_upsample["label"] = df0.iloc[0].loc["label"]
    return df0_upsample