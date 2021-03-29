import os

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from models import Xgboost
from configs import XgbConfig
from utils import get_experiment_name, load_label_map
from augment import (
    plus7rotation,
    minus7rotation,
    gaussSample,
    cutout,
    upsample,
    downsample,
)
from tqdm.auto import tqdm


def flatten(arr, max_seq_len=200):
    arr = np.array(arr)
    arr = np.pad(arr, ((0, max_seq_len - arr.shape[0]), (0, 0)), "constant")
    arr = arr.flatten()
    return arr


def combine_xy(x, y):
    x, y = np.array(x), np.array(y)
    _, length = x.shape
    x = x.reshape((-1, length, 1))
    y = y.reshape((-1, length, 1))
    return np.concatenate((x, y), -1).astype(np.float32)


def split_xy(value):
    return value[:, :, 0], value[:, :, 1]


def augment_sample(df, augs):
    df = df.copy()
    pose = combine_xy(df.pose_x, df.pose_y)
    h1 = combine_xy(df.hand1_x, df.hand1_y)
    h2 = combine_xy(df.hand2_x, df.hand2_y)
    input_df = pd.DataFrame.from_dict(
        {
            "uid": df.uid,
            "pose": pose.tolist(),
            "hand1": h1.tolist(),
            "hand2": h2.tolist(),
            "label": df.label,
        }
    )
    augmented_samples = []
    for augmentation in augs:
        df_augmented = augmentation(input_df)
        pose_x, pose_y = df_augmented.pose
        hand1_x, hand1_y = df_augmented.hand1
        hand2_x, hand2_y = df_augmented.hand2
        save_df = pd.DataFrame.from_dict(
            {
                "uid": df_augmented.uid + "_" + augmentation.__name__,
                "label": df_augmented.label,
                "pose_x": pose_x,
                "pose_y": pose_y,
                "hand1_x": hand1_x,
                "hand1_y": hand1_y,
                "hand2_x": hand2_x,
                "hand2_y": hand2_y,
                "n_frames": df.n_frames,
            }
        )
        augmented_samples.append(save_df)

    return pd.concat(augmented_samples, axis=0)


def preprocess(df, use_augs, label_map, mode):
    feature_cols = ["pose_x", "pose_y", "hand1_x", "hand1_y", "hand2_x", "hand2_y"]
    x, y = [], []
    i = 0
    if use_augs and mode == "train":
        pbar = tqdm(total=df.shape[0] * 7, desc=f"Processing {mode} file....")
    else:
        pbar = tqdm(total=df.shape[0], desc=f"Processing {mode} file....")
    while i < df.shape[0]:
        if use_augs and mode == "train":
            augs = [
                plus7rotation,
                minus7rotation,
                gaussSample,
                cutout,
                upsample,
                downsample,
            ]
            augmented_rows = augment_sample(df.iloc[i], augs)
            df = pd.concat([df, augmented_rows], axis=0)
        row = df.loc[i, feature_cols]
        flatten_features = np.hstack(list(map(flatten, row.values)))
        x.append(flatten_features)
        y.append(row.label)
        i += 1
        pbar.update(1)
    x = np.stack(x)
    y = np.array(y)
    return x, y


def fit(args):
    train_df = pd.read_json(
        os.path.join(args.data_dir, f"{args.dataset}_train_keypoints.json")
    )
    val_df = pd.read_json(
        os.path.join(args.data_dir, f"{args.dataset}_val_keypoints.json")
    )

    label_map = load_label_map(args.dataset)
    x_train, y_train = preprocess(train_df, args.use_augs, label_map, "train")
    x_val, y_val = preprocess(val_df, args.use_augs, label_map, "val")

    model = Xgboost(config=XgbConfig)
    model.fit(x_train, y_train, x_val, y_val)

    exp_name = get_experiment_name(args)
    save_path = os.path.join(args.save_dir, exp_name, ".pickle.dat")
    model.save(save_path)


def evaluate(args):
    test_df = pd.read_json(
        os.path.join(args.data_dir, f"{args.dataset}_test_keypoints.json")
    )

    label_map = load_label_map(args.dataset)
    x_test, y_test = preprocess(test_df, args.use_augs, label_map, "test")

    exp_name = get_experiment_name(args)
    model = Xgboost(config=XgbConfig)
    load_path = os.path.join(args.save_dir, exp_name, ".pickle.dat")
    model.load(load_path)
    print("### Model loaded ###")

    test_preds = model(x_test)
    print("Test accuracy:", accuracy_score(y_test, test_preds))
