import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from generate_keypoints import process_video
from models import Transformer
from configs import TransformerConfig
from utils import load_json, load_label_map
import shutil

parser = argparse.ArgumentParser(description="Evaluate function")
parser.add_argument("--data_dir", required=True, help="data directory")
args = parser.parse_args()


class KeypointsDataset(data.Dataset):
    def __init__(
        self,
        keypoints_dir,
        max_frame_len=200,
        frame_length=1080,
        frame_width=1920,
    ):
        self.files = sorted(glob.glob(os.path.join(keypoints_dir, "*.json")))
        self.max_frame_len = max_frame_len
        self.frame_length = frame_length
        self.frame_width = frame_width

    def interpolate(self, arr):

        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        if np.count_nonzero(~np.isnan(arr_x)) == 0:
            arr_x = np.zeros(arr_x.shape)
        if np.count_nonzero(~np.isnan(arr_y)) == 0:
            arr_y = np.zeros(arr_y.shape)

        arr_x = arr_x * self.frame_width
        arr_y = arr_y * self.frame_length

        return np.stack([arr_x, arr_y], axis=-1)

    def combine_xy(self, x, y):
        x, y = np.array(x), np.array(y)
        _, length = x.shape
        x = x.reshape((-1, length, 1))
        y = y.reshape((-1, length, 1))
        return np.concatenate((x, y), -1).astype(np.float32)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        row = pd.read_json(file_path, typ="series")
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
        }

    def __len__(self):
        return len(self.files)


@torch.no_grad()
def inference(dataloader, model, device, label_map):
    model.eval()
    predictions = []

    for batch in tqdm(dataloader, desc="Eval"):
        input_data = batch["data"].to(device)
        output = model(input_data).detach().cpu()
        output = torch.argmax(torch.softmax(output, dim=-1), dim=-1).numpy()
        predictions.append({"uid": batch["uid"][0], "predicted_label": label_map[output[0]]})

    return predictions


video_paths = glob.glob(os.path.join(args.data_dir, "*"))
save_dir = "keypoints_dir"
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
for path in tqdm(video_paths, desc="Processing Videos"):
    process_video(path, save_dir)

label_map = load_label_map("include")
dataset = KeypointsDataset(
    keypoints_dir=save_dir,
    max_frame_len=169,
)

dataloader = data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
label_map = dict(zip(label_map.values(), label_map.keys()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = TransformerConfig(size="large", max_position_embeddings=256)
model = Transformer(config=config, n_classes=263)
model = model.to(device)

pretrained_model_name = "include_no_cnn_transformer_large.pth"
pretrained_model_links = load_json("pretrained_links.json")
if not os.path.isfile(pretrained_model_name):
    link = pretrained_model_links[pretrained_model_name]
    torch.hub.download_url_to_file(link, pretrained_model_name, progress=True)

ckpt = torch.load(pretrained_model_name)
model.load_state_dict(ckpt["model"])
print("### Model loaded ###")

preds = inference(dataloader, model, device, label_map)
print(json.dumps(preds, indent=2))
