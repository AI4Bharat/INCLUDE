import warnings
import argparse

import train_nn
import train_xgb
from cnn_runner import save_cnn_features

parser = argparse.ArgumentParser(
    description="INCLUDE trainer for xgboost, lstm and transformer"
)
parser.add_argument("--seed", default=0, type=int, help="seed value")
parser.add_argument(
    "--dataset", default="include", type=str, help="options: include or include50"
)
parser.add_argument(
    "--use_augs",
    action="store_true",
    help="use augmented data",
)
parser.add_argument(
    "--use_cnn",
    action="store_true",
    help="use mobilenet to convert keypoints to videos and generate embeddings from CNN",
)
parser.add_argument(
    "--model",
    default="lstm",
    type=str,
    help="options: lstm, transformer, xgboost",
)
parser.add_argument(
    "--data_dir",
    default="",
    type=str,
    required=True,
    help="location to train, val and test json files",
)
parser.add_argument(
    "--save_path",
    default="./",
    type=str,
    help="location to save trained model",
)
args = parser.parse_args()


if __name__ == "__main__":

    if args.model == "xgboost":
        if args.use_cnn:
            warnings.warn(
                "use_cnn flag set to true for xgboost model. xgboost will not use cnn features"
            )
        trainer = train_xgb

    else:
        if args.use_cnn:
            save_cnn_features(args)
            if args.use_augs:
                warnings.warn("cannot perform augmentation on cnn features")

        trainer = train_nn

    trainer.fit(args)
    trainer.evaluate(args)
