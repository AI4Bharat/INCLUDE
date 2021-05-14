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
parser.add_argument(
    "--epochs", default=50, type=int, help="number of epochs to train the model"
)
parser.add_argument("--batch_size", default=128, type=int, help="batch size of data")
parser.add_argument(
    "--learning_rate",
    default=1e-4,
    type=float,
    help="learning rate for training neural net",
)
parser.add_argument(
    "--transformer_size", default="small", type=str, help="options: small, large"
)
parser.add_argument(
    "--use_pretrained",
    default=None,
    help="use pretrained model. options: evaluate, resume_training",
)
args = parser.parse_args()


if __name__ == "__main__":

    if args.model == "xgboost":
        if args.use_pretrained:
            raise Exception("Pre-trained models are not available for XGBoost")
        if args.use_cnn:
            warnings.warn(
                "use_cnn flag set to true for xgboost model. xgboost will not use cnn features"
            )
        train_xgb.fit(args)
        train_xgb.evaluate(args)

    else:
        if args.use_cnn:
            save_cnn_features(args)
            if args.use_augs:
                warnings.warn("cannot perform augmentation on cnn features")
        if args.use_pretrained == "evaluate":
            train_nn.evaluate(args)
            print("###  Evaluated from pretrained model  ###")
        else:
            print("### Starting to train.  ###")
            train_nn.fit(args)
            train_nn.evaluate(args)
