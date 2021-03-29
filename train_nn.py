import os
import logging

import torch
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from models import CNN, LSTM, Transformer
from configs import CnnConfig, LstmConfig, TransformerConfig
from utils import (
    seed_everything,
    AverageMeter,
    EarlyStopping,
    load_label_map,
    get_experiment_name,
)
from dataset import KeypointsDataset, FeaturesDatset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader, model, optimizer, device):
    model.train()

    losses = AverageMeter()
    accuracy = AverageMeter()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_data = batch["data"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        preds = model(input_data)
        loss = F.cross_entropy(preds, label)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        accuracy.update(
            accuracy_score(
                label.cpu().numpy(),
                torch.argmax(torch.softmax(preds.detach()).cpu().numpy()),
            )
        )
        pbar.set_postfix(loss=losses.avg, accuracy=accuracy.avg)

    torch.cuda.empty_cache()
    loss_avg = losses.avg
    accuracy_avg = accuracy.avg
    return loss_avg, accuracy_avg


@torch.no_grad()
def validate(dataloader, model, device):
    model.eval()

    losses = AverageMeter()
    accuracy = AverageMeter()

    pbar = tqdm(dataloader, desc="Eval")
    for batch in pbar:
        input_data = batch["data"].to(device)
        label = batch["label"].to(device)

        preds = model(input_data)
        loss = F.cross_entropy(preds, label)

        losses.update(loss.item())
        accuracy.update(
            accuracy_score(
                label.cpu().numpy(),
                torch.argmax(torch.softmax(preds.detach()).cpu().numpy()),
            )
        )
        pbar.set_postfix(loss=losses.avg, accuracy=accuracy.avg)

    torch.cuda.empty_cache()
    loss_avg = losses.avg
    accuracy_avg = accuracy.avg
    return loss_avg, accuracy_avg


def fit(args):
    exp_name = get_experiment_name(args)
    logging.basicConfig(
        filename=f"{exp_name}.log", level=logging.INFO, format="%(message)s"
    )
    seed_everything(args.seed)
    label_map = load_label_map(args.dataset)

    if args.use_cnn:
        train_dataset = FeaturesDatset(
            features_dir=os.path.join(args.data_dir, f"{args.dataset}_train_features"),
            label_map=label_map,
            mode="train",
        )
        val_dataset = FeaturesDatset(
            features_dir=os.path.join(args.data_dir, f"{args.dataset}_val_features"),
            label_map=label_map,
            mode="val",
        )

    else:
        train_dataset = KeypointsDataset(
            keypoints_path=os.path.join(
                args.data_dir, f"{args.dataset}_train_keypoints.json"
            ),
            use_augs=args.use_augs,
            label_map=label_map,
            mode="train",
        )
        val_dataset = KeypointsDataset(
            keypoints_path=os.path.join(
                args.data_dir, f"{args.dataset}_val_keypoints.json"
            ),
            use_augs=False,
            label_map=label_map,
            mode="val",
        )

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    n_classes = 50
    if args.dataset == "include":
        n_classes = 263

    if args.model == "lstm":
        config = LstmConfig()
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = LSTM(config=config, n_classes=n_classes)
    else:
        config = TransformerConfig(size=args.size)
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = Transformer(config=config, n_classes=n_classes)

    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate)

    model_path = os.path.join(args.save_path, exp_name)
    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1}/{args.epochs}")
        train_loss, train_acc = train(train_dataloader, model, optimizer, device)
        val_loss, val_acc = validate(val_dataloader, model, device)
        logging.info(
            "Epoch: {}, train loss: {}, train acc: {}, val loss: {}, val acc: {}".format(
                epoch + 1, train_loss, train_acc, val_loss, val_acc
            )
        )
        es(
            model_path=model_path,
            epoch_score=val_acc,
            model=model,
            optimizer=optimizer,
        )
        if es.early_stop:
            print("Early stopping")
            break


def evaluate(args):
    label_map = load_label_map(args.dataset)
    n_classes = 50
    if args.dataset == "include":
        n_classes = 263

    if args.use_cnn:
        dataset = FeaturesDatset(
            features_dir=os.path.join(args.data_dir, f"{args.dataset}_test_features"),
            label_map=label_map,
            mode="test",
        )

    else:
        dataset = KeypointsDataset(
            keypoints_path=os.path.join(
                args.data_dir, f"{args.dataset}_test_keypoints.json"
            ),
            use_augs=False,
            label_map=label_map,
            mode="test",
        )

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    if args.model == "lstm":
        config = LstmConfig()
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = LSTM(config=config, n_classes=n_classes)
    else:
        config = TransformerConfig(size=args.size)
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = Transformer(config=config, n_classes=n_classes)

    model = model.to(device)

    exp_name = get_experiment_name(args)
    model_path = os.path.join(args.save_path, exp_name, ".pth")
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"])
    print("### Model loaded ###")

    test_loss, test_acc = validate(dataloader, model, device)
    print("Evaluation Results:\n")
    print(f"Loss: {test_loss}, Accuracy: {test_acc}")
