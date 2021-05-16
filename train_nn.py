import os
import logging
import json

import torch
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models import CNN, LSTM, Transformer
from configs import CnnConfig, LstmConfig, TransformerConfig
from utils import (
    seed_everything,
    AverageMeter,
    EarlyStopping,
    load_label_map,
    get_experiment_name,
    load_json,
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
        preds = preds.detach().cpu()
        accuracy.update(
            accuracy_score(
                label.cpu().numpy(),
                torch.argmax(torch.softmax(preds, dim=-1), dim=-1).numpy(),
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
        preds = preds.detach().cpu()
        accuracy.update(
            accuracy_score(
                label.cpu().numpy(),
                torch.argmax(torch.softmax(preds, dim=-1), dim=-1).numpy(),
            )
        )
        pbar.set_postfix(loss=losses.avg, accuracy=accuracy.avg)

    torch.cuda.empty_cache()
    loss_avg = losses.avg
    accuracy_avg = accuracy.avg
    return loss_avg, accuracy_avg


def change_max_pos_embd(args, new_mpe_size, n_classes):
    config = TransformerConfig(
        size=args.transformer_size, max_position_embeddings=new_mpe_size
    )
    if args.use_cnn:
        config.input_size = CnnConfig.output_dim
    model = Transformer(config=config, n_classes=n_classes)
    model = model.to(device)
    return model


def pretrained_name(args):
    load_modelName = args.dataset
    if args.use_cnn:
        load_modelName += "_use_cnn"
    else:
        load_modelName += "_no_cnn"
    if args.model == "lstm":
        load_modelName += "_lstm.pth"
    elif args.model == "transformer":
        load_modelName += "_transformer"
        if args.transformer_size == "large":
            load_modelName += "_large.pth"
        elif args.transformer_size == "small":
            load_modelName += "_small.pth"
    return load_modelName


def load_pretrained(args, n_classes, model, optimizer=None, scheduler=None):
    load_modelName = pretrained_name(args)
    pretrained_model_links = load_json("pretrained_links.json")

    if not os.path.isfile(load_modelName):
        link = pretrained_model_links[load_modelName]
        torch.hub.download_url_to_file(link, load_modelName, progress=True)

    if args.model == "transformer":
        model = change_max_pos_embd(args, new_mpe_size=256, n_classes=n_classes)

    ckpt = torch.load(load_modelName)
    model.load_state_dict(ckpt["model"])
    if args.use_pretrained == "resume_training":
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    return model, optimizer, scheduler


def fit(args):
    exp_name = get_experiment_name(args)
    logging_path = os.path.join(args.save_path, exp_name) + ".log"
    logging.basicConfig(filename=logging_path, level=logging.INFO, format="%(message)s")
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
            keypoints_dir=os.path.join(
                args.data_dir, f"{args.dataset}_train_keypoints"
            ),
            use_augs=args.use_augs,
            label_map=label_map,
            mode="train",
            max_frame_len=169,
        )
        val_dataset = KeypointsDataset(
            keypoints_dir=os.path.join(args.data_dir, f"{args.dataset}_val_keypoints"),
            use_augs=False,
            label_map=label_map,
            mode="val",
            max_frame_len=169,
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
        config = TransformerConfig(size=args.transformer_size)
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = Transformer(config=config, n_classes=n_classes)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2
    )

    if args.use_pretrained == "resume_training":
        model, optimizer, scheduler = load_pretrained(
            args, n_classes, model, optimizer, scheduler
        )

    model_path = os.path.join(args.save_path, exp_name) + ".pth"
    es = EarlyStopping(patience=15, mode="max")
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1}/{args.epochs}")
        train_loss, train_acc = train(train_dataloader, model, optimizer, device)
        val_loss, val_acc = validate(val_dataloader, model, device)
        logging.info(
            "Epoch: {}, train loss: {}, train acc: {}, val loss: {}, val acc: {}".format(
                epoch + 1, train_loss, train_acc, val_loss, val_acc
            )
        )
        scheduler.step(val_acc)
        es(
            model_path=model_path,
            epoch_score=val_acc,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if es.early_stop:
            print("Early stopping")
            break

    print("### Training Complete ###")


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
            keypoints_dir=os.path.join(args.data_dir, f"{args.dataset}_test_keypoints"),
            use_augs=False,
            label_map=label_map,
            mode="test",
            max_frame_len=169,
        )

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.model == "lstm":
        config = LstmConfig()
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = LSTM(config=config, n_classes=n_classes)
    else:
        config = TransformerConfig(size=args.transformer_size)
        if args.use_cnn:
            config.input_size = CnnConfig.output_dim
        model = Transformer(config=config, n_classes=n_classes)

    model = model.to(device)

    if args.use_pretrained == "evaluate":
        model, _, _ = load_pretrained(args, n_classes, model)
        print("### Model loaded ###")

    else:
        exp_name = get_experiment_name(args)
        model_path = os.path.join(args.save_path, exp_name) + ".pth"
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["model"])
        print("### Model loaded ###")

    test_loss, test_acc = validate(dataloader, model, device)
    print("Evaluation Results:")
    print(f"Loss: {test_loss}, Accuracy: {test_acc}")
