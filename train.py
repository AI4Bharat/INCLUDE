import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch_runner as T
import os
import glob
from dataset import IncludeDataset
from model import LSTM, Transformer
import click
from sklearn.metrics import accuracy_score


class Trainer(T.TrainerModule):
    def __init__(
        self,
        model,
        optimizer,
        experiment_name,
        device,
        early_stop=False,
        early_stop_metric="accuracy",
        early_stop_params={"patience": 8, "mode": "max", "delta": 0.0},
        seed=0,
    ):
        super(Trainer, self).__init__(
            model=model,
            optimizer=optimizer,
            experiment_name=experiment_name,
            early_stop=early_stop,
            early_stop_metric=early_stop_metric,
            early_stop_params=early_stop_params,
            device=device,
            seed=seed,
        )

    def calc_metric(self, preds, targets):
        preds = (
            torch.argmax(torch.softmax(preds, dim=-1).detach(), dim=-1).cpu().numpy()
        )
        acc_score = accuracy_score(targets.cpu().numpy(), preds)
        return acc_score

    def loss_fct(self, preds, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, targets)
        return loss

    def train_one_step(self, batch, batch_id):
        input = batch["data"].to(self.device)
        targets = batch["label"].to(self.device)

        self.optimizer.zero_grad()
        preds = self.model(input)
        loss = self.loss_fct(preds, targets)
        loss.backward()
        self.optimizer.step()
        acc_score = self.calc_metric(preds, targets)

        return {"loss": loss.item(), "accuracy": acc_score}

    def valid_one_step(self, batch, batch_id):
        input = batch["data"].to(self.device)
        targets = batch["label"].to(self.device)

        preds = self.model(input)
        loss = self.loss_fct(preds, targets)
        acc_score = self.calc_metric(preds, targets)
        return {"loss": loss.item(), "accuracy": acc_score}


@click.command()
@click.option("--use_augs", is_flag=True, help="use augmentations when training")
@click.option("--model_name", required=True, help="lstm or transformer")
@click.option("--exp_name", default="lstm_model", help="exp name")
@click.option("--epochs", default=25, help="number of epochs")
@click.option("--batch_size", default=64, help="batch size")
def main(use_augs, model_name, exp_name, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = IncludeDataset(
        df_path="xgb_train.csv", use_augs=use_augs, mode="train"
    )
    val_dataset = IncludeDataset(df_path="xgb_test.csv", use_augs=use_augs, mode="val")
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if model_name == "lstm":
        model = LSTM().to(device)
    else:
        model = Transformer().to(device)

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
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        experiment_name=exp_name,
        device=device,
    )
    trainer.fit(train_dataloader, val_dataloader, batch_size=batch_size, epochs=epochs)


if __name__ == "__main__":
    main()