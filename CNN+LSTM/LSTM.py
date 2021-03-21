import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from tqdm.auto import tqdm
import gc
import csv
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('include50')


#this is the LSTM code for the CNN+LSTM approach

def seed_everything(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


with open('DATA/LSTM-TRAIN.npy', 'rb') as f:
	X_train = np.load(f)
	f.close()

with open('DATA/LSTM-TEST.npy', 'rb') as f:
	X_test = np.load(f)
	f.close()

with open('DATA/TRAIN_LSTM_LABELS.npy', 'rb') as f:
	y_train = np.load(f)
	f.close()

with open('DATA/TEST_LSTM_LABELS.npy', 'rb') as f:
	y_test= np.load(f)
	f.close()

assert X_train.shape == (3475, 154, 1280)
assert X_test.shape == (817, 154, 1280)
assert y_train.shape[0] == 3475
assert y_test.shape[0] == 817

def encode(labels):
	y = labels.reshape((-1, 1))
	enc = LabelEncoder()
	y = enc.fit_transform(y)
	y = y.reshape((-1, 1))
	enc = OneHotEncoder()
	y = enc.fit_transform(y).toarray()
	return y

y_train = encode(y_train)
y_test = encode(y_test)
print(y_train.shape)
print(y_test.shape)
#
# # print(y_train.shape)
# # print(y_test.shape)

class AverageMeter:
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class EarlyStopping:
	def __init__(self, patience=5, mode="min", delta=0.0):
		self.patience = patience
		self.counter = 0
		self.mode = mode
		self.best_score = None
		self.early_stop = False
		self.delta = delta
		if self.mode == "min":
			self.val_score = np.Inf
		else:
			self.val_score = -np.Inf

	def __call__(self, model_path, epoch_score, model, optimizer, scheduler=None):

		if self.mode == "min":
			score = -1.0 * epoch_score
		else:
			score = np.copy(epoch_score)

		score_not_improved = False
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
		elif score <= self.best_score + self.delta:
			self.counter += 1
			score_not_improved = True
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
			self.counter = 0

		return score_not_improved

	def save_checkpoint(self, epoch_score, model, optimizer, scheduler, model_path):
		if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
			print(
				"Validation score improved ({} --> {}). Saving model!".format(
					self.val_score, epoch_score
				)
			)
			torch.save(
				{
					"model": model.state_dict(),
					"optimizer": optimizer.state_dict(),
					"scheduler": scheduler.state_dict() if scheduler else scheduler,
					"score": epoch_score,
				},
				model_path,
			)
		self.val_score = epoch_score


class IncludeDataset(torch.utils.data.Dataset):
	def __init__(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def __getitem__(self, idx):
		return {
			"data": self.x_train[idx].astype(np.float32),
			"label": np.argmax(self.y_train[idx]),
		}

	def __len__(self):
		return len(self.x_train)

class IncludeModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=1280,
			hidden_size=128,
			num_layers=1,
			batch_first=True,
			bidirectional=True,
		)
		self.l1 = nn.Linear(in_features=256, out_features=263)

	def forward(self, x):
		x, (_, _) = self.lstm(x)
		x = torch.max(x, dim=1).values
		x = F.dropout(x, p=0.3)
		x = self.l1(x)
		return x

def calc_metric(preds, targets):
	preds = (
		torch.argmax(torch.softmax(preds, dim=-1).detach(), dim=-1).cpu().numpy()
	)
	acc_score = accuracy_score(targets.cpu().numpy(), preds)
	return acc_score

def loss_fct(preds, targets):
	criterion = nn.CrossEntropyLoss()
	loss = criterion(preds, targets)
	return loss

def train(dataloader, model, optimizer, device):
	model.train()
	pbar = tqdm(dataloader, total=len(dataloader), desc="Training")
	losses = AverageMeter()
	acc_score = AverageMeter()

	for batch in pbar:
		inputs = batch["data"].to(device)
		targets = batch["label"].to(device)

		optimizer.zero_grad()
		preds = model(inputs)
		loss = loss_fct(preds, targets)
		loss.backward()
		optimizer.step()

		losses.update(loss.item())
		acc_score.update(calc_metric(preds, targets))
		pbar.set_postfix(loss=losses.avg, accuracy=acc_score.avg)

	torch.cuda.empty_cache()
	gc.collect()
	losses_avg = losses.avg
	acc_avg = acc_score.avg
	del acc_score, losses
	return losses_avg, acc_avg

@torch.no_grad()
def val(dataloader, model, device):
	model.eval()
	pbar = tqdm(dataloader, total=len(dataloader), desc="Training")
	losses = AverageMeter()
	acc_score = AverageMeter()

	for batch in pbar:
		inputs = batch["data"].to(device)
		targets = batch["label"].to(device)

		preds = model(inputs)
		loss = loss_fct(preds, targets)

		losses.update(loss.item())
		acc_score.update(calc_metric(preds, targets))
		pbar.set_postfix(loss=losses.avg, accuracy=acc_score.avg)

	torch.cuda.empty_cache()
	gc.collect()
	losses_avg = losses.avg
	acc_avg = acc_score.avg
	del acc_score, losses
	return losses_avg, acc_avg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
lr=1e-3

train_dataset = IncludeDataset(X_train, y_train)
val_dataset = IncludeDataset(X_test, y_test)
train_dataloader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=batch_size,
	shuffle=True,
	num_workers=0,
	pin_memory=False,
)
val_dataloader = torch.utils.data.DataLoader(
	val_dataset,
	batch_size=batch_size,
	shuffle=False,
	num_workers=0,
	pin_memory=False,
)

model = IncludeModel().to(device)
optimizer =  torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, mode="max", patience=3, verbose=True
)
es = EarlyStopping(mode="max")

for epoch in range(epochs):
	print(f"Epoch: {epoch+1}/{epochs}")
	_, _ = train(train_dataloader, model, optimizer, device)
	eval_loss, eval_acc = val(val_dataloader, model, device)
	scheduler.step(eval_acc)
	es("lstm_torch.pth", eval_acc, model, optimizer, scheduler)
	if es.early_stop:
		print("Early Stopping")
		break
