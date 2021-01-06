from src.data import WordsDataset
from src.model import WordsModel
from src.metrics import compute_metrics

import matplotlib.pyplot as plt
import torchvision
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm as tqdm

#consts
OUT_FEAT      = 23
ROOT_DIR      = r'words/png_files/'
TRAIN_SUB_DIR = r'train/'
VAL_SUB_DIR   = r'val/'


ap = argparse.ArgumentParser()
ap.add_argument('-e','--exp_name',required=True,type=str,
                help="Experiment ID")
ap.add_argument('-m','--model',required=True,type=str,
                help="Model to run")
ap.add_argument('-b','--batch_size',default=16,type=int,
                help="Batch size")
ap.add_argument('--max_epoch',default=100,type=int,
                help="Max number of epochs to train")
ap.add_argument('--early_stop',default=10,type=int,
                help='Interrupt training if model doesnt improve in n epochs')
ap.add_argument('--lr',default=0.00001,type=float,
                help='Lerning rate')
ap.add_argument('--seed',default=2021,type=int,
                help='Rnd seed')
args = vars(ap.parse_args())

#general
exp_name = args["exp_name"]

#training hparams
seed          = args["seed"]
batch_size    = args["batch_size"]
max_epoch     = args["max_epoch"]
early_stop    = args["early_stop"]
lr            = args["lr"]
loss_function = 'CrossEntropyLoss'

#model hparams
model_name  = args["model"]
out_feat    = OUT_FEAT
start_epoch = 0

model = WordsModel(
    model_name = model_name,
    out_feat   = out_feat
)

# print(model)

save_dir = f"runs/{exp_name}_{model_name}"
(Path(save_dir) / 'checkpoints').mkdir(parents=True, exist_ok=True)
(Path(save_dir) / 'logs').mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(save_dir+'/logs/')

optimizer = optim.Adam(model.parameters(), lr=lr)

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

fix_seeds(seed)

def validation_epoch(model, data_loader, epoch=0, is_cuda=True):
    model.eval()
    if is_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    epoch_outputs = []
    epoch_labels  = []

    with torch.no_grad():

        for i, data in enumerate(tqdm(data_loader)):

            inputs, labels, _ = data
            if is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)

            epoch_outputs.append(outputs.detach().cpu().numpy())
            epoch_labels.append(labels.detach().cpu().numpy())

            loss = criterion(outputs,labels)

        epoch_outputs_torch = torch.tensor(np.vstack(epoch_outputs))
        epoch_labels_torch  = torch.tensor(np.hstack(epoch_labels))

        # print(epoch_outputs_torch)
        # print(epoch_labels_torch)

        loss_validation = criterion(epoch_outputs_torch, epoch_labels_torch)
        acc_validation, f1_validation = compute_metrics(epoch_outputs_torch, epoch_labels_torch)

        writer.add_scalar('loss/validation', loss_validation, epoch)
        writer.add_scalar('acc/validation', acc_validation, epoch)
        writer.add_scalar('f1/validation', f1_validation, epoch)

        print(f'Validation set --> loss: {loss_validation:.4f} // acc: {acc_validation:.4f} // f1: {f1_validation:.4f}')

        return f1_validation, dict(epoch=epoch, loss_validation=loss_validation.item(), acc_validation=acc_validation.item(), f1_validation=f1_validation.item())

def train_epoch(model, data_loader, epoch=0, is_cuda=True):
    model.train()
    if is_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    epoch_outputs = []
    epoch_labels  = []

    for i, data in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()

        inputs, labels, _ = data

        if is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)

        epoch_outputs.append(outputs.detach().cpu().numpy())
        epoch_labels.append(labels.detach().cpu().numpy())

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    epoch_outputs_torch = torch.tensor(np.vstack(epoch_outputs))
    epoch_labels_torch  = torch.tensor(np.hstack(epoch_labels))

    # print(epoch_outputs_torch)
    # print(epoch_labels_torch)

    loss_train = criterion(epoch_outputs_torch, epoch_labels_torch)
    acc_train, f1_train = compute_metrics(epoch_outputs_torch, epoch_labels_torch)

    writer.add_scalar('loss/train', loss_train, epoch)
    writer.add_scalar('acc/train', acc_train, epoch)
    writer.add_scalar('f1/train', f1_train, epoch)

    print(f'Training (epoch {epoch}/{max_epoch}) --> loss: {loss_train:.4f} // acc: {acc_train:.4f} // f1: {f1_train:.4f}')

    return f1_train, dict(epoch=epoch, loss_train=loss_train.item(), acc_train=acc_train.item(), f1_train=f1_train.item())

criterion = nn.CrossEntropyLoss()

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, ))
])

train_data = WordsDataset(ROOT_DIR, TRAIN_SUB_DIR, transform=transforms)
val_data   = WordsDataset(ROOT_DIR, VAL_SUB_DIR, transform=transforms)
# print(len(train_data))
# print(len(val_data))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

inputs, labels, names = iter(val_loader).next()
# print(names)
# print(labels)
print(f'validation set information -> input shape: {inputs.shape} // labels shape: {labels.shape} // size: {len(val_loader)} batches')
inputs, labels, names = iter(train_loader).next()
print(f'train set information -> input shape: {inputs.shape} // labels shape: {labels.shape} // size: {len(train_loader)} batches')


#TRAINING MODEL
epoch = -1
best_f1, _ = validation_epoch(model, val_loader, epoch, is_cuda=True)
epoch_since_best = 0

history = []

for epoch in range(start_epoch, max_epoch):
    if epoch_since_best > early_stop:
        print("early stop activated!")
        break

    _, train_metrics = train_epoch(model, train_loader, epoch, is_cuda=True)
    if epoch == 0:
        model.update_weight()
    current_f1, val_metrics = validation_epoch(model, val_loader, epoch, is_cuda=True)

    history.append({**train_metrics, **val_metrics})
    df = pd.DataFrame(history)
    df.to_csv(Path(save_dir) / 'history.csv', index=False)

    if current_f1 > best_f1:
        best_f1 = current_f1

        model_file_name = f'{model.__class__.__name__}_{epoch}_{best_f1:.4f}'
        save_name       = f'{save_dir}/checkpoints/{model_file_name}.pt'

        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'best_f1':best_f1,
            'loss_function':loss_function,
            'out_feat':out_feat,
            'model_name':model_name
        }, save_name)

        best = df.loc[df.f1_validation.idxmax()].to_dict()
        (Path(save_dir) / 'summary.yaml').write_text(yaml.dump(best))

        epoch_since_best = 0
    else:
        epoch_since_best += 1
