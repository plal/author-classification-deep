import yaml
import torch
import torchvision
import argparse
import numpy as np
from pathlib import Path
from einops import asnumpy
from tqdm import tqdm as tqdm
import pandas as pd

import torch.optim as optim
import torch.nn.functional as f
from torch import nn
from torchvision import transforms
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader

from src.data import WordsDataset
from src.metrics import compute_metrics
from src.model import WordsModel

#consts
ROOT_DIR      = r'words/png_files/'
TEST_SUB_DIR = r'test/'

def test_epoch(model, data_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    all_outputs = []
    all_labels  = []
    # all_names   = []

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(tqdm(data_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            all_outputs.append(asnumpy(outputs))
            all_labels.append(asnumpy(labels))

            loss = criterion(outputs, labels)

        y_pred = np.concatenate(all_outputs)
        y_true = np.concatenate(all_labels)

        y_pred = torch.tensor(np.vstack(y_pred))
        y_true  = torch.tensor(np.hstack(y_true))

        test_loss = criterion(y_pred, y_true)
        test_acc, test_f1 = compute_metrics(y_pred, y_true)

        print(f'Test set --> loss: {test_loss:.4f} // acc: {test_acc:.4f} // f1: {test_f1:.4f}')

        return test_f1, dict(loss_test=test_loss, acc_test=test_acc, f1_test=test_f1)

ap = argparse.ArgumentParser()
ap.add_argument('-ckpt','--checkpoint', required=True, type=str,
                help='Path to checkpoint being tested')
ap.add_argument('-b','--batch_size', required=True, type=int,
                help='Batch size')

args = vars(ap.parse_args())

ckpt_path  = args["checkpoint"]
batch_size = args["batch_size"]

ckpt = torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

model_name    = ckpt["model_name"]
loss_function = ckpt["loss_function"]
out_feat      = ckpt["out_feat"]
best_f1_train = ckpt["best_f1"]

model = WordsModel(
    model_name = model_name,
    out_feat   = out_feat
)

model.load_state_dict(ckpt["model_state_dict"])

if loss_function == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, ))
])

test_data = WordsDataset(ROOT_DIR, TEST_SUB_DIR, transform=transforms)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

inputs, labels, _ = iter(test_loader).next()
print(f'Test set information -> input shape: {inputs.shape} // labels shape: {labels.shape} // size: {len(test_loader)} batches')

save_dir = (Path(ckpt_path).parents[1] / 'test')
save_dir.mkdir(parents=True, exist_ok=True)

f1_test, test_metrics = test_epoch(model, test_loader)

with open(save_dir / 'test-metrics.yaml', 'w') as outfile:
    yaml.dump(test_metrics, outfile, default_flow_style=False)

print(f'Test metrics saved in model folder: {save_dir}.')

# print(model_name, loss_function, out_feat, best_f1_train)
