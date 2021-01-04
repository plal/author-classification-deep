from src.data import WordsDataset
from src.model import WordsModel

import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import argparse

#consts
OUT_FEAT      = 22
ROOT_DIR      = r'words/png_files/'
TRAIN_SUB_DIR = r'train/'
VAL_SUB_DIR   = r'val/'


ap = argparse.ArgumentParser()
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

#training hparams
seed       = args["seed"]
batch_size = args["batch_size"]
max_epoch  = args["max_epoch"]
early_stop = args["early_stop"]
lr         = args["lr"]

#model hparams
model_name  = args["model"]
out_feat    = OUT_FEAT
start_epoch = 0

model = WordsModel(
    model_name = model_name,
    out_feat   = out_feat
)

# print(model)

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
