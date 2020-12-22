import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import PIL
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

class_dict = {
    'c03a':0,'c03b':1,'c03c':2,'c03d':3,'c03e':4,'c03f':5,
    'g06a':6,'g06b':7,'g06c':8,'g06d':9,'g06e':10,'g06f':11,
    'g06g':12,'g06h':13,'g06i':14,'g06j':15,'g06k':16,'g06l':17,
    'g06m':18,'g06n':19,'g06o':20,'g06p':21,'g06r':22
}

class WordsDataset(Dataset):

    def __init__(self, root_dir, sub_folder=r"train/", transform=None):
        self.root_dir   = root_dir
        self.sub_folder = sub_folder
        self.data_path  = os.path.join(self.root_dir, self.sub_folder)
        self.data       = os.listdir(self.data_path)
        self.transform  = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]
        image_path = os.path.join(self.data_path, image_name)
        image = PIL.Image.open(image_path)
        image_name_tokens = image_name.split('-')
        label_raw = image_name_tokens[0]+image_name_tokens[1][-1]
        label = class_dict[label_raw]

        if self.transform:
            image = self.transform(image)
        return (image, label, image_name)
