from src.data import WordsDataset
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader


root_dir   = r'words/png_files/'
train_sub_folder = r'train/'
val_sub_folder   = r'val/'

train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, ))
])

train_data = WordsDataset(root_dir, train_sub_folder, transform=train_transforms)
val_data   = WordsDataset(root_dir, val_sub_folder, transform=train_transforms)
print(len(train_data))
print(len(val_data))

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

inputs, labels, names = iter(val_loader).next()
# print(names)
# print(labels)
print(f'validation set information -> input shape: {inputs.shape} // labels shape: {labels.shape} // size: {len(val_loader)} batches')
inputs, labels, names = iter(train_loader).next()
print(f'train set information -> input shape: {inputs.shape} // labels shape: {labels.shape} // size: {len(train_loader)} batches')
