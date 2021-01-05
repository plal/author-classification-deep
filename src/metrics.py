import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve

import matplotlib.pyplot as plt

def compute_metrics(outputs:torch.Tensor, labels:torch.Tensor):

    outputs = outputs.argmax(dim=1)

    outputs_np = outputs.cpu().numpy()
    labels_np  = labels.cpu().numpy()

    acc_val = accuracy_score(labels_np, outputs_np)
    f1_val  = f1_score(labels_np, outputs_np, average='weighted')

    return acc_val, f1_val
