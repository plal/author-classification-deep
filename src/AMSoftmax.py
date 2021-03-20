import torch.nn as nn
import numpy as np
import torch

class AdditiveMarginSoftmax(nn.Module):
    # AMSoftmax
    def __init__(self, margin=0.35, s=30):
        super().__init__()

        self.m = margin #
        self.s = s
        self.epsilon = 0.000000000001
        print('AMSoftmax m = ' + str(margin))

    def forward(self, predicted, target):

        # ------------ AM Softmax ------------ #
        predicted = predicted / (predicted.norm(p=2, dim=0) + self.epsilon)
        indexes = range(predicted.size(0))
        cos_theta_y = predicted[indexes, target]
        cos_theta_y_m = cos_theta_y - self.m
        exp_s = np.e ** (self.s * cos_theta_y_m)

        sum_cos_theta_j = (np.e ** (predicted * self.s)).sum(dim=1) - (np.e ** (predicted[indexes, target] * self.s))

        log = -torch.log(exp_s/(exp_s+sum_cos_theta_j+self.epsilon)).mean()

        return log
