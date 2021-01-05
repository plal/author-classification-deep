import torchvision
from torch import nn
import torch.nn.functional as f

class WordsModel(nn.Module):

    def __init__(self, model_name, out_feat=22, pretrained=False):
        super().__init__()

        self.model_name = model_name
        self.model      = None
        self.out_feat   = out_feat
        self.pretrained = pretrained
        self.init_model()

    def forward(self, x):
        return self.model(x)

    def update_weight(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def init_model(self):
        if '50' in self.model_name:
            self.model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif '34' in self.model_name:
            self.model = torchvision.models.resnet34(pretrained=self.pretrained)
        else:
            raise ValueError('The supported models are resnet[34,50]')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.eval()

        in_feat = {'resnet34':512, 'resnet50':2048 }
        clf = nn.Linear(in_features=in_feat[self.model_name], out_features=self.out_feat, bias=True)
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = clf
