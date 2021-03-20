import torch
import torch.nn as nn

class AMSoftmaxClassifier(nn.Module): # substitui a nn.linear dentro do modulo
    def __init__(self,
                 in_feats,
                 n_classes=23,
                 m=0.3,
                 s=15):
        super(AMSoftmaxClassifier, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        return costh


class AMSoftmaxLoss(nn.Module): # substitui a nn.crossentropy no loop de treinamento
    def __init__(self,
                 m=0.3,
                 s=15):
        super(AMSoftmaxLoss, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, lb):
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(x.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = x - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss
