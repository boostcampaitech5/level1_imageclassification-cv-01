import torch
import torch.nn as nn
import torch.nn.functional as F



#class FocalLoss(nn.Module):

#class LabelSmoothingLoss(nn.Module):
    
#class F1Loss(nn.Module):


class CrossEntropyLossWithClassBalancing(nn.Module):
    def __init__(self, num_classes=18, weight=torch.tensor([0.7, 0.6, 1.1, 0.5, 0.5, 1.1, 1.1, 1.1, 1.3, 1.1, 1.1, 1.3, 1.1, 1.1, 1.3, 1.1, 1.1, 1.3])):
        super(CrossEntropyLossWithClassBalancing, self).__init__()
        self.num_classes = num_classes
        if weight is None:
            self.weight = torch.ones(num_classes).cuda()
        else:
            self.weight = weight.cuda()

    def forward(self, inputs, targets):
    
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)

        return ce_loss

_criterion_entrypoints = {
    'cross_entropy_class_balancing' : CrossEntropyLossWithClassBalancing
}

