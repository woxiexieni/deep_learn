import torch
import torch.nn as nn
class EnsembleClassifier(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleClassifier, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=1)