import torch
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F


class ConvNetWithCentroids(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.centroids = nn.parameter.Parameter(data=torch.zeros((10, 2)).float(), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # torch.nn.init.kaiming_normal_(self, )

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = - torch.square(torch.linalg.norm(x.reshape(-1, 10, 2)-self.centroids, dim=-1, keepdim=True)) / 2
        x = F.softmax(x, dim=1).squeeze()
        return x, self.centroids

# Net = torchvision.models.resnet18(pretrained=True)
