import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpNet(nn.Module):
    def __init__(self, return_logits=False):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 120)   # 3072  => 120
        self.fc2 = nn.Linear(120, 84)            # 120   => 84
        self.fc3 = nn.Linear(84, 10)             # 84    => 10
        self._return_logits = return_logits
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

        # torch.nn.init.kaiming_normal_(self, )

        # if isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=1.0)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        # elif isinstance(module, nn.Linear):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1) if not self._return_logits else x
        return x

# Net = torchvision.models.resnet18(pretrained=True)
