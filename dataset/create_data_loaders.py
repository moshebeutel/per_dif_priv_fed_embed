from typing import Callable

import torch.utils.data
import torchvision
from torchvision.transforms import transforms

from common.config import Config


class DatasetsCreationRepository:
    def __init__(self):
        self._repository = {'cifar10': torchvision.datasets.CIFAR10}

    def register(self, dataset_name: str, create_fn: Callable[..., torch.utils.data.Dataset]) -> None:
        self._repository[dataset_name] = create_fn

    def contains(self, dataset_name: str):
        return dataset_name in self._repository.keys()

    def get_dataset_creator(self, dataset_name: str) -> Callable[..., torch.utils.data.Dataset]:
        assert self.contains(dataset_name)
        return self._repository[dataset_name]


def get_data_loaders(dataset_name: str, batch_size: int):
    repo = DatasetsCreationRepository()
    dataset_fn = repo.get_dataset_creator(dataset_name=dataset_name)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dataset_fn(root=Config.DATASETS_ROOT_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=Config.DATASETS_ROOT_PATH, train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


