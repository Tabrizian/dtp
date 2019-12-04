from dtp.datasets.base_dataset import BaseDataset

import torchvision.transforms as transforms
import torchvision
import torch


class MNIST(BaseDataset):
    def __init__(self, sizes, location='./data'):
        self.location = location
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.data = torchvision.datasets.MNIST(
            root=location,
            train=True,
            download=True,
            transform=self.transformer
        )

        testset = torchvision.datasets.MNIST(root=location, train=False, download=True, transform=self.transformer)
        self.test = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        self.fake = torchvision.datasets.MNIST(
            root=location,
            train=True,
            download=True,
            transform=self.transformer
        )
        self.fake.targets[self.fake.targets == 9] = 8

        super().__init__(self.data, sizes)

