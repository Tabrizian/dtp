from dtp.datasets.base_dataset import BaseDataset

import torchvision.transforms as transforms
import torchvision
import torch


class CIFAR10(BaseDataset):
    def __init__(self, sizes, location='./data'):
        self.location = location
        self.transformer = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        self.data = torchvision.datasets.CIFAR10(
            root=location,
            train=True,
            download=True,
            transform=self.transformer
        )
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=location, train=False, download=True, transform=transform_test)
        self.test = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        self.fake = torchvision.datasets.CIFAR10(
            root=location,
            train=True,
            download=True,
            transform=self.transformer
        )
        self.fake.targets[self.fake.targets == 9] = 8

        super().__init__(self.data, sizes)
