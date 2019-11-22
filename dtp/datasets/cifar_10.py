from dtp.datasets.base_dataset import BaseDataset

import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np

class CIFAR10(BaseDataset):
    def __init__(self, location='../../data'):
        self.location = location
        # self.transformer = transforms.Compose([
        #     transforms.ToTensor(),
        #     # First value corresponds to mean and the second value is the std
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.trainset = torchvision.datasets.CIFAR10(
            root=self.location,
            train=True,
            download=True,
        #    transform=self.transformer
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=128,
            shuffle=True,
            num_workers=2
        )

        self.testset = torchvision.datasets.EMNIST(
            split='digits',
            root=self.location,
            train=False,
            download=True,
        #    transform=self.transformer
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=100,
            shuffle=False,
            num_workers=2
        )

    def sample_iid(self, num_users):
        num_items = int(len(self.trainset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(self.trainset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users


