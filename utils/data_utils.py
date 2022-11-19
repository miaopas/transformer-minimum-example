import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader():

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((128, 128), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    
    trainset = datasets.CIFAR10(root="./data",
                                train=True,
                                download=True,
                                transform=transform_train)
    testset = datasets.CIFAR10(root="./data",
                                train=False,
                                download=True,
                                transform=transform_test)

    
   

    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=256,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=256,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
