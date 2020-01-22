import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import os


def train_transform(image_size=512, crop_size=256):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def eval_transform(image_size=512):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class Dataset(data.Dataset):

    def __init__(self, root, transform=None):
        super(Dataset, self).__init__()
        self.data = os.listdir(root)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        full_path = os.path.join(self.root, self.data[index])
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def infinite_sampler(n):
    i = 0
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            order = np.random.permutation(n)
            i = 0


class InfiniteSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(infinite_sampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
