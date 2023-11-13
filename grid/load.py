import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class AddSaltPepperNoise(object):
    def __init__(self, salt=0.02, pepper=0.02):
        self.salt = salt
        self.pepper = pepper

    def __call__(self, tensor):
        data = torch.rand(tensor.size())
        tensor[data < self.salt] = 1
        tensor[data > 1 - self.pepper] = 0
        return tensor


class AddRandomOcclusion(object):
    def __init__(self, occlusion_size=8):
        self.occlusion_size = occlusion_size

    def __call__(self, img):
        x, y = np.random.randint(0, img.size(1) - self.occlusion_size, 2)
        img[:, x : x + self.occlusion_size, y : y + self.occlusion_size] = 0
        return img


transform_noisy = transforms.Compose(
    [
        transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(0.5, 0.3),
        AddSaltPepperNoise(0.2, 0.3),
        AddRandomOcclusion(8),
    ]
)

train_dataset_noisy = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform_noisy
)
test_dataset_noisy = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform_noisy
)

train_loader = DataLoader(train_dataset_noisy, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset_noisy, batch_size=512, shuffle=False)
