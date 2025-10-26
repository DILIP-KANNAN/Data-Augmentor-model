# utils/datasets.py
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import torch
from PIL import Image

def get_transforms(img_size=64, nc=3):
    # Note: Tanh -> normalize to [-1, 1] (mean=0.5,std=0.5)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*nc, (0.5,)*nc)
    ])
    return transform

def get_dataloader(dataset_name='MNIST', root='./data', batch_size=128, img_size=64, nc=3, download=True):
    transform = get_transforms(img_size=img_size, nc=nc)

    if dataset_name.lower() == 'mnist':
        ds = MNIST(root=root, train=True, transform=transform, download=download)
        # MNIST is 1-channel; convert to 3-channel by repeating if nc==3
        if nc == 3:
            ds.data = ds.data  # We will handle channel conversion in a wrapper
            # Use a small wrapper dataset
            class _Wrapper(torch.utils.data.Dataset):
                def __init__(self, ds, transform):
                    self.ds = ds
                    self.transform = transform
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, idx):
                    img, label = self.ds[idx]
                    # img currently is Tensor 1xH x W normalized by transform -> but transform converts to 1 channel
                    if img.shape[0] == 1 and nc == 3:
                        img = img.repeat(3,1,1)
                    return img, label
            ds = _Wrapper(ds, transform)
    elif dataset_name.lower() == 'cifar10':
        ds = CIFAR10(root=root, train=True, transform=transform, download=download)
    else:
        raise ValueError('dataset_name must be MNIST or CIFAR10 for now.')

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader
