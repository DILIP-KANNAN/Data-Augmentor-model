# utils/train_utils.py
import torch
import random
import numpy as np
from torchvision.utils import make_grid

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def weights_init(m):
    import torch.nn as nn
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def sample_noise(nz, batch_size, device):
    return torch.randn(batch_size, nz, 1, 1, device=device)
