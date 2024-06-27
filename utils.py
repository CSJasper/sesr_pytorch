import torch
import torch.nn as nn
from torchvision import transforms
from typing import Callable, List, Tuple, Optional
import os
import random
from PIL import Image


class Config:
    scale = 2

config = Config()

SCALE = config.scale

if SCALE != 2 and SCALE != 4:
    raise ValueError("Only x2 or x4 SISR is currently supported.")

PATCH_SIZE_HR = 128 if SCALE == 2 else 200
PATCH_SIZE_LR = PATCH_SIZE_HR // SCALE
PATCHES_PER_IMAGE = 64


# Convert RGB image to YCbCr
def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    ycbcr_from_rgb = torch.tensor([[65.481, 128.553, 24.966],
                                   [-37.797, -74.203, 112.0],
                                   [112.0, -93.786, -18.214]])
    rgb = rgb.float() / 255.0
    ycbcr = torch.matmul(rgb, ycbcr_from_rgb.T)
    return ycbcr + torch.tensor([16.0, 128.0, 128.0])

# Get the Y-Channel only
def rgb_to_y(example: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    lr_ycbcr = rgb_to_ycbcr(example['lr'])
    hr_ycbcr = rgb_to_ycbcr(example['hr'])
    return lr_ycbcr[..., 0:1] / 255.0, hr_ycbcr[..., 0:1] / 255.0

# Extract random patches for training
def random_patch(lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def lr_offset(axis: int):
        size = lr.shape[axis]
        return random.randint(0, size - PATCH_SIZE_LR)
    
    lr_offset_x, lr_offset_y = lr_offset(0), lr_offset(1)
    hr_offset_x, hr_offset_y = SCALE * lr_offset_x, SCALE * lr_offset_y
    lr_patch = lr[lr_offset_x:lr_offset_x + PATCH_SIZE_LR, lr_offset_y:lr_offset_y + PATCH_SIZE_LR]
    hr_patch = hr[hr_offset_x:hr_offset_x + PATCH_SIZE_HR, hr_offset_y:hr_offset_y + PATCH_SIZE_HR]
    return lr_patch, hr_patch

# Data augmentations
def augment(lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr = torch.flip(lr, [0])
        hr = torch.flip(hr, [0])
    k = random.randint(0, 3)
    lr = torch.rot90(lr, k, [0, 1])
    hr = torch.rot90(hr, k, [0, 1])
    return lr, hr

# Get many random patches for each image
def patches(lr: torch.Tensor, hr: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    patches_lr, patches_hr = [], []
    for _ in range(PATCHES_PER_IMAGE):
        lr_patch, hr_patch = random_patch(lr, hr)
        lr_patch, hr_patch = augment(lr_patch, hr_patch)
        patches_lr.append(lr_patch)
        patches_hr.append(hr_patch)
    return patches_lr, patches_hr

# Generate INT8 TorchScript (PyTorch equivalent of TFLite)
def generate_int8_torchscript(model: torch.nn.Module,
                              filename: str,
                              path: Optional[str] = '/tmp',
                              fake_quant: bool = False) -> str:
    model.eval()
    scripted_model = torch.jit.script(model)
    
    if not os.path.exists(path):
        os.makedirs(path)
    torchscript_filename = os.path.join(path, filename + '.pt')
    
    scripted_model.save(torchscript_filename)
    
    return torchscript_filename