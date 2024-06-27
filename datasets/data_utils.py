import torch
import numpy as np
import random
from typing import Callable, List, Tuple


class RandomPatch:
    def __init__(
            self,
            scale: int,
            patch_size_lr: int,
            patch_size_hr:int,

    ):
        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = patch_size_hr
    
    def __call__(self, lr: torch.Tensor, hr:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        def lr_offset(axis: int):
            size = lr.shape[axis]
            return random.randint(0, size - self.patch_size_lr)
        
        lr_offset_x, lr_offset_y = lr_offset(0), lr_offset(1)
        hr_offset_x, hr_offset_y = self.scale * lr_offset_x, self.scale * lr_offset_y
        lr_patch = lr[lr_offset_x:lr_offset_x + self.patch_size_lr, lr_offset_y:lr_offset_y + self.patch_size_lr]
        hr_patch = hr[hr_offset_x:hr_offset_x + self.patch_size_hr, hr_offset_y:hr_offset_y + self.patch_size_hr]
        return lr_patch, hr_patch
    
# def random_patch(
#         lr:np.ndarray,
#         hr:np.ndarray,
#         scale:int,
#         patch_size_lr:int,
#         patch_size_hr:int
# ) -> Tuple[np.ndarray, np.ndarray]:
#     def lr_offset(axis: int):
#         size = lr.shape[axis]
#         return random.randint(0, size - patch_size_lr)
    
#     lr_offset_x, lr_offset_y = lr_offset(0), lr_offset(1)
#     hr_offset_x, hr_offset_y = scale * lr_offset_x, scale * lr_offset_y
#     lr_patch = lr[lr_offset_x:lr_offset_x + patch_size_lr, lr_offset_y:lr_offset_y + patch_size_lr]
#     hr_patch = hr[hr_offset_x:hr_offset_x + patch_size_hr, hr_offset_y:hr_offset_y + patch_size_hr]
#     return lr_patch, hr_patch


class Augment:
    def __call__(
            self,
            lr: torch.Tensor,
            hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            lr = torch.flip(lr, [0])
            hr = torch.flip(hr, [0])
        k = random.randint(0, 3)
        lr = torch.rot90(lr, k, [0, 1])
        hr = torch.rot90(hr, k, [0, 1])
        return lr, hr

# def augment(lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     if random.random() < 0.5:
#         lr = np.flip(lr)
#         hr = np.flip(hr)
#     k = random.randint(0, 3)
#     lr = np.rot90(lr, k).copy()
#     hr = np.rot90(hr, k).copy()
#     return lr, hr

class GeneratePatches:
    def __init__(
            self,
            scale:int,
            patch_size_lr:int,
            patch_size_hr:int,
            patches_per_images:int

    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = patch_size_hr
        self.patches_per_images = patches_per_images
        self.random_patch = RandomPatch(scale, patch_size_lr, patch_size_hr)
        self.augment = Augment()
    

    def __call__(self, lr: torch.Tensor, hr: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        patches_lr, patches_hr = [], []

        for _ in range(self.patches_per_images):
            lr_patch, hr_patch = self.random_patch(lr, hr)
            lr_patch, hr_patch = self.augment(lr_patch, hr_patch)
            patches_lr.append(lr_patch)
            patches_hr.append(hr_patch)

        return patches_lr, patches_hr


# def patches(lr: np.ndarray,
#             hr: np.ndarray, 
#             scale: int, 
#             patch_size_lr: int,
#             patch_size_hr: int,
#             patches_per_images:int
#             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     patches_lr, patches_hr = [], []

#     for _ in range(patches_per_images):
#         lr_patch, hr_patch = random_patch(lr, hr, scale, patch_size_lr, patch_size_hr)
#         lr_patch, hr_patch = augment(lr_patch, hr_patch)
#         patches_lr.append(lr_patch)
#         patches_hr.append(hr_patch)

#     return patches_lr, patches_hr
    
def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    ycbcr_from_rgb = np.array([[65.481, 128.553, 24.966],
                               [-37.797, -74.203, 112.0],
                               [112.0, -93.786, -18.214]])
    rgb = img.astype(np.float32) / 255.0
    ycbcr = np.dot(rgb, ycbcr_from_rgb.T)
    return ycbcr + np.array([16.0, 128.0, 128.0])

def rgb_to_y(img: np.ndarray) -> np.ndarray:
    ycbcr = rgb_to_ycbcr(img)
    return ycbcr[..., 0:1] / 255.0
