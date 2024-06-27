import os
import numpy as np
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets.data_utils import rgb_to_y


# Do not use this dataset class for demo
class DIV2KDataset(Dataset):
    def __init__(
            self,
            config:dict,
            lr_path:str,
            hr_path:str,
            training_task:str='train'
    ):
        super().__init__()
        self.config = config
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.training_task = training_task

        self.lr_images = sorted(os.listdir(lr_path))
        self.hr_images = sorted(os.listdir(hr_path))

        assert len(self.lr_images) == len(self.hr_images), "Number of LR images and HR images must be the same"
    

    def __getitem__(
            self,
            index
    ):
        lr_img = Image.open(os.path.join(self.lr_path, self.lr_images[index]))
        hr_img = Image.open(os.path.join(self.hr_path, self.hr_images[index]))

        lr_img = np.array(lr_img)
        lr_img = rgb_to_y(lr_img)
        lr_img = lr_img.astype(np.float32).transpose([2, 0, 1]) / 255.0

        hr_img = np.array(hr_img)
        hr_img = rgb_to_y(hr_img)
        hr_img = hr_img.astype(np.float32).transpose([2, 0, 1]) / 255.0

        lr_img = torch.Tensor(np.array(lr_img))
        #lr_img = lr_img.to(self.config['device'])

        hr_img = torch.Tensor(np.array(hr_img))
        #hr_img = hr_img.to(self.config['device'])

        if self.config['task'] == 'train' and self.training_task != 'valid':
            lr_img, hr_img = self.get_patch_pair(lr_img, hr_img)

        return lr_img, hr_img
        

    
    def __len__(self) -> int:
        return len(self.lr_images)
    

    def get_patch_pair(self, lr_img, hr_img):
        scale = self.config['scale']
        lr_h, lr_w = lr_img.shape[1:] # c h w

        lr_patch_size = self.config['patch_size']
        lr_h_start = random.randint(0, lr_h - lr_patch_size)
        lr_w_start = random.randint(0, lr_w - lr_patch_size)
        lr_patch = lr_img[:, lr_h_start: lr_h_start + lr_patch_size, lr_w_start: lr_w_start + lr_patch_size]

        hr_patch_size = lr_patch_size * scale
        hr_h_start = lr_h_start * scale
        hr_w_start = lr_w_start * scale
        hr_patch = hr_img[:, hr_h_start: hr_h_start + hr_patch_size, hr_w_start: hr_w_start + hr_patch_size]

        return lr_patch, hr_patch


# class DIV2KDataset(data.Dataset):
#     def __init__(self, args, train=True, benchmark=False):
#         self.args = args
#         self.train = train
#         self.split = 'train' if train else 'test'
#         self.benchmark = benchmark
#         self.scale = args.scale
#         self.idx_scale = 0

#         self._set_filesystem(args.dir_data)

#         def _load_bin():
#             self.images_hr = np.load(self._name_hrbin())
#             self.images_lr = [
#                 np.load(self._name_lrbin(s)) for s in self.scale
#             ]

#         if args.ext == 'img' or benchmark:
#             self.images_hr, self.images_lr = self._scan()
#         elif args.ext.find('sep') >= 0:
#             self.images_hr, self.images_lr = self._scan()
#             if args.ext.find('reset') >= 0:
#                 print('Preparing seperated binary files')
#                 for v in self.images_hr:
#                     hr = misc.imread(v)
#                     name_sep = v.replace(self.ext, '.npy')
#                     np.save(name_sep, hr)
#                 for si, s in enumerate(self.scale):
#                     for v in self.images_lr[si]:
#                         lr = misc.imread(v)
#                         name_sep = v.replace(self.ext, '.npy')
#                         np.save(name_sep, lr)

#             self.images_hr = [
#                 v.replace(self.ext, '.npy') for v in self.images_hr
#             ]
#             self.images_lr = [
#                 [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
#                 for i in range(len(self.scale))
#             ]

#         elif args.ext.find('bin') >= 0:
#             try:
#                 if args.ext.find('reset') >= 0:
#                     raise IOError
#                 print('Loading a binary file')
#                 _load_bin()
#             except:
#                 print('Preparing a binary file')
#                 bin_path = os.path.join(self.apath, 'bin')
#                 if not os.path.isdir(bin_path):
#                     os.mkdir(bin_path)

#                 list_hr, list_lr = self._scan()
#                 hr = [misc.imread(f) for f in list_hr]
#                 np.save(self._name_hrbin(), hr)
#                 del hr
#                 for si, s in enumerate(self.scale):
#                     lr_scale = [misc.imread(f) for f in list_lr[si]]
#                     np.save(self._name_lrbin(s), lr_scale)
#                     del lr_scale
#                 _load_bin()
#         else:
#             print('Please define data type')

#     def _scan(self):
#         raise NotImplementedError

#     def _set_filesystem(self, dir_data):
#         raise NotImplementedError

#     def _name_hrbin(self):
#         raise NotImplementedError

#     def _name_lrbin(self, scale):
#         raise NotImplementedError

#     def __getitem__(self, idx):
#         lr, hr, filename = self._load_file(idx)
#         lr, hr = self._get_patch(lr, hr)
#         lr, hr = common.set_channel([lr, hr], self.args.n_colors)
#         lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
#         return lr_tensor, hr_tensor, filename

#     def __len__(self):
#         return len(self.images_hr)

#     def _get_index(self, idx):
#         return idx

#     def _load_file(self, idx):
#         idx = self._get_index(idx)
#         lr = self.images_lr[self.idx_scale][idx]
#         hr = self.images_hr[idx]
#         if self.args.ext == 'img' or self.benchmark:
#             filename = hr
#             lr = misc.imread(lr)
#             hr = misc.imread(hr)
#         elif self.args.ext.find('sep') >= 0:
#             filename = hr
#             lr = np.load(lr)
#             hr = np.load(hr)
#         else:
#             filename = str(idx + 1)

#         filename = os.path.splitext(os.path.split(filename)[-1])[0]

#         return lr, hr, filename

#     def _get_patch(self, lr, hr):
#         patch_size = self.args.patch_size
#         scale = self.scale[self.idx_scale]
#         multi_scale = len(self.scale) > 1
#         if self.train:
#             lr, hr = common.get_patch(
#                 lr, hr, patch_size, scale, multi_scale=multi_scale
#             )
#             lr, hr = common.augment([lr, hr])
#             lr = common.add_noise(lr, self.args.noise)
#         else:
#             ih, iw = lr.shape[0:2]
#             hr = hr[0:ih * scale, 0:iw * scale]

#         return lr, hr

#     def set_scale(self, idx_scale):
#         self.idx_scale = idx_scale

