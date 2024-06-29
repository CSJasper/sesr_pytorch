import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as nps
import random
from PIL import Image
from datasets.data_utils import GeneratePatches
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Tuple
from datasets.DIV2K import DIV2KDataset
from models import model_utils, sesr
from tqdm import tqdm
#from torchvision import datasets


def psnr(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for ADAM')
    parser.add_argument('--model_name', type=str, default='SESR', help='Name of the model')
    parser.add_argument('--quant_W', action='store_true', help='Quantize weights')
    parser.add_argument('--quant_A', action='store_true', help='Quantize activations')
    parser.add_argument('--gen_lite', action='store_true', help='Generate TorchScript (equivalent to TFLITE)')
    parser.add_argument('--tflite_height', type=int, default=1080, help='Height of LR image in TorchScript')
    parser.add_argument('--tflite_width', type=int, default=1920, help='Width of LR image in TorchScript')
    parser.add_argument('--scale', type=int, default=2, help='Scale of SISR')
    parser.add_argument('--linear_block_type', type=str, default="collapsed")
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--feature_size", type=int, default=256)
    parser.add_argument("--int_features", type=int, default=16)
    args = parser.parse_args()

    DATASET_NAME = 'div2k' if args.scale == 2 else 'div2k/bicubic_x4'
    os.makedirs('logs/', exist_ok=True)
    BASE_SAVE_DIR = 'logs/x2_models/' if args.scale == 2 else 'logs/x4_models/'
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    SUFFIX = 'QAT' if args.quant_W and args.quant_A else "FP32"

    if args.scale == 4:
        PATH_2X = f'logs/x2_models/{args.model_name}_m{args.m}_f{args.int_features}_x2_fs{args.feature_size}_{args.linear_block_type}Training_{SUFFIX}'

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     GeneratePatches(
    #         args.scale,
    #         64,
    #         64,
    #         64
    #     ),
    # ])

    dataset_train = DIV2KDataset(
        config={
            'device': 'cuda',
            'task': 'train',
            'scale': args.scale,
            'patch_size': 64,
        },
        lr_path='/database2/jasper/SR/DIV2K/DIV2K_train_LR_bicubic/X2/',
        hr_path='/database2/jasper/SR/DIV2K/DIV2K_train_HR/',
        training_task='train'
    )

    # dataset_train = DIV2KDataset(
    #     lr_dir="/database2/jasper/SR/DIV2K/DIV2K_train_LR_bicubic/X2/",
    #     hr_dir="/database2/jasper/SR/DIV2K/DIV2K_train_HR/",
    #     scale=args.scale,
    #     patch_size=64,
    #     patches_per_image=64,
    #     transform=transform
    #     )

    val_dataset = DIV2KDataset(
        config={
            'device': 'cuda',
            'task': 'valid',
            'scale': args.scale,
            'patch_size': 64,
        },
        lr_path='/database2/jasper/SR/DIV2K/DIV2K_valid_LR_bicubic/X2/',
        hr_path='/database2/jasper/SR/DIV2K/DIV2K_valid_HR/',
        training_task='valid'
    )

    # val_dataset = DIV2KDataset(
    #     lr_dir="/database2/jasper/SR/DIV2K/DIV2K_valid_LR_bicubic/X2/",
    #     hr_dir="/database2/jasper/SR/DIV2K/DIV2K_valid_HR/",
    #     scale=args.scale,
    #     patch_size=64,
    #     patches_per_image=64,
    #     transform=transform
    # )
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    if args.model_name == 'SESR':
        if args.linear_block_type == 'collapsed':
            LinearBlock_fn = model_utils.LinearBlock_c
        else:
            LinearBlock_fn = model_utils.LinearBlock_e
        model = sesr.SESR(
            config={
                "m": args.m,
                "feature_size": args.feature_size,
                "int_features": args.int_features,
                "quant_W": args.quant_W > 0,
                "quant_A": args.quant_A > 0,
                "export_lite": args.gen_lite,
                "scale": args.scale,
            },
            LinearBlock_fn=LinearBlock_fn,
            mode='train'
        ).to('cuda')
        # model = sesr.SESR(
        #     m=args.m,
        #     feature_size=args.feature_size,
        #     LinearBlock_fn=LinearBlock_fn,
        #     quant_W=args.quant_W,
        #     quant_A=args.quant_A,
        #     export_lite=args.gen_lite,
        #     mode='train'
        # )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    if args.scale == 4:
        base_model = torch.load(PATH_2X)
        model.load_state_dict(base_model.state_dict(), strict=False)

    model.train()

    for epoch in range(args.epochs):
        train_bar = tqdm(
            train_loader,
            desc=f'Train epoch {epoch + 1}/{args.epochs}',
            )
        for lr_imgs, hr_imgs in train_bar:
            lr_imgs = lr_imgs.to('cuda')
            hr_imgs = hr_imgs.to('cuda')
            optimizer.zero_grad()
            preds = model(lr_imgs)
            loss = F.l1_loss(preds, hr_imgs)
            loss.backward()
            optimizer.step()
        
        # validation
        model.eval()
        with torch.no_grad():
            psnr_values = []
            val_bar = tqdm(
                val_loader,
                desc='Validation',
            )
            for lr_imgs, hr_imgs in val_bar:
                lr_imgs = lr_imgs.to('cuda')
                hr_imgs = hr_imgs.to('cuda')
                preds = model(lr_imgs)
                psnr_values.append(psnr(hr_imgs, preds).item())
            avg_psnr = sum(psnr_values) / len(psnr_values)
            print(f'Epoch: {epoch + 1}/{args.epochs}, PSNR: {avg_psnr:.2f}')
        model.train()

    final_savel_path = f'{BASE_SAVE_DIR}{args.model_name}_m{args.m}_f{args.int_features}_x{args.scale}_fs{args.feature_size}_{args.linear_block_type}Training_{SUFFIX}.pth'
    torch.save(model.state_dict(), final_savel_path)

    # if args.gen_lite:
    #     y_lr = torch.randn(1, 1, args.tflite_height, args.tflite_width)
    #     model_quant = sesr.SESR(

    #     )



