import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models import model_utils
from models.quantize_utils import ActivationQuantizationBlock
from typing import Callable, List, Tuple

class SESR(nn.Module):
    def __init__(self,
                 config: dict,
                 LinearBlock_fn: Callable,
                 mode: str):
        super(SESR, self).__init__()
        """
          Define a residualFlag that is true if using expanded LinearBlock for 
          training. This will be used in the forward function for SESR. If collapsed 
          LinearBlock is used for training, then short residuals are already collapsed 
          within the LinearBlock class.
        """
        self.config = config
        self.export_lite = config['export_lite']
        self.residualFlag = True if LinearBlock_fn == model_utils.LinearBlock_e else False
        self.input_block = LinearBlock_fn(in_filters=1,
                                          num_inner_layers=1, 
                                          kernel_size=5, 
                                          padding='same', 
                                          out_filters=config['int_features'],
                                          feature_size=config['feature_size'],
                                          quant_W=config['quant_W'],
                                          mode=mode)
        self.inputs_A_quant = ActivationQuantizationBlock(enabled=config['quant_A'], mode=mode)
        self.input_block_A_quant = ActivationQuantizationBlock(enabled=config['quant_A'], mode=mode)
        self.linear_blocks = nn.ModuleList([
            LinearBlock_fn(in_filters=config['int_features'],
                           num_inner_layers=1, 
                           kernel_size=3, 
                           padding='same', 
                           out_filters=config['int_features'], 
                           feature_size=config['feature_size'],
                           quant_W=config['quant_W'],
                           mode=mode)
            for i in range(config['m'])])
        if config['quant_W'] and config['quant_A']:
            print('Quantization mode: Using ReLU instead of PReLU activations.')
            self.activations = nn.ModuleList([nn.ReLU() for _ in range(config['m'])])
        else:
            self.activations = nn.ModuleList([nn.PReLU(num_parameters=1) for _ in range(config['m'])])
        self.linear_block_A_quant = nn.ModuleList([ActivationQuantizationBlock(enabled=config['quant_A'], mode=mode) for _ in range(config['m'])])
        self.output_block = LinearBlock_fn(in_filters=config['int_features'],
                                           num_inner_layers=1, 
                                           kernel_size=5, 
                                           padding='same', 
                                           out_filters=config['scale'][0]**2,
                                           feature_size=config['feature_size'],
                                           quant_W=config['quant_W'],
                                           mode=mode)
        self.output_block_A_quant = ActivationQuantizationBlock(enabled=config['quant_A'], mode=mode)
        self.residual_1_A_quant = ActivationQuantizationBlock(enabled=config['quant_A'], mode=mode)
        self.residual_2_A_quant = ActivationQuantizationBlock(enabled=config['quant_A'], mode=mode)

    def forward(self, inputs):
        inputs = self.inputs_A_quant(inputs)
        features_0 = features = self.input_block_A_quant(self.input_block(inputs - 0.5))

        for linear_block, activation, quant in zip(self.linear_blocks, self.activations, self.linear_block_A_quant):
            if self.residualFlag:
                features = activation(linear_block(features) + features)
            else:
                features = activation(linear_block(features))
            features = quant(features)

        residual_1 = self.residual_1_A_quant(features + features_0)
        output_block = self.output_block_A_quant(self.output_block(residual_1))
        features = self.residual_2_A_quant(output_block + inputs)  # residual across whole network

        if not self.export_lite:
            features = F.pixel_shuffle(features, upscale_factor=2)
        else:
            features = features.permute(3, 1, 2, 0)
            features = F.pixel_unshuffle(features, downscale_factor=2)
            features = features.permute(3, 1, 2, 0)
        if self.config['scale'] == 4:  # Another depth_to_space if scale == 4
            if not self.export_lite:
                features = F.pixel_shuffle(features, upscale_factor=2)
            else:
                features = features.permute(3, 1, 2, 0)
                features = F.pixel_unshuffle(features, downscale_factor=2)
                features = features.permute(3, 1, 2, 0)
        return torch.clamp(features, min=0., max=1.)

    def get_config(self):
        config = {
            'feature_size': self.config['feature_size'],
            'm': self.config['m'],
            'LinearBlock_fn': model_utils.LinearBlock_e,
            'quant_W': self.config['quant_W'],
            'quant_A': self.config['quant_A'],
            'export_lite': self.config['export_lite'],
            'mode': 'train'
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(config=config, LinearBlock_fn=model_utils.LinearBlock_e, mode='train')
