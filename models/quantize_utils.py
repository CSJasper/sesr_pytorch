from typing import Callable, List, Tuple, Union
import torch
import torch.nn as nn


def compute_ranges(kernel: torch.Tensor, per_channel: bool, symmetric: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    axes = torch.arange(kernel.ndim - 1) if per_channel else None
    with torch.no_grad():
        if symmetric:
            quant_max = torch.max(torch.abs(kernel), dim=axes, keepdim=True).values if per_channel else torch.max(torch.abs(kernel))
            quant_min = -quant_max
        else:
            quant_max = torch.max(kernel, dim=axes, keepdim=True).values if per_channel else torch.max(kernel)
            quant_min = torch.min(kernel, dim=axes, keepdim=True).values if per_channel else torch.min(kernel)
        
        if per_channel:
            quant_max = quant_max.unsqueeze(-1)
            quant_min = quant_min.unsqueeze(-1)

    return quant_min, quant_max


def floor_ste(x:torch.Tensor) -> torch.Tensor:

    class FloorSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return torch.floor(input)
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None
        
    
    return FloorSTE.apply(x)
        

def get_nudged_ranges_scale(
        min: torch.Tensor,
        max: torch.Tensor,
        num_bits: int,
        narrow_range: bool=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quant_max = torch.pow(torch.tensor(2.), torch.tensor(num_bits, dtype=torch.float32)) - 1.
    quant_min = torch.tensor(1.) if narrow_range else torch.tensor(0.)

    scale = (max - min) / (quant_max - quant_min)

    zero_point_from_min = quant_min - min / scale
    nudged_zero_point = torch.round(zero_point_from_min)
    nudged_zero_point = torch.where(zero_point_from_min < quant_min,
                                    quant_min * torch.ones_like(nudged_zero_point),
                                    nudged_zero_point)
    nudged_zero_point = torch.where(zero_point_from_min > quant_max,
                                    quant_max * torch.ones_like(nudged_zero_point),
                                    nudged_zero_point)
    
    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return nudged_min, nudged_max, scale


def fake_quant_with_min_max_vars(
        inputs: torch.Tensor,
        min: torch.Tensor,
        max: torch.Tensor,
        num_bits: int,
        narrow_range: bool=False
) -> torch.Tensor:
    nudged_min, nudged_max, scale = get_nudged_ranges_scale(min, max, num_bits, narrow_range)
    clipped_data = torch.clamp(inputs, nudged_min, nudged_max)
    shifted_data = clipped_data - nudged_min
    quant_data = floor_ste(shifted_data / scale + 0.5)
    quant_data = quant_data * scale + nudged_min

    return quant_data

fake_quant_with_min_max_vars_per_channel = fake_quant_with_min_max_vars

class ActivationQuantizationBlock(nn.Module):
    def __init__(self,
                 enabled: bool,
                 mode: str):
        super().__init__()
        self.enabled = enabled
        self.mode = mode

        if self.enabled:
            self.quant_min = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
            self.quant_max = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
            if self.mode == 'train':
                self.quant_initialized = False

        if self.mode == 'train':
            self.fake_quant_with_min_max_vars_fn = fake_quant_with_min_max_vars
        elif self.mode == 'infer':
            self.fake_quant_with_min_max_vars_fn = lambda inputs, min, max: torch.fake_quantize_per_channel_affine(inputs, min, max, 256, 0)

    
    def init_quant_ranges(self, inputs: torch.Tensor) -> None:
        quant_max = inputs.max()
        quant_min = inputs.min()

        self.quant_max.data = quant_max
        self.quant_min.data = quant_min
        self.quant_initialized = True
    

    def forward(self, inputs):
        if self.enabled:
            if self.mode == 'train':
                if not self.quant_initialized:
                    self.init_quant_ranges(inputs)
            
            return self.fake_quant_with_min_max_vars_fn(
                inputs,
                min=self.quant_min,
                max=self.quant_max,
                num_bits=8,
                narrow_range=False
            )
        else:
            return inputs
        
    