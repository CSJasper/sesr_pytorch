import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.quantize_utils import fake_quant_with_min_max_vars_per_channel, fake_quant_with_min_max_vars, compute_ranges

class LinearBlock_e(nn.Module):
    def __init__(self,
                 in_filters:int,
                 num_inner_layers: int,
                 kernel_size: int,
                 padding: str,
                 out_filters:int,
                 feature_size: int,
                 quant_W: bool,
                 mode: str
                 ):
        super().__init__()
        """
        Expanded linear block. Input --> 3x3 Conv to expand number of channels 
        to 'feature_size' --> 1x1 Conv to project channels into 'out_filters'. 

        At inference time, this can be analytically collapsed into a single, 
        small 3x3 Conv layer. See also the LinearBlock_c class which is a 
        very efficient method to train linear blocks without any loss in 
        image quality.
        """

        assert not quant_W, 'expanded linear block not compatible with w quant'

        def conv2d(filters: int, kernel_size_: int) -> nn.Module:
            return nn.Conv2d(in_filters, filters, kenrel_size=kernel_size_, padding=padding)
        
        layers = []
        for _ in range(num_inner_layers):
            layers.append([conv2d(filters=feature_size, kernel_size_=kernel_size)])
        layers.append(conv2d(filters=out_filters, kernel_size_=1))
        self.block = nn.Sequential(*layers)
        self.mode = mode
    

    def forward(self, inputs):
        return self.block(inputs)
    

class LinearBlock_c(nn.Module):
    def __init__(
            self,
            in_filters: int,
            num_inner_layers: int,
            kernel_size: int,
            padding: str,
            out_filters: int,
            feature_size: int,
            quant_W: bool,
            mode: str,
    ):
        super().__init__()

        """
        This is a simulated linear block in the train path. The idea is to collapse 
        linear block at each training step to speed up the forward pass. The backward 
        pass still updates all the expanded weights. 

        After training is completed, the weight generation ops are replaced by
        a tf.constant at pb/tflite generation time.

        ----------------------------------------------------------------
        |                            padded_identity                   |
        |                                   |                          |
        |                         conv1x1(inCh, r*inCh)  [optional]    |
        |                                   |                          |
        |                        convkxk(r*inCh, r*inCh)               |
        |                                   |                          |
        |                         conv1x1(r*inCh, outCh)               |
        |                                   |                          |
        |  simulating residual: identity -> +                          |
        |         (or) padded_conv1x1_wt    | (weight_tensor generated)|
        ----------------------------------------------------------------
                                            |
                    input_tensor -> Actual convkxk(inCh, outCh)
                                            |
                                        Final output
        """

        def conv2d(in_filters: int, filters: int, kernel_size_, padding_: str) -> nn.Module:
            return nn.Conv2d(in_filters, filters, kernel_size=kernel_size_, padding=padding_)
        
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.feature_size = feature_size
        self.quant_W = quant_W
        self.mode = mode

        onebyone = True if num_inner_layers > 1 else False

        kernel_size = [kernel_size, kernel_size]
        self.kx, self.ky = kernel_size

        conv1 = conv2d(in_filters, feature_size, [1, 1], 'valid')
        conv2 = conv2d(feature_size,  feature_size, kernel_size, 'valid') if onebyone else conv2d(in_filters, feature_size, kernel_size, 'valid')
        conv3 = conv2d(feature_size, out_filters, [1, 1], 'valid')

        self.collapsed_weights = None

        if onebyone:
            self.collapse = nn.Sequential(conv1, conv2, conv3)
        else:
            self.collapse = nn.Sequential(conv2, conv3)

        
        if self.mode == 'train':
            self.fake_quant_with_min_max_vars_per_channel_fn = fake_quant_with_min_max_vars_per_channel
        elif self.mode == 'infer':
            self.fake_quant_with_min_max_vars_per_channel_fn = fake_quant_with_min_max_vars_per_channel
        
    
        delta = torch.eye(self.in_filters)

        delta = delta.unsqueeze(1).unsqueeze(1)  # [in_filters, 1, 1, in_filters]

        delta = delta.permute(0, 3, 1, 2)

        delta = F.pad(delta, pad=[self.kx - 1, self.kx -1 , self.ky - 1, self.ky - 1])

        # delta = delta.permute(0, 2, 3, 1)

        self.delta = torch.nn.Parameter(delta, requires_grad=False)


        if self.quant_W:
            self.wt_quant_min = torch.nn.Parameter(torch.zeros(self.out_filters), requires_grad=True)
            self.wt_quant_max = torch.nn.Parameter(torch.zeros(self.out_filters), requires_grad=True)
        
            if self.mode == 'train':
                self.wt_quant_initialized = torch.nn.Parameter(torch.tensor(False), requires_grad=False)

        kernel_dim = [self.kx, self.ky, self.in_filters, self.out_filters]
        residual = np.zeros(kernel_dim, dtype=np.float32)

        if self.in_filters == self.out_filters:
            mid_kx = int(self.kx / 2)
            mid_ky = int(self.ky / 2)

            for out_ch in range(self.out_filters):
                residual[mid_kx, mid_ky, out_ch, out_ch] = 1.0

        self.residual = torch.nn.Parameter(torch.Tensor(residual), requires_grad=False)

    def init_wt_quant_ranges(self, kernel: torch.Tensor) -> None:
        quant_max, quant_min = compute_ranges(kernel, per_channel=True, symmetric=True)
        self.wt_quant_max.data.copy_(quant_max)
        self.wt_quant_min.data.copy_(quant_min)
        self.wt_quant_initialized.data.copy_(torch.tensor(True))

    
    def forward(self, inputs):

        if self.mode == 'train' or self.collapsed_weights is None:
            wt_tensor = self.collapse(self.delta)

            wt_tensor = torch.flip(wt_tensor, dims=[1, 2])  # [batch_size, out_filters, kx, ky]

            wt_tensor = wt_tensor.permute(2, 3, 0, 1)  # [kx, ky, batch_size, out_filters]

            wt_tensor += self.residual  # residual shape : [kx, ky, batch_size, out_filters]
            
            if self.mode == 'infer':
                self.collapsed_weights = torch.nn.Parameter(wt_tensor, requires_grad=False)

                self.collapse = None
        else:
            wt_tensor = self.collapsed_weights

        if self.mode == 'train':
            if self.quant_W:
                if not self.wt_quant_initialized:
                    self.init_wt_quant_ranges(wt_tensor)
        elif self.mode == 'infer':
            pass
        else:
            assert False, self.mode

        if self.quant_W:
            wt_tensor = self.fake_quant_with_min_max_vars_per_channel_fn(
                wt_tensor, 
                min=self.wt_quant_min, 
                max=self.wt_quant_max, 
                num_bits=8, 
                narrow_range=True
                )
        wt_tensor = wt_tensor.permute(3, 2, 0, 1)

        out = F.conv2d(inputs, wt_tensor, stride=1, padding='same')

        return out