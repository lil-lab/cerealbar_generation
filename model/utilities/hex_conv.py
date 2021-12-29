"""Utilities for convolutions."""
import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional

from agent import util


def get_parity_matrix(size: int, center_even=False):
    # Note: assumed to be square.
    parities = torch.tensor(np.mod(np.tile(np.linspace(0, size - 1, size), (size, 1)), 2))
    print(center_even)

    if center_even:
        parities = 1 - parities

    return parities


def offset_hex_convolve(input_tensor: torch.Tensor,
                        conv_filter: torch.Tensor,
                        bias: torch.Tensor,
                        parity_matrix: torch.Tensor,
                        padding: int = 1,
                        stride: int = 1) -> torch.Tensor:
    # Even computation
    even_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 1, 0]]).to(util.DEVICE)
    # Filter can stay as-is.
    even_output = functional.conv2d(input_tensor, conv_filter * even_mask, bias=bias, padding=padding, stride=stride)

    # Odd computation
    odd_mask = torch.tensor([[0, 1, 0], [1, 1, 1], [1, 1, 1]]).to(util.DEVICE).float()
    # Need to adjust the conv_filter
    conv_filter_shift_parity_matrix = get_parity_matrix(conv_filter.size(-1)).to(util.DEVICE)
    shifted_conv_filter = torch.zeros(conv_filter.size()).float().to(util.DEVICE)
    shifted_conv_filter[:, :, 1:] = conv_filter[:, :, :-1]
    odd_conv_filter = conv_filter * conv_filter_shift_parity_matrix + shifted_conv_filter * (1 - conv_filter_shift_parity_matrix)

    odd_output = functional.conv2d(input_tensor, odd_conv_filter.float() * odd_mask, bias=bias, padding=padding, stride=stride)

    # Combine the two.
    # Note: the parity matrix MUST have values of 0 for even rows, and 1 for odd rows.
    # TODO: Not sure how this works for stride > 1, because then the parity matrix will not be the same size.
    parity_matrix = parity_matrix.unsqueeze(1)
    output = even_output * (1 - parity_matrix) + odd_output * parity_matrix
    return output


def _get_hex_conv_mask(kernel_size: int) -> torch.Tensor:
    # This is a mask on the filter which zeros out the corners of the convolution.
    # See https://arxiv.org/pdf/1803.02108.pdf, Figure 4a.
    mask = torch.ones((kernel_size, kernel_size))
    cutoff_amount = (kernel_size - 1) // 2
    for i in range(cutoff_amount):
        for j in range(cutoff_amount - i):
            mask[i][j] = 0.
            mask[kernel_size - 1 - i][kernel_size - 1 - j] = 0.
    return mask


class HexConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = False):
        super(HexConv, self).__init__()

        if kernel_size % 2 != 1:
            raise ValueError('Kernel size must be odd for Hex Conv: %s' % kernel_size)

        self._filter = nn.Parameter(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)),
                                    requires_grad=True)
        nn.init.kaiming_uniform_(self._filter, a=math.sqrt(5))
        self._bias = None
        if bias:
            self._bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._filter)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._bias, -bound, bound)
        self._stride = stride
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._padding = padding
        self._mask = nn.Parameter(_get_hex_conv_mask(kernel_size), requires_grad=False)

    def forward(self, input_tensor: torch.Tensor):
        """Input must be in axial coordinates. """
        masked_filter = self._filter * self._mask.detach()
        print(masked_filter)
        return functional.conv2d(input_tensor, masked_filter, bias=self._bias, stride=self._stride,
                                 padding=self._padding)


class HexCrop(nn.Module):
    def __init__(self, crop_size: int):
        """Crops an N x N region around the center of a tensor, where N = crop size."""
        super(HexCrop, self).__init__()
        if crop_size % 2 != 1:
            raise ValueError('Crop size must be odd for Hex Crop: %s' % crop_size)
        self._crop_mask = nn.Parameter(_get_hex_conv_mask(crop_size), requires_grad=False)
        self._crop_size = crop_size

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Crops a square portion around the center of the input tensor, masking out values not in the neighborhood
        of the crop value. Input must be in axial coordinates."""
        batch_size, num_channels, height, width = input_tensor.size()
        if height != width:
            raise ValueError('Input tensor must be square. Got input dimensions of %s x %s' % (height, width))

        # Crop around the center.
        center_px: int = height // 2
        min_px: int = center_px - int(self._crop_size / 2)
        max_px: int = center_px + int(self._crop_size / 2) + 1

        cropped_square = input_tensor[:, :, min_px:max_px, min_px:max_px].contiguous()

        # Mask
        return cropped_square * self._crop_mask.detach()
