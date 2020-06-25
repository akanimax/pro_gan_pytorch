""" Module contains custom layers """
from typing import Any, List

import numpy as np

import torch
from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Linear


def update_average(model_tgt, model_src, beta):
    """
    function to calculate the Exponential moving averages for the Generator weights
    This function updates the exponential average weights based on the current training
    Args:
        model_tgt: target model
        model_src: source model
        beta: value of decay beta
    Returns: None (updates the target model)
    """

    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)


class EqualizedConv2d(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch.conv2d(
            input=x,
            weight=self.weight * self.scale,  # scale the weight on runtime
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class EqualizedConvTranspose2d(ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor, output_size: Any = None) -> Tensor:
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size
        )
        return torch.conv_transpose2d(
            input=x,
            weight=self.weight * self.scale,  # scale the weight on runtime
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class EqualizedLinear(Linear):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__(in_features, out_features, bias)

        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_features
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.weight * self.scale, self.bias)


class PixelwiseNorm(torch.nn.Module):
    """
    ------------------------------------------------------------------------------------
    Pixelwise feature vector normalization.
    reference:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    ------------------------------------------------------------------------------------
    """

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    @staticmethod
    def forward(x: Tensor, alpha: float = 1e-8) -> Tensor:
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    Args:
        group_size: Size of each group into which the batch is split
        num_new_features: number of additional feature maps added
    """

    def __init__(self, group_size: int = 4, num_new_features: int = 1) -> None:
        """

        Args:
            group_size: Size of each group into which the batch is split
            num_new_features: number of additional feature maps added
        """
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}, num_new_features={self.num_new_features}"

    def forward(self, x: Tensor, alpha: float = 1e-8) -> Tensor:
        """
        forward pass of the layer
        Args:
            x: input activation volume
            alpha: small number for numerical stability
        Returns: y => x appended with standard deviation constant map
        """
        batch_size, channels, height, width = x.shape

        # reshape x and create the splits of the input accordingly
        y = torch.reshape(
            x,
            [
                batch_size,
                self.num_new_features,
                channels // self.num_new_features,
                height,
                width,
            ],
        )

        y_split = y.split(self.group_size)
        y_list: List[Tensor] = []
        for y in y_split:
            group_size = y.shape[0]

            # [G x M x C' x H x W] Subtract mean over batch.
            y = y - y.mean(dim=0, keepdim=True)

            # [G x M x C' x H x W] Calc standard deviation over batch
            y = torch.sqrt(y.square().mean(dim=0, keepdim=False) + alpha)

            # [M x C' x H x W]  Take average over feature_maps and pixels.
            y = y.mean(dim=[1, 2, 3], keepdim=True)

            # [M x 1 x 1 x 1] Split channels into c channel groups
            y = y.mean(dim=1, keepdim=False)

            # [M x 1 x 1]  Replicate over group and pixels.
            y = y.view((1, *y.shape)).repeat(group_size, 1, height, width)

            # append this to the y_list:
            y_list.append(y)

        y = torch.cat(y_list, dim=0)

        # [B x (N + C) x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
