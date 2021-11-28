from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

import torch as th
from .custom_layers import EqualizedConv2d
from .modules import (
    ConDisFinalBlock,
    DisFinalBlock,
    DisGeneralConvBlock,
    GenGeneralConvBlock,
    GenInitialBlock,
)
from torch import Tensor
from torch.nn import Conv2d, LeakyReLU, ModuleList, Sequential
from torch.nn.functional import avg_pool2d, interpolate


def nf(
    stage: int,
    fmap_base: int = 16 << 10,
    fmap_decay: float = 1.0,
    fmap_min: int = 1,
    fmap_max: int = 512,
) -> int:
    """
    computes the number of fmaps present in each stage
    Args:
        stage: stage level
        fmap_base: base number of fmaps
        fmap_decay: decay rate for the fmaps in the network
        fmap_min: minimum number of fmaps
        fmap_max: maximum number of fmaps

    Returns: number of fmaps that should be present there
    """
    return int(
        np.clip(
            int(fmap_base / (2.0 ** (stage * fmap_decay))),
            fmap_min,
            fmap_max,
        ).item()
    )


class Generator(th.nn.Module):
    """
    Generator Module (block) of the GAN network
    Args:
        depth: required depth of the Network (**starts from 2)
        num_channels: number of output channels (default = 3 for RGB)
        latent_size: size of the latent manifold
        use_eql: whether to use equalized learning rate
    """

    def __init__(
        self,
        depth: int = 10,
        num_channels: int = 3,
        latent_size: int = 512,
        use_eql: bool = True,
    ) -> None:
        super().__init__()

        # object state:
        self.depth = depth
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.layers = ModuleList(
            [GenInitialBlock(latent_size, nf(1), use_eql=self.use_eql)]
        )
        for stage in range(1, depth - 1):
            self.layers.append(GenGeneralConvBlock(nf(stage), nf(stage + 1), use_eql))

        self.rgb_converters = ModuleList(
            [
                ConvBlock(nf(stage), num_channels, kernel_size=(1, 1))
                for stage in range(1, depth)
            ]
        )

    def forward(
        self, x: Tensor, depth: Optional[int] = None, alpha: float = 1.0
    ) -> Tensor:
        """
        forward pass of the Generator
        Args:
            x: input latent noise
            depth: depth from where the network's output is required
            alpha: value of alpha for fade-in effect

        Returns: generated images at the give depth's resolution
        """
        depth = self.depth if depth is None else depth
        assert depth <= self.depth, f"Requested output depth {depth} cannot be produced"

        if depth == 2:
            y = self.rgb_converters[0](self.layers[0](x))
        else:
            y = x
            for layer_block in self.layers[: depth - 2]:
                y = layer_block(y)
            residual = interpolate(self.rgb_converters[depth - 3](y), scale_factor=2)
            straight = self.rgb_converters[depth - 2](self.layers[depth - 2](y))
            y = (alpha * straight) + ((1 - alpha) * residual)
        return y

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "depth": self.depth,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size,
                "use_eql": self.use_eql,
            },
            "state_dict": self.state_dict(),
        }


class Discriminator(th.nn.Module):
    """
    Discriminator of the GAN
    Args:
        depth: depth of the discriminator. log_2(resolution)
        num_channels: number of channels of the input images (Default = 3 for RGB)
        latent_size: latent size of the final layer
        use_eql: whether to use the equalized learning rate
        num_classes: number of classes for a conditional discriminator (Default = None)
                     meaning unconditional discriminator
    """

    def __init__(
        self,
        depth: int = 7,
        num_channels: int = 3,
        latent_size: int = 512,
        use_eql: bool = True,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.use_eql = use_eql
        self.num_classes = num_classes
        self.conditional = num_classes is not None

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        if self.conditional:
            self.layers = [ConDisFinalBlock(nf(1), latent_size, num_classes, use_eql)]
        else:
            self.layers = [DisFinalBlock(nf(1), latent_size, use_eql)]

        for stage in range(1, depth - 1):
            self.layers.insert(
                0, DisGeneralConvBlock(nf(stage + 1), nf(stage), use_eql)
            )
        self.layers = ModuleList(self.layers)
        self.from_rgb = ModuleList(
            reversed(
                [
                    Sequential(
                        ConvBlock(num_channels, nf(stage), kernel_size=(1, 1)),
                        LeakyReLU(0.2),
                    )
                    for stage in range(1, depth)
                ]
            )
        )

    def forward(
        self, x: Tensor, depth: int, alpha: float, labels: Optional[Tensor] = None
    ) -> Tensor:
        """
        forward pass of the discriminator
        Args:
            x: input to the network
            depth: current depth of operation (Progressive GAN)
            alpha: current value of alpha for fade-in
            labels: labels for conditional discriminator (Default = None)
                    shape => (Batch_size,) shouldn't be a column vector

        Returns: raw discriminator scores
        """
        assert (
            depth <= self.depth
        ), f"Requested output depth {depth} cannot be evaluated"

        if self.conditional:
            assert labels is not None, "Conditional discriminator required labels"

        if depth > 2:
            residual = self.from_rgb[-(depth - 2)](
                avg_pool2d(x, kernel_size=2, stride=2)
            )
            straight = self.layers[-(depth - 1)](self.from_rgb[-(depth - 1)](x))
            y = (alpha * straight) + ((1 - alpha) * residual)
            for layer_block in self.layers[-(depth - 2) : -1]:
                y = layer_block(y)
        else:
            y = self.from_rgb[-1](x)
        if self.conditional:
            y = self.layers[-1](y, labels)
        else:
            y = self.layers[-1](y)
        return y

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "depth": self.depth,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size,
                "use_eql": self.use_eql,
                "num_classes": self.num_classes,
            },
            "state_dict": self.state_dict(),
        }


def create_generator_from_saved_model(saved_model_path: Path) -> Generator:
    # load the data from the saved_model
    loaded_data = torch.load(saved_model_path)

    # create a generator from the loaded data:
    generator_data = (
        loaded_data["shadow_generator"]
        if "shadow_generator" in loaded_data
        else loaded_data["generator"]
    )
    generator = Generator(**generator_data["conf"])
    generator.load_state_dict(generator_data["state_dict"])

    return generator
