import numpy as np

import torch as th
from custom_layers import EqualizedConv2d
from modules import GenGeneralConvBlock, GenInitialBlock
from torch.nn import Conv2d, ModuleList
from torch.nn.functional import interpolate


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
            int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max,
        ).item()
    )


class Generator(th.nn.Module):
    """
    Generator Module (block) of the GAN network
    Args:
        depth: required depth of the Network
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

    def forward(self, x, depth, alpha):
        """
        forward pass of the Generator
        Args:
            x: input latent noise
            depth: depth from where the network's output is required
            alpha: value of alpha for fade-in effect

        Returns: generated images at the give depth's resolution
        """

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


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, AvgPool2d
        from modules import DisGeneralConvBlock
        from modules import DisFinalBlock

        super(Discriminator, self).__init__()

        assert feature_size != 0 and (
            (feature_size & (feature_size - 1)) == 0
        ), "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(
                2, height - 4
            ), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size

        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            from pro_gan_pytorch.custom_layers import EqualizedConv2d

            self.fromRGB = lambda out_channels: EqualizedConv2d(
                3, out_channels, (1, 1), bias=True
            )
        else:
            from torch.nn import Conv2d

            self.fromRGB = lambda out_channels: Conv2d(
                3, out_channels, (1, 1), bias=True
            )

        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql,
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(
                    self.feature_size, self.feature_size, use_eql=self.use_eql
                )
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """

        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

            straight = self.layers[height - 1](self.rgb_to_features[height](x))

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[: height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y)

        return out


class ConditionalDiscriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, num_classes, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param num_classes: number of classes for conditional discrimination
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, AvgPool2d
        from modules import DisGeneralConvBlock
        from modules import ConDisFinalBlock

        super(ConditionalDiscriminator, self).__init__()

        assert feature_size != 0 and (
            (feature_size & (feature_size - 1)) == 0
        ), "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(
                2, height - 4
            ), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.final_block = ConDisFinalBlock(
            self.feature_size, self.num_classes, use_eql=self.use_eql
        )

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            from pro_gan_pytorch.custom_layers import EqualizedConv2d

            self.fromRGB = lambda out_channels: EqualizedConv2d(
                3, out_channels, (1, 1), bias=True
            )
        else:
            from torch.nn import Conv2d

            self.fromRGB = lambda out_channels: Conv2d(
                3, out_channels, (1, 1), bias=True
            )

        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql,
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(
                    self.feature_size, self.feature_size, use_eql=self.use_eql
                )
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, labels, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param labels: labels required for conditional discrimination
                       note that these are pure integer labels of shape [B x 1]
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values
        """

        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

            straight = self.layers[height - 1](self.rgb_to_features[height](x))

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[: height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y, labels)

        return out
