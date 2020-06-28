import torch
from .custom_layers import (
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    MinibatchStdDev,
    PixelwiseNorm,
)
from torch import Tensor
from torch.nn import AvgPool2d, Conv2d, ConvTranspose2d, Embedding, LeakyReLU, Module
from torch.nn.functional import interpolate


class GenInitialBlock(Module):
    """
    Module implementing the initial block of the input
    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels of the block
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(GenInitialBlock, self).__init__()
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d
        ConvTransposeBlock = EqualizedConvTranspose2d if use_eql else ConvTranspose2d

        self.conv_1 = ConvTransposeBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv_2 = ConvBlock(
            out_channels, out_channels, (3, 3), padding=1, bias=True
        )
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        y = self.pixNorm(y)  # normalize the latents to hypersphere
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y = self.pixNorm(y)
        return y


class GenGeneralConvBlock(torch.nn.Module):
    """
    Module implementing a general convolutional block
    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels required
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(GenGeneralConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = ConvBlock(
            out_channels, out_channels, (3, 3), padding=1, bias=True
        )
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = interpolate(x, scale_factor=2)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y


class DisFinalBlock(torch.nn.Module):
    """
    Final block for the Discriminator
    Args:
        in_channels: number of input channels
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(DisFinalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(
            in_channels + 1, in_channels, (3, 3), padding=1, bias=True
        )
        self.conv_2 = ConvBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv_3 = ConvBlock(out_channels, 1, (1, 1), bias=True)
        self.batch_discriminator = MinibatchStdDev()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.batch_discriminator(x)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y = self.conv_3(y)
        return y.view(-1)


class ConDisFinalBlock(torch.nn.Module):
    """ Final block for the Conditional Discriminator
        Uses the Projection mechanism
        from the paper -> https://arxiv.org/pdf/1802.05637.pdf
        Args:
            in_channels: number of input channels
            num_classes: number of classes for conditional discrimination
            use_eql: whether to use equalized learning rate
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_classes: int, use_eql: bool
    ) -> None:
        super(ConDisFinalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(
            in_channels + 1, in_channels, (3, 3), padding=1, bias=True
        )
        self.conv_2 = ConvBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv_3 = ConvBlock(out_channels, 1, (1, 1), bias=True)

        # we also need an embedding matrix for the label vectors
        self.label_embedder = Embedding(num_classes, out_channels, max_norm=1)
        self.batch_discriminator = MinibatchStdDev()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        y = self.batch_discriminator(x)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # embed the labels
        labels = self.label_embedder(labels)  # [B x C]

        # compute the inner product with the label embeddings
        y_ = torch.squeeze(torch.squeeze(y, dim=-1), dim=-1)  # [B x C]
        projection_scores = (y_ * labels).sum(dim=-1)  # [B]

        # normal discrimination score
        y = self.lrelu(self.conv_3(y))  # This layer has linear activation

        # calculate the total score
        final_score = y.view(-1) + projection_scores

        # return the output raw discriminator scores
        return final_score


class DisGeneralConvBlock(torch.nn.Module):
    """
    General block in the discriminator
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(DisGeneralConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(in_channels, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.downSampler = AvgPool2d(2)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)
        return y
