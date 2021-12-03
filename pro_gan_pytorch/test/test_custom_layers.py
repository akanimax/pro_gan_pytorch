import torch

from ..custom_layers import (
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    EqualizedLinear,
    MinibatchStdDev,
    PixelwiseNorm,
)
from .utils import assert_almost_equal, device, assert_tensor_validity


# noinspection PyPep8Naming
def test_EqualizedConv2d() -> None:
    mock_in = torch.randn(32, 21, 16, 16).to(device)
    conv_block = EqualizedConv2d(21, 3, kernel_size=(3, 3), padding=1).to(device)
    print(f"Equalized conv block: {conv_block}")

    mock_out = conv_block(mock_in)

    # check output
    assert_tensor_validity(mock_out, (32, 3, 16, 16))

    # check the weight's scale
    assert_almost_equal(conv_block.weight.data.std().cpu(), 1, error_margin=1e-1)


# noinspection PyPep8Naming
def test_EqualizedConvTranspose2d() -> None:
    mock_in = torch.randn(32, 21, 16, 16).to(device)

    conv_transpose_block = EqualizedConvTranspose2d(
        21, 3, kernel_size=(3, 3), padding=1
    ).to(device)
    print(f"Equalized conv transpose block: {conv_transpose_block}")

    mock_out = conv_transpose_block(mock_in)

    # check output
    assert_tensor_validity(mock_out, (32, 3, 16, 16))

    # check the weight's scale
    assert_almost_equal(
        conv_transpose_block.weight.data.std().cpu(), 1, error_margin=1e-1
    )


# noinspection PyPep8Naming
def test_EqualizedLinear() -> None:
    # test the forward for the first res block
    mock_in = torch.randn(32, 13).to(device)

    lin_block = EqualizedLinear(13, 52).to(device)
    print(f"Equalized linear block: {lin_block}")

    mock_out = lin_block(mock_in)

    # check output
    assert_tensor_validity(mock_out, (32, 52))

    # check the weight's scale
    assert_almost_equal(lin_block.weight.data.std().cpu(), 1, error_margin=1e-1)


# noinspection PyPep8Naming
def test_PixelwiseNorm() -> None:
    mock_in = torch.randn(1, 13, 1, 1).to(device)
    normalizer = PixelwiseNorm()
    print(f"\nNormalizerBlock: {normalizer}")
    mock_out = normalizer(mock_in)

    # check output
    assert_tensor_validity(mock_out, mock_in.shape)

    # we cannot comment that the norm of the output tensor
    # will always be less than the norm of the input tensor
    # so no more checking can be done


# noinspection PyPep8Naming
def test_MinibatchStdDev() -> None:
    mock_in = torch.randn(16, 13, 16, 16).to(device)
    minStdD = MinibatchStdDev()
    print(f"\nMiniBatchStdDevBlock: {minStdD}")
    mock_out = minStdD(mock_in)

    # check output
    assert mock_out.shape[1] == mock_in.shape[1] + 1
    assert_tensor_validity(mock_out, (16, 14, 16, 16))
