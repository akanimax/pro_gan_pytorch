from typing import Any

import numpy as np

import torch as th
from custom_layers import (EqualizedConv2d, EqualizedConvTranspose2d,
                           EqualizedLinear)

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def assert_almost_equal(x: Any, y: Any, error_margin: float = 3.0) -> None:
    assert np.abs(x - y) <= error_margin


def test_EqualizedConv2d():
    mock_in = th.randn(32, 21, 16, 16).to(device)
    conv_block = EqualizedConv2d(21, 3, kernel_size=(3, 3), padding=1).to(device)
    print("\nEqualized conv block:\n%s" % str(conv_block))

    mock_out = conv_block(mock_in)

    # check output
    assert mock_out.shape == (32, 3, 16, 16)
    assert th.isnan(mock_out).sum().item() == 0
    assert th.isinf(mock_out).sum().item() == 0

    # check the weight's scale
    assert_almost_equal(conv_block.weight.data.std().cpu(), 1, error_margin=1e-1)


def test_EqualizedConvTranspose2d():
    mock_in = th.randn(32, 21, 16, 16).to(device)

    conv_transpose_block = EqualizedConvTranspose2d(
        21, 3, kernel_size=(3, 3), padding=1
    ).to(device)
    print("\nEqualized conv block:\n%s" % str(conv_transpose_block))

    mock_out = conv_transpose_block(mock_in)

    # check output
    assert mock_out.shape == (32, 3, 16, 16)
    assert th.isnan(mock_out).sum().item() == 0
    assert th.isinf(mock_out).sum().item() == 0

    # check the weight's scale
    assert_almost_equal(
        conv_transpose_block.weight.data.std().cpu(), 1, error_margin=1e-1
    )


def test_EqualizedLinear():
    # test the forward for the first res block
    mock_in = th.randn(32, 13).to(device)

    lin_block = EqualizedLinear(13, 52).to(device)
    print("\nEqualized linear block:\n%s" % str(lin_block))

    mock_out = lin_block(mock_in)

    # check output
    assert mock_out.shape == (32, 52)
    assert th.isnan(mock_out).sum().item() == 0
    assert th.isinf(mock_out).sum().item() == 0

    # check the weight's scale
    assert_almost_equal(lin_block.weight.data.std().cpu(), 1, error_margin=1e-1)


#
#
# class Test_PixelwiseNorm():
#
#     def setUp(self):
#         self.normalizer = cL.PixelwiseNorm()
#
#     def test_forward(self):
#         mock_in = th.randn(1, 13, 1, 1).to(device)
#         mock_out = self.normalizer(mock_in)
#
#         # check output
#         self.assertEqual(mock_out.shape, mock_in.shape)
#         self.assertEqual(th.isnan(mock_out).sum().item(), 0)
#         self.assertEqual(th.isinf(mock_out).sum().item(), 0)
#
#         # we cannot comment that the norm of the output tensor
#         # will always be less than the norm of the input tensor
#         # so no more checking can be done
#
#     def tearDown(self):
#         # delete the computational resources
#         del self.normalizer
#
#
# class Test_MinibatchStdDev():
#
#     def setUp(self):
#         self.minStdD = cL.MinibatchStdDev()
#
#     def test_forward(self):
#         mock_in = th.randn(1, 13, 16, 16).to(device)
#         mock_out = self.minStdD(mock_in)
#
#         # check output
#         self.assertEqual(mock_out.shape[1], mock_in.shape[1] + 1)
#         self.assertEqual(th.isnan(mock_out).sum().item(), 0)
#         self.assertEqual(th.isinf(mock_out).sum().item(), 0)
#
#     def tearDown(self):
#         # delete the computational resources
#         del self.minStdD
