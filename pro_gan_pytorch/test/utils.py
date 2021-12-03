from typing import Any, Tuple

import numpy as np

import torch
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_almost_equal(x: Any, y: Any, error_margin: float = 3.0) -> None:
    assert np.abs(x - y) <= error_margin


def assert_tensor_validity(
    test_tensor: Tensor, expected_shape: Tuple[int, ...]
) -> None:
    assert test_tensor.shape == expected_shape
    assert torch.isnan(test_tensor).sum().item() == 0
    assert torch.isinf(test_tensor).sum().item() == 0
