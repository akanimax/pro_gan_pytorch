from typing import Any

import numpy as np

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_almost_equal(x: Any, y: Any, error_margin: float = 3.0) -> None:
    assert np.abs(x - y) <= error_margin
