import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import torch
from torch import Tensor

from pro_gan_pytorch import losses
from pro_gan_pytorch.losses import WganGP, StandardGAN


def adjust_dynamic_range(
    data: Tensor,
    drange_in: Optional[Tuple[float, float]] = (-1.0, 1.0),
    drange_out: Optional[Tuple[float, float]] = (0.0, 1.0),
):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0])
        )
        bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
        data = data * scale + bias

    return torch.clamp(data, min=drange_out[0], max=drange_out[1])


def post_process_generated_images(imgs: Tensor) -> np.array:
    imgs = adjust_dynamic_range(
        imgs.permute(0, 2, 3, 1), drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
    )
    return (imgs * 255.0).detach().cpu().numpy().astype(np.uint8)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# noinspection PyPep8Naming
def str2GANLoss(v):
    if v.lower() == "wgan_gp":
        return WganGP()
    elif v.lower() == "standard_gan":
        return StandardGAN()
    else:
        raise argparse.ArgumentTypeError(
            "Unknown gan-loss function requested."
            f"Please consider contributing a your GANLoss to: "
            f"{str(Path(losses.__file__).absolute())}"
        )
