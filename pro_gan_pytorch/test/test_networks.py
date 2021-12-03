import numpy as np

import torch

from ..networks import Discriminator, Generator
from .utils import device


# noinspection PyPep8Naming
def test_Generator() -> None:
    batch_size, latent_size = 2, 512
    num_channels = 3
    depth = 10  # resolution 1024 x 1024
    mock_generator = Generator(depth=depth, num_channels=num_channels).to(device)
    mock_latent = torch.randn((batch_size, latent_size)).to(device)

    print(f"Generator Network:\n{mock_generator}")

    with torch.no_grad():
        for res_log2 in range(2, depth + 1):
            rgb_images = mock_generator(mock_latent, depth=res_log2, alpha=1)
            print(f"RGB output shape at depth {res_log2}: {rgb_images.shape}")
            assert rgb_images.shape == (
                batch_size,
                num_channels,
                2 ** res_log2,
                2 ** res_log2,
            )
            assert torch.isnan(rgb_images).sum().item() == 0
            assert torch.isinf(rgb_images).sum().item() == 0


# noinspection PyPep8Naming
def test_DiscriminatorUnconditional() -> None:
    batch_size, latent_size = 2, 512
    num_channels = 3
    depth = 10  # resolution 1024 x 1024
    mock_discriminator = Discriminator(depth=depth, num_channels=num_channels).to(
        device
    )
    mock_inputs = [
        torch.randn((batch_size, num_channels, 2 ** stage, 2 ** stage)).to(device)
        for stage in range(2, depth + 1)
    ]

    print(f"Discriminator Network:\n{mock_discriminator}")

    with torch.no_grad():
        for res_log2 in range(2, depth + 1):
            mock_input = mock_inputs[res_log2 - 2]
            print(f"RGB input image shape at depth {res_log2}: {mock_input.shape}")
            score = mock_discriminator(mock_input, depth=res_log2, alpha=1)
            assert score.shape == (batch_size,)
            assert torch.isnan(score).sum().item() == 0
            assert torch.isinf(score).sum().item() == 0


# noinspection PyPep8Naming
def test_DiscriminatorConditional() -> None:
    batch_size, latent_size = 2, 512
    num_channels = 3
    depth = 10  # resolution 1024 x 1024
    mock_discriminator = Discriminator(
        depth=depth, num_channels=num_channels, num_classes=10
    ).to(device)
    mock_inputs = [
        torch.randn((batch_size, num_channels, 2 ** stage, 2 ** stage)).to(device)
        for stage in range(2, depth + 1)
    ]
    mock_labels = torch.from_numpy(np.array([3, 7])).to(device)

    print(f"Discriminator Network:\n{mock_discriminator}")
    with torch.no_grad():
        for res_log2 in range(2, depth + 1):
            mock_input = mock_inputs[res_log2 - 2]
            print(f"RGB input image shape at depth {res_log2}: {mock_input.shape}")
            score = mock_discriminator(
                mock_input, depth=res_log2, alpha=1, labels=mock_labels
            )
            assert score.shape == (batch_size,)
            assert torch.isnan(score).sum().item() == 0
            assert torch.isinf(score).sum().item() == 0
