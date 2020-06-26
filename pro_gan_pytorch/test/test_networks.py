from test.utils import device

import torch
from networks import Generator


def test_Generator() -> None:
    batch_size, latent_size = 2, 512
    num_channels = 3
    depth = 10  # resolution 1024 x 1024
    mock_generator = Generator(depth=depth, num_channels=num_channels).to(device)
    mock_latent = torch.randn((batch_size, latent_size)).to(device)

    print(f"Generator Network:\n{mock_generator}")

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
