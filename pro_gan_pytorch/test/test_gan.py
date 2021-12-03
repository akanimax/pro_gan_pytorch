from pathlib import Path

# noinspection PyPackageRequirements
import matplotlib.pyplot as plt

import torch

from ..data_tools import ImageDirectoryDataset, get_transform
from ..gan import ProGAN
from ..networks import Discriminator, Generator
from .utils import device


def test_pro_gan_progressive_downsample_batch() -> None:
    batch = torch.randn((4, 3, 1024, 1024)).to(device)
    batch = torch.clamp(batch, min=0, max=1)
    progan = ProGAN(Generator(10), Discriminator(10), device=device)

    for res_log2 in range(2, 10):
        modified_batch = progan.progressive_downsample_batch(
            batch, depth=res_log2, alpha=0.001
        )
        print(f"Downsampled batch at res_log2 {res_log2}: {modified_batch.shape}")
        plt.figure()
        plt.title(f"Image at resolution: {int(2 ** res_log2)}x{int(2 ** res_log2)}")
        plt.imshow(modified_batch.permute((0, 2, 3, 1))[0].cpu().numpy())
        assert modified_batch.shape == (
            batch.shape[0],
            batch.shape[1],
            int(2 ** res_log2),
            int(2 ** res_log2),
        )

    plt.figure()
    plt.title(f"Image at resolution: {1024}x{1024}")
    plt.imshow(batch.permute((0, 2, 3, 1))[0].cpu().numpy())
    plt.show()


def test_pro_gan_train(test_data_path: Path) -> None:
    depth = 4
    progan = ProGAN(Generator(depth), Discriminator(depth), device=device)
    progan.train(
        dataset=ImageDirectoryDataset(
            test_data_path,
            transform=get_transform(
                new_size=(int(2 ** depth), int(2 ** depth)), flip_horizontal=False
            ),
            rec_dir=False,
        ),
        epochs=[10 for _ in range(3)],
        batch_sizes=[256, 256, 256],
        fade_in_percentages=[50 for _ in range(3)],
        save_dir=Path("./test_train"),
        num_samples=64,
        feedback_factor=10,
    )
    print("test_finished")
