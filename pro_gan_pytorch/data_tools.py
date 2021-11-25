""" Module for the data loading pipeline for the model to train """
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor

from .utils import adjust_dynamic_range


class NoOp(object):
    """A NoOp image transform utility. Does nothing, but makes the code cleaner"""

    def __call__(self, whatever: Any) -> Any:
        return whatever

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


def get_transform(
    new_size: Optional[Tuple[int, int]] = None, flip_horizontal: bool = False
) -> Callable[[Image.Image], Tensor]:
    """
    obtain the image transforms required for the input data
    Args:
        new_size: size of the resized images (if needed, could be None)
        flip_horizontal: whether to randomly mirror input images during training
    Returns: requested transform object from TorchVision
    """
    return Compose(
        [
            RandomHorizontalFlip(p=0.5) if flip_horizontal else NoOp(),
            Resize(new_size) if new_size is not None else NoOp(),
            ToTensor(),
        ]
    )


class ImageDirectoryDataset(Dataset):
    """pyTorch Dataset wrapper for the simple case of flat directory images dataset
    Args:
        data_dir: directory containing all the images
        transform: whether to apply a certain transformation to the images
        rec_dir: whether to search all the sub-level directories for files
                 recursively
    """

    def __init__(
        self,
        data_dir: Path,
        transform: Callable[[Image.Image], Tensor] = get_transform(),
        input_data_range: Tuple[float, float] = (0.0, 1.0),
        output_data_range: Tuple[float, float] = (-1.0, 1.0),
        rec_dir: bool = False,
    ) -> None:
        # define the state of the object
        self.rec_dir = rec_dir
        self.data_dir = data_dir
        self.transform = transform
        self.output_data_range = output_data_range
        self.input_data_range = input_data_range

        # setup the files for reading
        self.files = self._get_files(data_dir, rec_dir)

    def _get_files(self, path: Path, rec: bool = False) -> List[Path]:
        """
        helper function to search the given directory and obtain all the files in it's
        structure
        Args:
            path: path to the root directory
            rec: whether to search all the sub-level directories for files recursively
        Returns: list of all found paths
        """
        files = []
        for possible_file in path.iterdir():
            if possible_file.is_file():
                files.append(possible_file)
            elif rec and possible_file.is_dir():
                files.extend(self._get_files(possible_file))
        return files

    def __len__(self) -> int:
        """
        compute the length of the dataset
        Returns: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, item: int) -> Tensor:
        """
        obtain the image (read and transform)
        Args:
            item: index for the required image
        Returns: img => image array
        """
        # read the image:
        image = self.files[item]
        if image.name.endswith(".npy"):
            img = np.load(str(image))
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(image)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # bring the image in the required range
        img = adjust_dynamic_range(
            img, drange_in=self.input_data_range, drange_out=self.output_data_range
        )

        return img


def get_data_loader(
    dataset: Dataset, batch_size: int, num_workers: int = 3
) -> DataLoader:
    """
    generate the data_loader from the given dataset
    Args:
        dataset: Torch dataset object
        batch_size: batch size for training
        num_workers: num of parallel readers for reading the data
    Returns: dataloader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
