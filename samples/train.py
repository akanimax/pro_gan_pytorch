""" script for generating samples from a trained model """

import argparse
from pathlib import Path

import torch
from pro_gan_pytorch.data_tools import ImageDirectoryDataset, get_transform
from pro_gan_pytorch.gan import ProGAN
from pro_gan_pytorch.networks import Discriminator, Generator
from pro_gan_pytorch.utils import str2bool
from torch.backends import cudnn

# turn fast mode on
cudnn.benchmark = True

# define the device for the training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "Train Progressively grown GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "train_path",
        action="store",
        type=lambda x: Path(x),
        help="Path to the images folder for training the ProGAN",
    )

    parser.add_argument(
        "output_dir",
        action="store",
        type=lambda x: Path(x),
        help="Path to the directory for saving the logs and models",
    )

    parser.add_argument(
        "--rec_dir",
        action="store",
        type=str2bool,
        default=True,
        help="whether images stored under one folder or has a recursive dir structure",
        required=False,
    )

    parser.add_argument(
        "--flip_horizontal",
        action="store",
        type=str2bool,
        default=True,
        help="whether to apply mirror augmentation",
        required=False,
    )

    parser.add_argument(
        "--depth",
        action="store",
        type=int,
        default=10,
        help="depth of the generator and the discriminator",
        required=False,
    )

    parser.add_argument(
        "--num_channels",
        action="store",
        type=int,
        default=3,
        help="number of channels of in the image data",
        required=False,
    )

    parser.add_argument(
        "--latent_size",
        action="store",
        type=int,
        default=512,
        help="latent size of the generator and the discriminator",
        required=False,
    )

    parser.add_argument(
        "--use_eql",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the equalized learning rate",
        required=False,
    )

    parser.add_argument(
        "--use_ema",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the exponential moving averages",
        required=False,
    )

    parser.add_argument(
        "--ema_beta",
        action="store",
        type=float,
        default=0.999,
        help="value of the ema beta",
        required=False,
    )

    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=[172 for _ in range(9)],
        help="Mapper network configuration",
    )

    parser.add_argument(
        "--batch_sizes",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=[512, 256, 128, 64, 32, 16, 16, 16, 16],
        help="Mapper network configuration",
    )

    parser.add_argument(
        "--fade_in_percentages",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=[50 for _ in range(9)],
        help="Mapper network configuration",
    )

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function of the script
    Args:
        args: parsed commandline arguments

    Returns: None
    """
    print(f"Selected arguments: {args}")

    generator = Generator(
        depth=args.depth,
        num_channels=args.num_channels,
        latent_size=args.latent_size,
        use_eql=args.use_eql,
    )
    discriminator = Discriminator(
        depth=args.depth,
        num_channels=args.num_channels,
        latent_size=args.latent_size,
        use_eql=args.use_eql,
    )

    progan = ProGAN(
        generator,
        discriminator,
        device=device,
        use_ema=args.use_ema,
        ema_beta=args.ema_beta,
    )

    progan.train(
        dataset=ImageDirectoryDataset(
            args.train_path,
            transform=get_transform(
                new_size=(int(2 ** args.depth), int(2 ** args.depth)),
                flip_horizontal=args.flip_horizontal,
            ),
            rec_dir=args.rec_dir,
        ),
        epochs=args.epochs,
        batch_sizes=args.batch_sizes,
        fade_in_percentages=args.fade_in_percentages,
        save_dir=args.output_dir,
        feedback_factor=10,
        checkpoint_factor=20,
    )


if __name__ == "__main__":
    main(parse_arguments())
