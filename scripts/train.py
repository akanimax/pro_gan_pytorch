""" script for training a ProGAN (Progressively grown gan model) """

import argparse
from pathlib import Path

import torch
from torch.backends import cudnn

from pro_gan_pytorch.data_tools import ImageDirectoryDataset, get_transform
from pro_gan_pytorch.gan import ProGAN
from pro_gan_pytorch.networks import Discriminator, Generator
from pro_gan_pytorch.utils import str2bool, str2GANLoss

# turn fast mode on
cudnn.benchmark = True

# define the device for the training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments() -> argparse.Namespace:
    """
    command line arguments parser
    Returns: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "Train Progressively grown GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments (input path to the data and the output directory for saving training assets)
    parser.add_argument("train_path", action="store", type=Path,
                        help="Path to the images folder for training the ProGAN")
    parser.add_argument("output_dir", action="store", type=Path,
                        help="Path to the directory for saving the logs and models")

    # Optional arguments
    # dataset related options:
    parser.add_argument("--rec_dir", action="store", type=str2bool, default=True, required=False,
                        help="whether images stored under one folder or has a recursive dir structure")
    parser.add_argument("--flip_horizontal", action="store", type=str2bool, default=True, required=False,
                        help="whether to apply mirror augmentation")

    # model architecture related options:
    parser.add_argument("--depth", action="store", type=int, default=10, required=False,
                        help="depth of the generator and the discriminator")
    parser.add_argument("--num_channels", action="store", type=int, default=3, required=False,
                        help="number of channels of in the image data")
    parser.add_argument("--latent_size", action="store", type=int, default=512, required=False,
                        help="latent size of the generator and the discriminator")

    # training related options:
    parser.add_argument("--use_eql", action="store", type=str2bool, default=True, required=False,
                        help="whether to use the equalized learning rate")
    parser.add_argument("--use_ema", action="store", type=str2bool, default=True, required=False,
                        help="whether to use the exponential moving averages")
    parser.add_argument("--ema_beta", action="store", type=float, default=0.999, required=False,
                        help="value of the ema beta")
    parser.add_argument("--epochs", action="store", type=int, required=False, nargs="+",
                        default=[42 for _ in range(9)],
                        help="number of epochs over the training dataset per stage")
    parser.add_argument("--batch_sizes", action="store", type=int, required=False, nargs="+",
                        default=[32, 32, 32, 32, 16, 16, 8, 4, 2],
                        help="batch size used for training the model per stage")
    parser.add_argument("--batch_repeats", action="store", type=int, required=False, default=4,
                        help="number of G and D steps executed per training iteration")
    parser.add_argument("--fade_in_percentages", action="store", type=int, required=False, nargs="+",
                        default=[50 for _ in range(9)],
                        help="number of iterations for which fading of new layer happens. Measured in percentage")
    parser.add_argument("--loss_fn", action="store", type=str2GANLoss, required=False, default="wgan_gp",
                        help="loss function used for training the GAN. "
                             "Current options: [wgan_gp, standard_gan]")
    parser.add_argument("--g_lrate", action="store", type=float, required=False, default=0.003,
                        help="learning rate used by the generator")
    parser.add_argument("--d_lrate", action="store", type=float, required=False, default=0.003,
                        help="learning rate used by the discriminator")
    parser.add_argument("--num_feedback_samples", action="store", type=int, required=False, default=4,
                        help="number of samples used for fixed seed gan feedback")
    parser.add_argument("--start_depth", action="store", type=int, required=False, default=2,
                        help="resolution to start the training from. "
                             "Example 2 --> (4x4) | 3 --> (8x8) ... | 10 --> (1024x1024)"
                             "Note that this is not a way to restart a partial training. "
                             "Resuming is not supported currently. But will be soon.")
    parser.add_argument("--num_workers", action="store", type=int, required=False, default=4,
                        help="number of dataloader subprocesses. It's a pytorch thing, you can ignore it ;)."
                             " Leave it to the default value unless things are weirdly slow for you.")
    parser.add_argument("--feedback_factor", action="store", type=int, required=False, default=10,
                        help="number of feedback logs written per epoch")
    parser.add_argument("--checkpoint_factor", action="store", type=int, required=False, default=10,
                        help="number of epochs after which a model snapshot is saved per training stage")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def train_progan(args: argparse.Namespace) -> None:
    """
    method to train the progan (progressively grown gan) given the configuration parameters
    Args:
        args: configuration used for the training
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
        loss_fn=args.loss_fn,
        batch_repeats=args.batch_repeats,
        gen_learning_rate=args.g_lrate,
        dis_learning_rate=args.d_lrate,
        num_samples=args.num_feedback_samples,
        start_depth=args.start_depth,
        num_workers=args.num_workers,
        feedback_factor=args.feedback_factor,
        checkpoint_factor=args.checkpoint_factor,
        save_dir=args.output_dir,
    )


def main() -> None:
    """
    Main function of the script
    Returns: None
    """
    train_progan(parse_arguments())


if __name__ == "__main__":
    main()
