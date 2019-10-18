""" live (realtime) latent space interpolations of trained models """

import argparse
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pro_gan_pytorch.PRO_GAN import Generator
from torchvision.utils import make_grid
from math import ceil, sqrt
from scipy.ndimage import gaussian_filter

# create the device for running the demo:
device = th.device("cuda" if th.cuda.is_available() else "cpu")


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None, help="path to the trained generator model")

    parser.add_argument("--depth", action="store", type=int,
                        default=9, help="Depth of the network")

    parser.add_argument("--out_depth", action="store", type=int,
                        default=6,
                        help="output depth of images. **Starts from 0")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=512, help="Latent size for the network")

    parser.add_argument("--num_points", action="store", type=int,
                        default=12, help="Number of samples to be seen")

    parser.add_argument("--transition_points", action="store", type=int,
                        default=30,
                        help="Number of transition samples for interpolation")

    parser.add_argument("--smoothing", action="store", type=float,
                        default=1.0,
                        help="amount of transitional smoothing")

    args = parser.parse_args()

    return args


def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return th.clamp(data, min=0, max=1)


def get_image(gen, point, depth, alpha):
    """
    obtain an All-resolution grid of images from the given point
    :param gen: the generator object
    :param point: random latent point for generation
    :param depth: value of depth for image generation (0 indexed)
    :param alpha: value of alpha for fade-in (between 0 and 1)
    :return: img => generated image
    """
    image = gen(point, depth, alpha).detach()
    image = adjust_dynamic_range(image).squeeze(dim=0)
    return image.cpu().numpy().transpose(1, 2, 0)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    # load the model for the demo
    gen = th.nn.DataParallel(
        Generator(
            depth=args.depth,
            latent_size=args.latent_size))
    gen.load_state_dict(th.load(args.generator_file, map_location=str(device)))

    # generate the set of points:
    total_frames = args.num_points * args.transition_points
    all_latents = th.randn(total_frames, args.latent_size).to(device)
    all_latents = th.from_numpy(
        gaussian_filter(
            all_latents.cpu(),
            [args.smoothing * args.transition_points, 0], mode="wrap"))
    all_latents = (all_latents /
                   all_latents.norm(dim=-1, keepdim=True)) * sqrt(args.latent_size)

    start_point = th.unsqueeze(all_latents[0], dim=0)
    points = all_latents[1:]

    fig, ax = plt.subplots()
    plt.axis("off")
    shower = plt.imshow(get_image(gen, start_point, args.out_depth, 1))

    def init():
        return shower,

    def update(point):
        shower.set_data(get_image(gen, th.unsqueeze(point, dim=0), args.out_depth, 1))
        return shower,

    # define the animation function
    ani = FuncAnimation(fig, update, frames=points, 
                        init_func=init)
    plt.show(ani)


if __name__ == '__main__':
    main(parse_arguments())
