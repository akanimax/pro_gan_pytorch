""" Generate single image samples from a particular depth of a model """

import argparse
import torch as th
import numpy as np
import os
from torch.backends import cudnn
from pro_gan_pytorch.PRO_GAN import Generator
from torch.nn.functional import interpolate
from scipy.misc import imsave
from tqdm import tqdm

# turn on the fast GPU processing mode on
cudnn.benchmark = True


# set the manual seed
# th.manual_seed(3)

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    parser.add_argument("--latent_size", action="store", type=int,
                        default=256,
                        help="latent size for the generator")

    parser.add_argument("--depth", action="store", type=int,
                        default=9,
                        help="depth of the network. **Starts from 1")

    parser.add_argument("--out_depth", action="store", type=int,
                        default=6,
                        help="output depth of images. **Starts from 0")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=300,
                        help="number of synchronized grids to be generated")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="interp_animation_frames/",
                        help="path to the output directory for the frames")

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


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    print("Creating generator object ...")
    # create the generator object
    gen = th.nn.DataParallel(Generator(
        depth=args.depth,
        latent_size=args.latent_size
    ))

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(
        th.load(args.generator_file, map_location=str(device))
    )

    # path for saving the files:
    save_path = args.out_dir

    print("Generating scale synchronized images ...")
    for img_num in tqdm(range(1, args.num_samples + 1)):
        # generate the images:
        with th.no_grad():
            point = th.randn(1, args.latent_size)
            point = (point / point.norm()) * (args.latent_size ** 0.5)
            ss_image = gen(point, depth=args.out_depth, alpha=1)
            # color adjust the generated image:
            ss_image = adjust_dynamic_range(ss_image)

        # save the ss_image in the directory
        imsave(os.path.join(save_path, str(img_num) + ".png"),
               ss_image.squeeze(0).permute(1, 2, 0).cpu())

    print("Generated %d images at %s" % (args.num_samples, save_path))


if __name__ == '__main__':
    main(parse_arguments())
