""" script for generating samples from a trained model """

import torch as th
import numpy as np
import argparse
import os
from torch.backends import cudnn
from torchvision.utils import make_grid
from math import sqrt, ceil
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import cv2


# turn fast mode on
cudnn.benchmark = True

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generator_file",
        action="store",
        type=str,
        help="pretrained weights file for generator",
        required=True,
    )

    parser.add_argument(
        "--latent_size",
        action="store",
        type=int,
        default=512,
        help="latent size for the generator",
    )

    parser.add_argument(
        "--depth",
        action="store",
        type=int,
        default=9,
        help="latent size for the generator",
    )

    parser.add_argument(
        "--out_depth",
        action="store",
        type=int,
        default=6,
        help="output depth of images. **Starts from 0",
    )

    parser.add_argument(
        "--time",
        action="store",
        type=float,
        default=300,
        help="Number of seconds for the video to make",
    )

    parser.add_argument(
        "--fps",
        action="store",
        type=int,
        default=60,
        help="Frames per second in the video",
    )

    parser.add_argument(
        "--smoothing",
        action="store",
        type=float,
        default=0.75,
        help="Smoothing amount in transition frames",
    )

    parser.add_argument(
        "--out_dir",
        action="store",
        type=str,
        default="interp_animation_frames/",
        help="path to the output directory for the frames",
    )

    parser.add_argument(
        "--video_only",
        action="store_true",
        help="Pass this to skip saving of individual frames.",
    )

    parser.add_argument(
        "--video_name", action="store", type=str, default="", help="Filename of video"
    )

    args = parser.parse_args()

    return args


def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0])
        )
        bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
        data = data * scale + bias
    return th.clamp(data, min=0, max=1)


def get_image(gen, point, depth, alpha):
    image = gen(point, depth, alpha).detach()
    image = adjust_dynamic_range(image).squeeze(dim=0)
    return image.cpu().numpy().transpose(1, 2, 0)


def main(args):
    """
    Main function of the script
    :param args: parsed commandline arguments
    :return: None
    """
    from pro_gan_pytorch.PRO_GAN import Generator

    # create generator object:
    print("Creating a generator object ...")
    generator = th.nn.DataParallel(
        Generator(depth=args.depth, latent_size=args.latent_size).to(device)
    )

    # load the trained generator weights
    print("loading the trained generator weights ...")
    generator.load_state_dict(th.load(args.generator_file, str(device)))

    # total_frames in the video:
    total_frames = int(args.time * args.fps)

    # Let's create the animation video from the latent space interpolation
    # all latent vectors:
    all_latents = th.randn(total_frames, args.latent_size).to(device)
    all_latents = gaussian_filter(all_latents.cpu(), [args.smoothing * args.fps, 0])
    all_latents = th.from_numpy(all_latents)
    all_latents = (
        all_latents / all_latents.norm(dim=-1, keepdim=True) * (sqrt(args.latent_size))
    )

    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    global_frame_counter = 1

    # If we're saving a video, make the video object
    if args.video_name:
        width = 2 ** (args.depth + 1)
        out_file = os.path.join(args.out_dir, args.video_name)
        video_out = cv2.VideoWriter(
            args.video_name, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, width)
        )

    # Run the main loop for the interpolation:
    print("Generating the video frames ...")
    for latent in tqdm(all_latents):
        latent = th.unsqueeze(latent, dim=0)

        # generate the image for this point:
        img = get_image(generator, latent, args.out_depth, 1) * 255

        if not args.video_only:
            cv2.imwrite(
                os.path.join(args.out_dir, "{:05d}.png".format(global_frame_counter)),
                img,
            )

        # Make an image of unsigned 8-bit integers for OpenCV
        if args.video_name:
            img_int = img.astype(np.uint8)
            video_out.write(img_int)

        # Increment the counter
        global_frame_counter += 1

    # video frames have been generated
    if not args.video_only:
        print("Video frames have been generated at:", args.out_dir)

    if args.video_name:
        print("Video saved to {}".format(out_file))
        video_out.release()


if __name__ == "__main__":
    main(parse_arguments())
