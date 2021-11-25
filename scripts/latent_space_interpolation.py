""" script for writing a video of the latent space interpolation from a trained model """
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.backends import cudnn
from tqdm import tqdm

from pro_gan_pytorch.networks import Generator
from pro_gan_pytorch.utils import adjust_dynamic_range

# turn fast mode on
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    # fmt: off
    # required arguments
    parser.add_argument("model_path", action="store", type=Path,
                        help="path to the trained_model.bin file")

    # options related to the video
    parser.add_argument("--output_path", action="store", type=Path, required=False,
                        default="./latent_space_walk.mp4",
                        help="path to the output video file location. "
                             "Please only use mp4 format with this tool (.mp4 extension). "
                             "I have banged my head too much to get anything else to work :(.")
    parser.add_argument("--generation_depth", action="store", type=int, default=None, required=False,
                        help="depth at which the images should be generated. "
                             "Starts from 2 --> (4x4) | 3 --> (8x8) etc.")
    parser.add_argument("--time", action="store", type=float, default=30, required=False,
                        help="number of seconds in the video")
    parser.add_argument("--fps", action="store", type=int, default=60, required=False,
                        help="fps of the generated video")
    parser.add_argument("--smoothing", action="store", type=float, default=0.75, required=False,
                        help="smoothness of walking in the latent-space."
                             " High values corresponds to more smoothing.")
    # fmt: on

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function of the script
    :param args: parsed commandline arguments
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data from the trained-model
    print(f"loading data from the trained model at: {args.model_path}")
    loaded_data = torch.load(args.model_path)

    # create a generator from the loaded data:
    print("creating the Generator object ...")
    generator_data = (
        loaded_data["shadow_generator"]
        if "shadow_generator" in loaded_data
        else loaded_data["generator"]
    )
    generator = Generator(**generator_data["conf"]).to(device)
    generator.load_state_dict(generator_data["state_dict"])

    # total_frames in the video:
    total_frames = int(args.time * args.fps)

    # create the video from the latent space interpolation (walk)
    # all latent vectors for each and every frame:
    all_latents = torch.randn(total_frames, generator.latent_size).to(device)
    all_latents = gaussian_filter(all_latents.cpu(), [args.smoothing * args.fps, 0])
    all_latents = torch.from_numpy(all_latents).to(device)

    # create output directory
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # make the cv2 video object
    print("Generating the video frames ...")
    generation_depth = (
        generator.depth if args.generation_depth is None else args.generation_depth
    )
    img_dim = 2 ** generation_depth
    video_out = cv2.VideoWriter(
        str(args.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (img_dim, img_dim),
    )

    # Run the main loop for the interpolation:
    for latent in tqdm(all_latents):
        latent = torch.unsqueeze(latent, dim=0)

        # generate the image for this latent vector:
        frame = generator.forward(latent, depth=generation_depth)
        frame = frame[0].permute(1, 2, 0)
        frame = adjust_dynamic_range(frame, drange_in=(-1, 1), drange_out=(0, 1))
        frame = (frame * 255.0).detach().cpu().numpy().astype(np.uint8)
        frame = frame[..., ::-1]  # need to reverse the channel order for cv2 :D

        # write the generated frame to the video
        video_out.write(frame)

    print(f"video has been generated and saved to {args.output_path}")

    # don't forget to close the video stream :)
    video_out.release()


if __name__ == "__main__":
    main(parse_arguments())
