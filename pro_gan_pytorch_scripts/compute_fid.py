""" script for computing the fid of a trained model when compared with the dataset images """
import argparse
import tempfile
from pathlib import Path

import imageio as imageio
import torch
from cleanfid import fid
from torch.backends import cudnn
from tqdm import tqdm

from pro_gan_pytorch.networks import create_generator_from_saved_model
from pro_gan_pytorch.utils import post_process_generated_images

# turn fast mode on
cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    """
    Returns: parsed arguments object
    """
    parser = argparse.ArgumentParser("ProGAN fid_score computation tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    # fmt: off
    # required arguments
    parser.add_argument("model_path", action="store", type=Path,
                        help="path to the trained_model.bin file")
    parser.add_argument("dataset_path", action="store", type=Path,
                        help="path to the directory containing the images from the dataset. "
                             "Note that this needs to be a flat directory")

    # optional arguments
    parser.add_argument("--generated_images_path", action="store", type=Path, default=None, required=False,
                        help="path to the directory where the generated images are to be written. "
                             "Uses a temporary directory by default. Provide this path if you'd like "
                             "to see the generated images yourself :).")
    parser.add_argument("--batch_size", action="store", type=int, default=4, required=False,
                        help="batch size used for generating random images")
    parser.add_argument("--num_generated_images", action="store", type=int, default=50_000, required=False,
                        help="number of generated images used for computing the FID")
    # fmt: on

    args = parser.parse_args()

    return args


def compute_fid(args: argparse.Namespace) -> None:
    """
    compute the fid for a given trained pro-gan model
    Args:
        args: configuration used for the fid computation
    Returns: None

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data from the trained-model
    print(f"loading data from the trained model at: {args.model_path}")
    generator = create_generator_from_saved_model(args.model_path).to(device)

    # create the generated images directory:
    if args.generated_images_path is not None:
        args.generated_images_path.mkdir(parents=True, exist_ok=True)
    generated_images_path = (
        args.generated_images_path
        if args.generated_images_path is not None
        else tempfile.TemporaryDirectory()
    )
    if args.generated_images_path is None:
        image_writing_path = Path(generated_images_path.name)
    else:
        image_writing_path = generated_images_path

    print("generating random images from the trained generator ...")
    with torch.no_grad():
        for img_num in tqdm(range(0, args.num_generated_images, args.batch_size)):
            num_imgs = min(args.batch_size, args.num_generated_images - img_num)
            random_latents = torch.randn(num_imgs, generator.latent_size, device=device)
            gen_imgs = post_process_generated_images(generator(random_latents))

            # write the batch of generated images:
            for batch_num, gen_img in enumerate(gen_imgs, start=1):
                imageio.imwrite(
                    image_writing_path / f"{img_num + batch_num}.png",
                    gen_img,
                )

    # compute the fid once all images are generated
    print("computing fid ...")
    score = fid.compute_fid(
        fdir1=args.dataset_path,
        fdir2=image_writing_path,
        mode="clean",
        num_workers=4,
    )
    print(f"fid score: {score: .3f}")

    # most importantly, don't forget to do the cleanup on the temporary directory:
    if hasattr(generated_images_path, "cleanup"):
        generated_images_path.cleanup()


def main() -> None:
    """
    Main function of the script
    Returns: None
    """
    compute_fid(parse_arguments())


if __name__ == "__main__":
    main()
