""" Module implementing ProGAN which is trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
import copy
import datetime
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from torch.nn.functional import avg_pool2d, interpolate
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from .custom_layers import update_average
from .data_tools import get_data_loader
from .losses import GANLoss, WganGP
from .networks import Discriminator, Generator
from .utils import adjust_dynamic_range


class ProGAN:
    def __init__(
        self,
        gen: Generator,
        dis: Discriminator,
        device=torch.device("cpu"),
        use_ema: bool = True,
        ema_beta: float = 0.999,
    ):
        assert gen.depth == dis.depth, (
            f"Generator and Discriminator depths are not compatible. "
            f"GEN_Depth: {gen.depth}  DIS_Depth: {dis.depth}"
        )
        self.gen = gen.to(device)
        self.dis = dis.to(device)
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.depth = gen.depth
        self.latent_size = gen.latent_size
        self.device = device

        # if code is to be run on GPU, we can use DataParallel:
        if device == torch.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        print(f"Generator Network: {self.gen}")
        print(f"Discriminator Network: {self.dis}")

        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # initialize the gen_shadow weights equal to the
            # weights of gen
            update_average(self.gen_shadow, self.gen, beta=0)

        # counters to maintain generator and discriminator gradient overflows
        self.gen_overflow_count = 0
        self.dis_overflow_count = 0

    def progressive_downsample_batch(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        Args:
            real_batch: batch of real samples
            depth: depth at which training is going on
            alpha: current value of the fader alpha

        Returns: modified real batch of samples

        """
        # downsample the real_batch for the given depth
        down_sample_factor = int(2 ** (self.depth - depth))
        prior_downsample_factor = int(2 ** (self.depth - depth + 1))

        ds_real_samples = avg_pool2d(
            real_batch, kernel_size=down_sample_factor, stride=down_sample_factor
        )

        if depth > 2:
            prior_ds_real_samples = interpolate(
                avg_pool2d(
                    real_batch,
                    kernel_size=prior_downsample_factor,
                    stride=prior_downsample_factor,
                ),
                scale_factor=2,
            )
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a linear combination of
        # ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        return real_samples

    def optimize_discriminator(
        self,
        loss: GANLoss,
        dis_optim: Optimizer,
        noise: Tensor,
        real_batch: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> float:
        """
        performs a single weight update step on discriminator using the batch of data
        and the noise
        Args:
            loss: the loss function to be used for the optimization
            dis_optim: discriminator optimizer
            noise: input noise for sample generation
            real_batch: real samples batch
            depth: current depth of optimization
            alpha: current alpha for fade-in
            labels: labels for conditional discrimination

        Returns: discriminator loss value
        """
        real_samples = self.progressive_downsample_batch(real_batch, depth, alpha)

        # generate a batch of samples
        fake_samples = self.gen(noise, depth, alpha).detach()
        dis_loss = loss.dis_loss(
            self.dis, real_samples, fake_samples, depth, alpha, labels=labels
        )

        # optimize discriminator
        dis_optim.zero_grad()
        dis_loss.backward()
        if self._check_grad_ok(self.dis):
            dis_optim.step()
        else:
            self.dis_overflow_count += 1

        return dis_loss.item()

    def optimize_generator(
        self,
        loss: GANLoss,
        gen_optim: Optimizer,
        noise: Tensor,
        real_batch: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> float:
        """
        performs a single weight update step on generator using the batch of data
        and the noise
        Args:
            loss: the loss function to be used for the optimization
            gen_optim: generator optimizer
            noise: input noise for sample generation
            real_batch: real samples batch
            depth: current depth of optimization
            alpha: current alpha for fade-in
            labels: labels for conditional discrimination

        Returns: generator loss value
        """
        real_samples = self.progressive_downsample_batch(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        gen_loss = loss.gen_loss(
            self.dis, real_samples, fake_samples, depth, alpha, labels=labels
        )

        # optimize the generator
        gen_optim.zero_grad()
        gen_loss.backward()
        if self._check_grad_ok(self.gen):
            gen_optim.step()
        else:
            self.gen_overflow_count += 1

        return gen_loss.item()

    @staticmethod
    def create_grid(
        samples: Tensor,
        scale_factor: int,
        img_file: Path,
    ) -> None:
        """
        utility function to create a grid of GAN samples
        Args:
            samples: generated samples for feedback
            scale_factor: factor for upscaling the image
            img_file: name of file to write
        Returns: None (saves a file)
        """
        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        samples = adjust_dynamic_range(
            samples, drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
        )

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))), padding=0)

    def _toggle_all_networks(self, mode="train"):
        for network in (self.gen, self.dis):
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")

    @staticmethod
    def _check_grad_ok(network: Module) -> bool:
        grad_ok = True
        for _, param in network.named_parameters():
            if param.grad is not None:
                param_ok = (
                    torch.sum(torch.isnan(param.grad)) == 0
                    and torch.sum(torch.isinf(param.grad)) == 0
                )
                if not param_ok:
                    grad_ok = False
                    break
        return grad_ok

    def get_save_info(
        self, gen_optim: Optimizer, dis_optim: Optimizer
    ) -> Dict[str, Any]:

        if self.device == torch.device("cpu"):
            generator_save_info = self.gen.get_save_info()
            discriminator_save_info = self.dis.get_save_info()
        else:
            generator_save_info = self.gen.module.get_save_info()
            discriminator_save_info = self.dis.module.get_save_info()
        save_info = {
            "generator": generator_save_info,
            "discriminator": discriminator_save_info,
            "gen_optim": gen_optim.state_dict(),
            "dis_optim": dis_optim.state_dict(),
        }
        if self.use_ema:
            save_info["shadow_generator"] = (
                self.gen_shadow.get_save_info()
                if self.device == torch.device("cpu")
                else self.gen_shadow.module.get_save_info()
            )
        return save_info

    def train(
        self,
        dataset: Dataset,
        epochs: List[int],
        batch_sizes: List[int],
        fade_in_percentages: List[int],
        loss_fn: GANLoss = WganGP(),
        batch_repeats: int = 4,
        gen_learning_rate: float = 0.003,
        dis_learning_rate: float = 0.003,
        num_samples: int = 16,
        start_depth: int = 2,
        num_workers: int = 3,
        feedback_factor: int = 100,
        save_dir=Path("./train"),
        checkpoint_factor: int = 10,
    ):
        """
        # TODO implement support for conditional GAN here
        Utility method for training the ProGAN.
        Note that you don't have to necessarily use this method. You can use the
        optimize_generator and optimize_discriminator and define your own
        training routine
        Args:
            dataset: object of the dataset used for training.
                     Note that this is not the dataloader (we create dataloader in this
                     method since the batch_sizes for resolutions can be different)
            epochs: list of number of epochs to train the network for every resolution
            batch_sizes: list of batch_sizes for every resolution
            fade_in_percentages: list of percentages of epochs per resolution
                                used for fading in the new layer not used for
                                first resolution, but dummy value is still needed
            loss_fn: loss function used for training
            batch_repeats: number of iterations to perform on a single batch
            gen_learning_rate: generator learning rate
            dis_learning_rate: discriminator learning rate
            num_samples: number of samples generated in sample_sheet
            start_depth: start training from this depth
            num_workers: number of workers for reading the data
            feedback_factor: number of logs per epoch
            save_dir: directory for saving the models (.bin files)
            checkpoint_factor: save model after these many epochs.
        Returns: None (Writes multiple files to disk)
        """

        print(f"Loaded the dataset with: {len(dataset)} images ...")
        assert (self.depth - 1) == len(
            batch_sizes
        ), "batch_sizes are not compatible with depth"
        assert (self.depth - 1) == len(epochs), "epochs are not compatible with depth"

        self._toggle_all_networks("train")

        # create the generator and discriminator optimizers
        gen_optim = torch.optim.Adam(
            params=self.gen.parameters(),
            lr=gen_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )
        dis_optim = torch.optim.Adam(
            params=self.dis.parameters(),
            lr=dis_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )

        # verbose stuff
        print("setting up the image saving mechanism")
        model_dir, log_dir = save_dir / "models", save_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        feedback_generator = self.gen_shadow if self.use_ema else self.gen

        # image saving mechanism
        with torch.no_grad():
            dummy_data_loader = get_data_loader(dataset, num_samples, num_workers)
            real_images_for_render = next(iter(dummy_data_loader))
            fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)
            self.create_grid(
                real_images_for_render,
                scale_factor=1,
                img_file=log_dir / "real_images.png",
            )
            self.create_grid(
                feedback_generator(fixed_input, self.depth, 1).detach(),
                scale_factor=1,
                img_file=log_dir / "fake_images_0.png",
            )

        # tensorboard summarywriter:
        summary = SummaryWriter(str(log_dir / "tensorboard"))

        # create a global time counter
        global_time = time.time()
        global_step = 0

        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth + 1):
            current_res = int(2 ** current_depth)
            print(f"\n\nCurrently working on Depth: {current_depth}")
            print("Current resolution: %d x %d" % (current_res, current_res))
            depth_list_index = current_depth - 2
            current_batch_size = batch_sizes[depth_list_index]
            data = get_data_loader(dataset, current_batch_size, num_workers)
            ticker = 1
            for epoch in range(1, epochs[depth_list_index] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                print(f"\nEpoch: {epoch}")
                total_batches = len(data)

                # compute the fader point
                fader_point = int(
                    (fade_in_percentages[depth_list_index] / 100)
                    * epochs[depth_list_index]
                    * total_batches
                )

                for (i, batch) in enumerate(data, start=1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    images = batch.to(self.device)

                    gan_input = torch.randn(current_batch_size, self.latent_size).to(
                        self.device
                    )

                    gen_loss, dis_loss = None, None
                    for _ in range(batch_repeats):
                        # optimize the discriminator:
                        dis_loss = self.optimize_discriminator(
                            loss_fn, dis_optim, gan_input, images, current_depth, alpha
                        )

                        # no idea why this needs to be done after discriminator
                        # iteration, but this is where it is done in the original
                        # code
                        if self.use_ema:
                            update_average(
                                self.gen_shadow, self.gen, beta=self.ema_beta
                            )

                        # optimize the generator:
                        gen_loss = self.optimize_generator(
                            loss_fn, gen_optim, gan_input, images, current_depth, alpha
                        )
                    global_step += 1

                    # provide a loss feedback
                    if (
                        i % max(int(total_batches / max(feedback_factor, 1)), 1) == 0
                        or i == 1
                        or i == total_batches
                    ):
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print(
                            "Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                            % (elapsed, i, dis_loss, gen_loss)
                        )
                        summary.add_scalar(
                            "dis_loss", dis_loss, global_step=global_step
                        )
                        summary.add_scalar(
                            "gen_loss", gen_loss, global_step=global_step
                        )
                        # create a grid of samples and save it
                        resolution_dir = log_dir / str(int(2 ** current_depth))
                        resolution_dir.mkdir(exist_ok=True)
                        gen_img_file = resolution_dir / f"{epoch}_{i}.png"

                        # this is done to allow for more GPU space
                        with torch.no_grad():
                            self.create_grid(
                                samples=feedback_generator(
                                    fixed_input, current_depth, alpha
                                ).detach(),
                                scale_factor=int(2 ** (self.depth - current_depth)),
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                if (
                    epoch % checkpoint_factor == 0
                    or epoch == 1
                    or epoch == epochs[depth_list_index]
                ):
                    save_file = model_dir / f"depth_{current_depth}_epoch_{epoch}.bin"
                    torch.save(self.get_save_info(gen_optim, dis_optim), save_file)

        self._toggle_all_networks("eval")
        print("Training completed ...")
