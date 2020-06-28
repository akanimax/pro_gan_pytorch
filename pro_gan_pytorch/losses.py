""" Module implementing various loss functions """
from typing import Optional

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from .networks import Discriminator


class GANLoss:
    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        calculate the discriminator loss using the following data
        Args:
            discriminator: the Discriminator used by the GAN
            real_samples: real batch of samples
            fake_samples: fake batch of samples
            depth: resolution log 2 of the images under consideration
            alpha: alpha value of the fader
            labels: optional in case of the conditional discriminator

        Returns: computed discriminator loss
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        calculate the generator loss using the following data
        Args:
            discriminator: the Discriminator used by the GAN
            real_samples: real batch of samples
            fake_samples: fake batch of samples
            depth: resolution log 2 of the images under consideration
            alpha: alpha value of the fader
            labels: optional in case of the conditional discriminator

        Returns: computed discriminator loss
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class StandardGAN(GANLoss):
    def __init__(self):
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            real_scores = discriminator(real_samples, depth, alpha, labels)
            fake_scores = discriminator(fake_samples, depth, alpha, labels)
        else:
            real_scores = discriminator(real_samples, depth, alpha)
            fake_scores = discriminator(fake_samples, depth, alpha)

        real_loss = self.criterion(
            real_scores, torch.ones(real_scores.shape).to(real_scores.device)
        )
        fake_loss = self.criterion(
            fake_scores, torch.zeros(fake_scores.shape).to(fake_scores.device)
        )
        return (real_loss + fake_loss) / 2

    def gen_loss(
        self,
        discriminator: Discriminator,
        _: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            fake_scores = discriminator(fake_samples, depth, alpha, labels)
        else:
            fake_scores = discriminator(fake_samples, depth, alpha)
        return self.criterion(
            fake_scores, torch.ones(fake_scores.shape).to(fake_scores.device)
        )


class WganGP(GANLoss):
    """
    Wgan-GP loss function. The discriminator is required for computing the gradient
    penalty.
    Args:
        drift: weight for the drift penalty
    """

    def __init__(self, drift: float = 0.001) -> None:
        self.drift = drift

    @staticmethod
    def _gradient_penalty(
        dis: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        reg_lambda: float = 10,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        private helper for calculating the gradient penalty
        Args:
            dis: the discriminator used for computing the penalty
            real_samples: real samples
            fake_samples: fake samples
            depth: current depth in the optimization
            alpha: current alpha for fade-in
            reg_lambda: regularisation lambda

        Returns: computed gradient penalty
        """
        batch_size = real_samples.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(real_samples.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        merged.requires_grad_(True)

        # forward pass
        if labels is not None:
            assert dis.conditional, "labels passed to an unconditional discriminator"
            op = dis(merged, depth, alpha, labels)
        else:
            op = dis(merged, depth, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(
            outputs=op,
            inputs=merged,
            grad_outputs=torch.ones_like(op),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            real_scores = discriminator(real_samples, depth, alpha, labels)
            fake_scores = discriminator(fake_samples, depth, alpha, labels)
        else:
            real_scores = discriminator(real_samples, depth, alpha)
            fake_scores = discriminator(fake_samples, depth, alpha)
        loss = (
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (self.drift * torch.mean(real_scores ** 2))
        )

        # calculate the WGAN-GP (gradient penalty)
        gp = self._gradient_penalty(
            discriminator, real_samples, fake_samples, depth, alpha, labels=labels
        )
        loss += gp

        return loss

    def gen_loss(
        self,
        discriminator: Discriminator,
        _: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            fake_scores = discriminator(fake_samples, depth, alpha, labels)
        else:
            fake_scores = discriminator(fake_samples, depth, alpha)
        return -torch.mean(fake_scores)
