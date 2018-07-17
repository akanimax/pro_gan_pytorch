""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import numpy as np
import torch as th


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Upsample
        from pro_gan_pytorch.CustomLayers import GenGeneralConvBlock, GenInitialBlock

        super(Generator, self).__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = GenInitialBlock(self.latent_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        if self.use_eql:
            from pro_gan_pytorch.CustomLayers import _equalized_conv2d
            self.toRGB = lambda in_channels: \
                _equalized_conv2d(in_channels, 3, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.toRGB = lambda in_channels: Conv2d(in_channels, 3, (1, 1), bias=True)

        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size,
                                            self.latent_size, use_eql=self.use_eql)
                rgb = self.toRGB(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = self.toRGB(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = Upsample(scale_factor=2)

    def forward(self, x, depth, alpha):
        """
        forward pass of the Generator
        :param x: input noise
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """
        from torch.nn.functional import tanh

        assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = tanh(self.rgb_converters[depth - 1](self.temporaryUpsampler(y)))
            straight = tanh(self.rgb_converters[depth](self.layers[depth - 1](y)))

            out = (alpha * straight) + ((1 - alpha) * residual)

        else:
            out = tanh(self.rgb_converters[0](y))

        return out


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, AvgPool2d
        from pro_gan_pytorch.CustomLayers import DisGeneralConvBlock, DisFinalBlock

        super(Discriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size

        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            from pro_gan_pytorch.CustomLayers import _equalized_conv2d
            self.fromRGB = lambda out_channels: \
                _equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.fromRGB = lambda out_channels: Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(self.feature_size,
                                            self.feature_size, use_eql=self.use_eql)
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """

        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))

            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y)

        return out


class ProGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, depth=7, latent_size=64, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", device=th.device("cpu")):
        """
        constructor for the class
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator
                         (Used only if loss is wgan or wgan-gp)
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid"]
                     Or an instance of GANLoss
        :param device: device to run the GAN on (GPU / CPU)
        """

        from torch.optim import Adam

        # Create the Generator and the Discriminator
        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)
        self.dis = Discriminator(depth, latent_size, use_eql=use_eql).to(device)

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift

        # define the optimizers for the discriminator and generator
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        # define the loss function used for training the GAN
        self.loss = self.__setup_loss(loss)

    def __setup_loss(self, loss):
        import pro_gan_pytorch.Losses as losses

        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = losses.WGAN_GP(self.device, self.dis, self.drift, use_gp=False)
                # note if you use just wgan, you will have to use weight clipping
                # in order to prevent gradient exploding

            elif loss == "wgan-gp":
                loss = losses.WGAN_GP(self.device, self.dis, self.drift, use_gp=True)

            elif loss == "lsgan":
                loss = losses.LSGAN(self.device, self.dis)

            elif loss == "lsgan-with-sigmoid":
                loss = losses.LSGAN_SIGMOID(self.device, self.dis)

            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, losses.GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """
        from torch.nn import AvgPool2d

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        real_samples = AvgPool2d(down_sample_factor)(real_batch)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        loss = self.loss.gen_loss(fake_samples, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        # return the loss value
        return loss.item()
