# pro_gan_pytorch
Package contains implementation of ProGAN. 
Paper titled "Progressive growing of GANs for improved 
Quality, Stability, and Variation". <br>
link -> https://arxiv.org/abs/1710.10196

# Steps to use:
1.) Install your appropriate version of PyTorch. 
The torch dependency in this package uses the most basic
"cpu" version. follow instructions on 
<a href="http://pytorch.org/"> http://pytorch.org </a> to 
install the "gpu" version of PyTorch.<br>

2.)  Install this package using pip:
    
    $ workon [your virtual environment]
    $ pip install pro-gan-pth
    
3.) In your code:
    
    import pytorch_pro_gan.PRO_GAN as pg
 
 Use the modules `pg.Generator`, `pg.Discriminator` and
 `pg.ProGAN`.
 
    Help on class ProGAN in module pro_gan_pytorch.PRO_GAN:
    
    class ProGAN(builtins.object)
     |  Wrapper around the Generator and the Discriminator
     |  
     |  Methods defined here:
     |  
     |  __init__(self, depth=7, latent_size=64, learning_rate=0.001, beta_1=0, beta_2=0.99, eps=1e-08, drift=0.001, 
                 n_critic=1, device=device(type='cpu'))
     |      constructor for the class
     |      :param depth: depth of the GAN (will be used for each generator and discriminator)
     |      :param latent_size: latent size of the manifold used by the GAN
     |      :param learning_rate: learning rate for Adam
     |      :param beta_1: beta_1 for Adam
     |      :param beta_2: beta_2 for Adam
     |      :param eps: epsilon for Adam
     |      :param n_critic: number of times to update discriminator
     |      :param device: device to run the GAN on (GPU / CPU)
     |  
     |  optimize_discriminator(self, noise, real_batch, depth, alpha)
     |      performs one step of weight update on discriminator using the batch of data
     |      :param noise: input noise of sample generation
     |      :param real_batch: real samples batch
     |      :param depth: current depth of optimization
     |      :param alpha: current alpha for fade-in
     |      :return: current loss (Wasserstein loss)
     |  
     |  optimize_generator(self, noise, depth, alpha)
     |      performs one step of weight update on generator for the given batch_size
     |      :param noise: input random noise required for generating samples
     |      :param depth: depth of the network at which optimization is done
     |      :param alpha: value of alpha for fade-in effect
     |      :return: current loss (Wasserstein estimate)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
