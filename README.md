# pro_gan_pytorch
Package contains implementation of ProGAN.<br> 
Paper titled "Progressive growing of GANs for improved 
Quality, Stability, and Variation". <br>
link -> https://arxiv.org/abs/1710.10196 <br>
Trained Examples at -> https://github.com/akanimax/pro_gan_pytorch-examples

# Exemplar Samples :)
### Training gif (fixed latent points):
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/celebA-HQ.gif"
     height=80% width=80%/>
</p>
<br>

### Trained Image sheet:
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/celebA-HQ.png"
 height=80% width=80%/>
</p>
<br>

## Other links
medium blog -> https://medium.com/@animeshsk3/the-unprecedented-effectiveness-of-progressive-growing-of-gans-37475c88afa3
<br>
Full training video -> https://www.youtube.com/watch?v=lzTm6Lq76Mo

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
    
    import pro_gan_pytorch.PRO_GAN as pg
 
 Use the modules `pg.Generator`, `pg.Discriminator` and
 `pg.ProGAN`. Mostly, you'll only need the ProGAN 
 module for training. For inference, you will probably 
 need the `pg.Generator`.

4.) Example Code for CIFAR-10 dataset:

    import torch as th
    import torchvision as tv
    import pro_gan_pytorch.PRO_GAN as pg

    # select the device to be used for training
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    data_path = "cifar-10/"

    def setup_data(download=False):
        """
        setup the CIFAR-10 dataset for training the CNN
        :param batch_size: batch_size for sgd
        :param num_workers: num_readers for data reading
        :param download: Boolean for whether to download the data
        :return: classes, trainloader, testloader => training and testing data loaders
        """
        # data setup:
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        transforms = tv.transforms.ToTensor()

        trainset = tv.datasets.CIFAR10(root=data_path,
                                       transform=transforms,
                                       download=download)

        testset = tv.datasets.CIFAR10(root=data_path,
                                      transform=transforms, train=False,
                                      download=False)

        return classes, trainset, testset


    if __name__ == '__main__':

        # some parameters:
        depth = 4
        # hyper-parameters per depth (resolution)
        num_epochs = [10, 20, 20, 20]
        fade_ins = [50, 50, 50, 50]
        batch_sizes = [128, 128, 128, 128]
        latent_size = 128

        # get the data. Ignore the test data and their classes
        _, dataset, _ = setup_data(download=True)

        # ======================================================================
        # This line creates the PRO-GAN
        # ======================================================================
        pro_gan = pg.ConditionalProGAN(num_classes=10, depth=depth, 
                                       latent_size=latent_size, device=device)
        # ======================================================================
    
        # ======================================================================
        # This line trains the PRO-GAN
        # ======================================================================
        pro_gan.train(
            dataset=dataset,
            epochs=num_epochs,
            fade_in_percentage=fade_ins,
            batch_sizes=batch_sizes
        )
        # ======================================================================  

## Thanks
Please feel free to open PRs / issues / suggestions here if 
you train on other datasets using this architecture. 
<br>

Best regards, <br>
@akanimax :)
