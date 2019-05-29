# pro_gan_pytorch
Package contains implementation of ProGAN.<br> 
Paper titled "Progressive growing of GANs for improved 
Quality, Stability, and Variation". <br>
link -> https://arxiv.org/abs/1710.10196 <br>
Trained Examples at -> https://github.com/akanimax/pro_gan_pytorch-examples

# :star: [New] Pretrained Models:
Please find the pretrained models under the `saved_models/` directory at the [drive_link](https://drive.google.com/drive/folders/1ex27dbFD_4Ycic6P9y3V9i63AvcuAe95)

# :star: [New] Demo:
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/demo.gif"
 height=80% width=80%/>
</p>
<br>

The repository now includes a latent-space interpolation animation demo under the `samples/` directory.
Just download all the pretrained weights from the above mentioned drive_link and put them in the `samples/` 
directory alongside the `demo.py` script. Note that there are a few tweakable parameters at the beginning
of the `demo.py` script so that you can play around with it. <br>

The demo loads up images for random points and then linearly interpolates among them to generate smooth 
animation. You need to have a good GPU (atleast GTX 1070) to see formidable FPS in the demo. The demo however 
can be optimized to do parallel generation of the images (It is completely sequential currently).

In order to load weights in the Generator, the process is the standard process for PyTorch model loading.
    
    import torch as th
    from pro_gan_pytorch import PRO_GAN as pg
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    gen = th.nn.DataParallel(pg.Generator(depth=9))
    gen.load_state_dict(th.load("GAN_GEN_SHADOW_8.pth", map_location=str(device)))

### Notes for the Above code:
1. Create a new generator module using pg (depth = 9 means the generating resolution will be 1024 x 1024). <br>
2. Note that DataParallel is required here because I have trained the models on Multiple GPUs. <br>
   you wouldn't need to wrap the Generator into a DataParallel if you train on CPU. <br>
   Which I don't think is feasible for a GAN in general (:D). <br>
3. You can simply load the weights into the gen as it is implemented as a PyTorch Module. <br>
4. map_location arg takes care of Device mismatch. As in, if you trained on GPU but inferring on CPU. <br>
5. **Also note that we need to use the `GAN_GEN_SHADOW_8.pth` model and not `GAN_GEN_8.pth`.** <br>
   **The shadow model contains the Exponential Moving Averaged weights (stable weights).**

# Exemplar Samples :)
### Training gif (fixed latent points):
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/celebA-HQ.gif"
     height=80% width=80%/>
</p>
<br>

### Generated Samples:
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/faces_sheet_1.png"
 height=80% width=80%/>
</p>
<br>
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/faces_sheet_2.png"
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
