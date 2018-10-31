# pro_gan_pytorch
Package contains implementation of ProGAN. 
Paper titled "Progressive growing of GANs for improved 
Quality, Stability, and Variation". <br>
link -> https://arxiv.org/abs/1710.10196 <br>
Trained Examples at -> https://github.com/akanimax/pro_gan_pytorch-examples

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
 `pg.ProGAN`. Mostly, you'll only need the ProGAN module.

4.) Example Code for CIFAR-10 dataset:

    import torch as th
    import torchvision as tv
    import pro_gan_pytorch.PRO_GAN as pg

    # select the device to be used for training
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    data_path = "cifar-10/"

    def setup_data(batch_size, num_workers, download=False):
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
        trainloader = th.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

        testset = tv.datasets.CIFAR10(root=data_path,
                                      transform=transforms, train=False,
                                      download=False)
        testloader = th.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

        return classes, trainloader, testloader


    if __name__ == '__main__':

        # some parameters:
        depth = 4
        num_epochs = 100  # number of epochs per depth (resolution)
        latent_size = 128

        # get the data. Ignore the test data and their classes
        _, train_data_loader, _ = setup_data(batch_size=32, num_workers=3, download=True)

        # ======================================================================
        # This line creates the PRO-GAN
        # ======================================================================
        pro_gan = pg.ProGAN(depth=depth, latent_size=latent_size, device=device)
        # ======================================================================

        # train the pro_gan using the cifar-10 data
        for current_depth in range(depth):
            print("working on depth:", current_depth)

            # note that the rest of the api indexes depth from 0
            for epoch in range(1, num_epochs + 1):
                print("\ncurrent_epoch: ", epoch)

                # calculate the value of aplha for fade-in effect
                alpha = epoch / num_epochs
                print("value of alpha:", alpha)

                # iterate over the dataset in batches:
                for i, batch in enumerate(train_data_loader, 1):
                    images, _ = batch
                    images = images.to(device)
                    # generate some random noise:
                    noise = th.randn(images.shape[0], latent_size).to(device)

                    # optimize discriminator:
                    dis_loss = pro_gan.optimize_discriminator(noise, images, current_depth, alpha)

                    # optimize generator:
                    gen_loss = pro_gan.optimize_generator(noise, current_depth, alpha)

                    print("Batch: %d  dis_loss: %.3f  gen_loss: %.3f"
                          % (i, dis_loss, gen_loss))

                print("epoch finished ...")

        print("training complete ...")
        
# #TODO
1.) ~~Add the conditional PRO_GAN module~~ (added in commit [ee7cf00b5f3e747c61e293a88f3e2f656117fcc2](https://github.com/akanimax/pro_gan_pytorch/commit/ee7cf00b5f3e747c61e293a88f3e2f656117fcc2))<br>
2.) Setup the travis - checker. (I have to figure out some good unit tests too :D lulz!) <br>
3.) Write an informative README.rst (although it is rarely read) <br>
