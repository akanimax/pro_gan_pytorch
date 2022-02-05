# pro_gan_pytorch
**Unofficial PyTorch** implementation of Paper titled "Progressive growing of GANs for improved 
Quality, Stability, and Variation". <br>
For the official TensorFlow code, please refer to 
[this repo](https://github.com/tkarras/progressive_growing_of_gans) <br>

![GitHub](https://img.shields.io/github/license/akanimax/pro_gan_pytorch)
![PyPi](https://img.shields.io/badge/pip--pro--gan--pth-3.4-brightgreen)

# How to use:
### Using the package
**Requirements (aka. we tested for):**
1. **Ubuntu** `20.04.3` or above
2. Python `3.8.3`
3. Nvidia GPU `GeForce 1080 Ti or above` min GPU-mem `8GB`
4. Nvidia drivers >= `470.86`
5. Nvidia cuda `11.3` | can be skipped since  pytorch ships with cuda, cudnn etc.

**Installing the package**
1. Easiest way is to create a new virtual-env 
so that your global python env doesn't get corrupted
2. Create and switch to your new virtual environment
```
    (your-machine):~$ python3 -m venv <env-store-path>/pro_gan_pth_env 
    (pro_gan_pth_env)(your-machine):~$ source <env-store-path>/pro_gan_pth_env/bin/activate
```
3. Install the `pro-gan-pth` package from pypi, if you meet 
all the above dependencies
```
    (pro_gan_pth_env)(your-machine):~$ pip install pro-gan-pth 
```    
4. Once installed, you can either use the installed commandline tools
`progan_train`, `progan_lsid` and `progan_fid`.
Note that the `progan_train` can be used with multiple gpus 
(If you have many :smile:). Just ensure that the gpus visible in the 
`CUDA_VISIBLE_DEVICES=0,1,2` environment variable. The other two tools only use a 
single GPU.


```
    (your-machine):~$ progan_train --help
    usage: Train Progressively grown GAN [-h] [--rec_dir REC_DIR] [--flip_horizontal FLIP_HORIZONTAL] [--depth DEPTH] [--num_channels NUM_CHANNELS] [--latent_size LATENT_SIZE] [--use_eql USE_EQL]
                                     [--use_ema USE_EMA] [--ema_beta EMA_BETA] [--epochs EPOCHS [EPOCHS ...]] [--batch_sizes BATCH_SIZES [BATCH_SIZES ...]] [--batch_repeats BATCH_REPEATS]
                                     [--fade_in_percentages FADE_IN_PERCENTAGES [FADE_IN_PERCENTAGES ...]] [--loss_fn LOSS_FN] [--g_lrate G_LRATE] [--d_lrate D_LRATE]
                                     [--num_feedback_samples NUM_FEEDBACK_SAMPLES] [--start_depth START_DEPTH] [--num_workers NUM_WORKERS] [--feedback_factor FEEDBACK_FACTOR]
                                     [--checkpoint_factor CHECKPOINT_FACTOR]
                                     train_path output_dir

    positional arguments:
      train_path            Path to the images folder for training the ProGAN
      output_dir            Path to the directory for saving the logs and models

    optional arguments:
      -h, --help            show this help message and exit
      --rec_dir REC_DIR     whether images stored under one folder or has a recursive dir structure (default: True)
      --flip_horizontal FLIP_HORIZONTAL
                            whether to apply mirror augmentation (default: True)
      --depth DEPTH         depth of the generator and the discriminator (default: 10)
      --num_channels NUM_CHANNELS
                            number of channels of in the image data (default: 3)
      --latent_size LATENT_SIZE
                            latent size of the generator and the discriminator (default: 512)
      --use_eql USE_EQL     whether to use the equalized learning rate (default: True)
      --use_ema USE_EMA     whether to use the exponential moving averages (default: True)
      --ema_beta EMA_BETA   value of the ema beta (default: 0.999)
      --epochs EPOCHS [EPOCHS ...]
                            number of epochs over the training dataset per stage (default: [42, 42, 42, 42, 42, 42, 42, 42, 42])
      --batch_sizes BATCH_SIZES [BATCH_SIZES ...]
                            batch size used for training the model per stage (default: [32, 32, 32, 32, 16, 16, 8, 4, 2])
      --batch_repeats BATCH_REPEATS
                            number of G and D steps executed per training iteration (default: 4)
      --fade_in_percentages FADE_IN_PERCENTAGES [FADE_IN_PERCENTAGES ...]
                            number of iterations for which fading of new layer happens. Measured in percentage (default: [50, 50, 50, 50, 50, 50, 50, 50, 50])
      --loss_fn LOSS_FN     loss function used for training the GAN. Current options: [wgan_gp, standard_gan] (default: wgan_gp)
      --g_lrate G_LRATE     learning rate used by the generator (default: 0.003)
      --d_lrate D_LRATE     learning rate used by the discriminator (default: 0.003)
      --num_feedback_samples NUM_FEEDBACK_SAMPLES
                            number of samples used for fixed seed gan feedback (default: 4)
      --start_depth START_DEPTH
                            resolution to start the training from. Example 2 --> (4x4) | 3 --> (8x8) ... | 10 --> (1024x1024)Note that this is not a way to restart a partial training. Resuming is not
                            supported currently. But will be soon. (default: 2)
      --num_workers NUM_WORKERS
                            number of dataloader subprocesses. It's a pytorch thing, you can ignore it ;). Leave it to the default value unless things are weirdly slow for you. (default: 4)
      --feedback_factor FEEDBACK_FACTOR
                            number of feedback logs written per epoch (default: 10)
      --checkpoint_factor CHECKPOINT_FACTOR
                            number of epochs after which a model snapshot is saved per training stage (default: 10)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

    (your-machine):~$ progan_lsid --help
    usage: ProGAN latent-space walk demo video creation tool [-h] [--output_path OUTPUT_PATH] [--generation_depth GENERATION_DEPTH] [--time TIME] [--fps FPS] [--smoothing SMOOTHING] model_path

    positional arguments:
      model_path            path to the trained_model.bin file

    optional arguments:
      -h, --help            show this help message and exit
      --output_path OUTPUT_PATH
                            path to the output video file location. Please only use mp4 format with this tool (.mp4 extension). I have banged my head too much to get anything else to work :(. (default:
                            ./latent_space_walk.mp4)
      --generation_depth GENERATION_DEPTH
                            depth at which the images should be generated. Starts from 2 --> (4x4) | 3 --> (8x8) etc. (default: None)
      --time TIME           number of seconds in the video (default: 30)
      --fps FPS             fps of the generated video (default: 60)
      --smoothing SMOOTHING
                            smoothness of walking in the latent-space. High values corresponds to more smoothing. (default: 0.75)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

    (your-machine):~$ progan_fid --help
    usage: ProGAN fid_score computation tool [-h] [--generated_images_path GENERATED_IMAGES_PATH] [--batch_size BATCH_SIZE] [--num_generated_images NUM_GENERATED_IMAGES] model_path dataset_path

    positional arguments:
      model_path            path to the trained_model.bin file
      dataset_path          path to the directory containing the images from the dataset. Note that this needs to be a flat directory

    optional arguments:
      -h, --help            show this help message and exit
      --generated_images_path GENERATED_IMAGES_PATH
                            path to the directory where the generated images are to be written. Uses a temporary directory by default. Provide this path if you'd like to see the generated images yourself
                            :). (default: None)
      --batch_size BATCH_SIZE
                            batch size used for generating random images (default: 4)
      --num_generated_images NUM_GENERATED_IMAGES
                            number of generated images used for computing the FID (default: 50000)
```
    
5. Or, you could import this as a python package in your code 
for more advanced use-cases:
``` 
    import pro_gan_pytorch as pg 
```
You can use all the modules in the package such as: `pg.networks.Generator`, 
`pg.networks.Discriminator`, `pg.gan.ProGAN` etc. Mostly, you'll only need 
the `pg.gan.ProGAN` module for training. For inference, you will probably only 
need the `pg.networks.Generator`. Please refer to the scripts for the tools as 
in 4. under `pro_gan_pytorch_scripts/` for examples on how to use the package. 
Besides, please feel free to just read the code. It's really easy to follow
(or at least I hope so :sweat_smile: :grimacing:).

### Developing the package
For more advanced use-cases in your project, or if you'd like to contribute new
features to this project, the following steps would help you get this project setup 
for developing. There are no standard set of rules for contributing here 
(no `CONTRIBUTING.md`) but let's try to maintain the overall ethos of the 
codebase :smile:.  

1. clone this repository
```
    (your-machine):~$ cd <path to project>
    (your-machine):<path to project>$ git clone https://github.com/akanimax/pro_gan_pytorch.git
```
2. Apologies in advance since the step 1. will take a while. I ended up 
pushing gifs and other large binary assets to git back then. 
I didn't know better :sad:. I'll see if this can be sorted out somehow. 
But once done setup a development virtual env, 
```
    (your-machine):<path to project>$ python3 -m venv pro-gan-pth-dev-env
    (your-machine):<path to project>$ source pro-gan-pth-dev-env/source/activate
```
3. Install the package in development mode:
```
    (pro-gan-pth-dev-env)(your-machine):<path to project>$ pip install -e .
```
4. Also install the dev requirements:
```
    (pro-gan-pth-dev-env)(your-machine):<path to project>$ pip install -r requirements-dev.txt
```
5. Now open the project in the editor of your choice, and you are good to go. 
I use `pytest` for testing and `black` for code formatting. Check out 
[this_link](https://black.readthedocs.io/en/stable/integrations/editors.html) for 
how to setup `black` with various IDEs.

6. There is no fancy CI, or automated testing, or docs building since this is a 
fairly tiny project. But we are open to considering these tools if more features
keep getting added to this project.

# Trained Models
We will be training models using this package on different datasets over the time.
Also, please feel free to open PRs for the following table if you end up training 
models for your own datasets. If you are contributing, then please setup 
a file hosting solution for serving the trained models. 

| Courtesy | Dataset        | Size  |Resolution  | GPUs used   | #Epochs per stage | Training time | FID score        | Link            | Qualitative samples | 
| :---     | :---           | :---  |:---        | :---        | :---              | :---          | :---             | :---            | :---                |
| @owang   | Metfaces       | ~1.3K |1024 x 1024 | 1 V100-32GB | 42                | 24 hrs        | 101.624          | [model_link](http://geometry.cs.ucl.ac.uk/projects/2021/pro_gan_pytorch/model_metfaces.bin) | ![image](https://drive.google.com/uc?export=view&id=1loYYvM_d1uG7CKtGkJRpKTwY5CQIldxm)
    

**Note that we compute the FID using the clean_fid version from 
[Parmar et. al.](https://www.cs.cmu.edu/~clean-fid/)**

# General cool stuff :smile: 
### Training timelapse (fixed latent points):
The training timelapse created from the images logged during the training 
looks really cool. 
<p align="center">
<img align="center" src ="https://github.com/akanimax/pro_gan_pytorch/blob/master/samples/celebA-HQ.gif"
     height=80% width=80%/>
</p>
<br>

Checkout this [YT video](https://www.youtube.com/watch?v=lzTm6Lq76Mo) for a 
4K version :smile:.

If interested please feel free to check out this 
[medium blog]( https://medium.com/@animeshsk3/the-unprecedented-effectiveness-of-progressive-growing-of-gans-37475c88afa3)
I wrote explaining the progressive growing technique.

# References

    1. Tero Karras, Timo Aila, Samuli Laine, & Jaakko Lehtinen (2018). 
    Progressive Growing of GANs for Improved Quality, Stability, and Variation. 
    In International Conference on Learning Representations.

    2. Parmar, Gaurav, Richard Zhang, and Jun-Yan Zhu. 
    "On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation." 
    arXiv preprint arXiv:2104.11222 (2021).

# Feature requests
- [ ] Conditional GAN support
- [ ] Tool for generating time-lapse video from the log images
- [ ] Integrating fid-metric computation as a training-logging

# Thanks
As always, <br>
please feel free to open PRs/issues/suggestions here. 
Hope this work is useful in your project :smile:. 

cheers :beers:! <br>
@akanimax :sunglasses:
