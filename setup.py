from setuptools import setup, find_packages

setup(
    name='pro-gan-pth',
    version='2.0',
    packages=find_packages("."),
    url='https://github.com/akanimax/pro_gan_pytorch',
    license='MIT',
    author='akanimax',
    author_email='akanimax@gmail.com',
    description='ProGAN package implemented as an extension of PyTorch nn.Module',
    install_requires=['numpy', 'torch', 'torchvision']
)
