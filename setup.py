from setuptools import setup

with open("requirements.txt", "r") as file_:
    project_requirements = file_.read().split("\n")

setup(
    name="pro-gan-pth",
    version="3.4",
    packages=["pro_gan_pytorch", "pro_gan_pytorch_scripts"],
    url="https://github.com/akanimax/pro_gan_pytorch",
    license="MIT",
    author="akanimax",
    author_email="akanimax@gmail.com",
    setup_requires=['wheel'],
    description="ProGAN package implemented as an extension of PyTorch nn.Module",
    install_requires=project_requirements,
    entry_points={
        "console_scripts": [
            f"progan_train=pro_gan_pytorch_scripts.train:main",
            f"progan_lsid=pro_gan_pytorch_scripts.latent_space_interpolation:main",
            f"progan_fid=pro_gan_pytorch_scripts.compute_fid:main",
        ]
    },
)
