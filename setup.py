from setuptools import find_packages, setup

with open("README.md", "r") as file_:
    project_description = file_.read()

with open("requirements.txt", "r") as file_:
    project_requirements = file_.read().split("\n")

setup(
    name="pro-gan-pth",
    version="3.3",
    packages=find_packages(".", exclude=("test", "samples")),
    url="https://github.com/akanimax/pro_gan_pytorch",
    license="MIT",
    author="akanimax",
    author_email="akanimax@gmail.com",
    description="ProGAN package implemented as an extension of PyTorch nn.Module",
    long_description=project_description,
    install_requires=project_requirements,
    entry_points={
        "console_scripts": [
            f"progan_train=scripts.train:main",
            f"progan_lsid=scripts.latent_space_interpolation:main",
            f"progan_fid=scripts.compute_fid:main",
        ]
    },
)
