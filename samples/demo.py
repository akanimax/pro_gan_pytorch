import torch as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pro_gan_pytorch import PRO_GAN as pg

# ==========================================================================
# Tweakable parameters
# ==========================================================================
depth = 8
num_points = 10
transition_points = 90
# ==========================================================================

# create the device for running the demo:
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# load the model for the demo
gen = th.nn.DataParallel(pg.Generator(depth=9))
gen.load_state_dict(th.load("GAN_GEN_SHADOW_8.pth", map_location=str(device)))

# function to generate an image given a latent_point
def get_image(point):
    img = gen(point, depth=depth, alpha=1).detach().squeeze(0).permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    return img

# generate the set of points:
fixed_points = th.randn(num_points, 512)
points = []  # start with an empty list
for i in range(len(fixed_points) - 1):
    pt_1 = fixed_points[i].view(1, -1)
    pt_2 = fixed_points[i + 1].view(1, -1)
    direction = pt_2 - pt_1
    for j in range(transition_points):
        pt = pt_1 + ((direction / transition_points) * j)
        points.append(pt)
    # also append the final point:
    points.append(pt_2)

start_point = points[0]
points = points[1:]

fig, ax = plt.subplots()
plt.axis("off")
shower = plt.imshow(get_image(start_point))

def init():
    return shower,

def update(point):
    shower.set_data(get_image(point))
    return shower,

ani = FuncAnimation(fig, update, frames=points,
                    init_func=init, blit=False	)
plt.show()
