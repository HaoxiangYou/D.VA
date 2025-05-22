import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
import matplotlib.pyplot as plt
from robot_renderer.hopper_renderer import HopperRenderer

renderer = HopperRenderer(img_height=256, img_width=256)

qpos = torch.tensor(
            [[0., 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0]], 
            device="cuda", 
            requires_grad=True)

images = renderer.render(qpos)
loss = torch.sum((images[0] - images[1])**2)
loss.backward()
print(qpos.grad)
images = images.detach().cpu().numpy()
for i in range(images.shape[0]):
    plt.figure(i)
    plt.imshow(images[i,:,:,:3])
plt.show()