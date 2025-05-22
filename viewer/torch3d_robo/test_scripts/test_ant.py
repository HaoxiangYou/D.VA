import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
import matplotlib.pyplot as plt
from robot_renderer.ant_renderer import AntRenderer
renderer = AntRenderer()

pos = torch.tensor(
            [[0, 0, 2.],
            [0, 0, 0]], 
            device="cuda", 
            requires_grad=True)
quat = torch.tensor(
            [[1, 0, 0, 0],
            [2**0.5/2, 0, 2**0.5/2, 0]], 
            device="cuda", 
            requires_grad=True)
joints = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0.]], 
            device="cuda", 
            requires_grad=True)
qpos = torch.concat([pos, quat, joints], dim=-1)

images = renderer.render(qpos)

loss = torch.sum((images[0] - images[1])**2)
loss.backward()
print(pos.grad)
images = images.detach().cpu().numpy()
for i in range(images.shape[0]):
    plt.figure(i)
    plt.imshow(images[i,:,:,:3])
plt.show()