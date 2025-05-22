import sys, os
proj_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(proj_dir)
import torch
import matplotlib.pyplot as plt
from robot_renderer.cartpole_renderer import CartpoleRenderer
renderer = CartpoleRenderer()

pos = torch.tensor(
            [[0., 0.],
            [0, -1], 
            [1.0, 1.0]], 
            device="cuda", 
            requires_grad=True)

images = renderer.render(pos)
loss = torch.sum((images[0] - images[1])**2)
loss.backward()
print(pos.grad)
images = images.detach().cpu().numpy()
for i in range(images.shape[0]):
    plt.figure(i)
    plt.imshow(images[i,:,:,:3])
plt.show()