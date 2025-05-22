import torch.nn as nn
import numpy as np
from models import model_utils

# Drqv2-style encoder
class Encoder(nn.Module):
    def __init__(self, obs_shape, output_dim, device='cuda'):
        super().__init__()

        assert len(obs_shape) == 3
        # This is currently hardcode for stacked images of size (3*3, 84, 84)
        self.repr_dim = 32 * 35 * 35
        self.output_dim = output_dim
        self.device = device

        conv_init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
        
        linear_init = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), 1.0)

        self.convnet = nn.Sequential(conv_init_(nn.Conv2d(obs_shape[0], 32, 3, stride=2)),
                                    nn.ReLU(), 
                                    conv_init_(nn.Conv2d(32, 32, 3, stride=1)),
                                    nn.ReLU(), 
                                    conv_init_(nn.Conv2d(32, 32, 3, stride=1)),
                                    nn.ReLU(), 
                                    conv_init_(nn.Conv2d(32, 32, 3, stride=1)),
                                    nn.ReLU()).to(self.device)
        
        self.trunk = nn.Sequential(linear_init(nn.Linear(self.repr_dim, self.output_dim)),
                                   nn.LayerNorm(self.output_dim), nn.Tanh()).to(self.device)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h)
        return h