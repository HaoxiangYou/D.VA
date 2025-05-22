import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from models import model_utils
from models.encoder import Encoder

class ActorDeterministicMLP(nn.Module):
    def __init__(self, state_obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.enable_vis_obs = cfg_network.get("vis_obs", False)
        if self.enable_vis_obs:
            vis_obs_dim = (9, cfg_network['img_height'], cfg_network['img_width'])
            self.encoder = Encoder(obs_shape=vis_obs_dim,
                                   output_dim=cfg_network["actor_mlp"]['units'][0])
            self.layer_dims = [cfg_network["actor_mlp"]['units'][0]] + cfg_network['actor_mlp']['units'] + [action_dim]
        else:
            self.layer_dims = [state_obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.actor = nn.Sequential(*modules).to(device)
        
        self.action_dim = action_dim

        if self.enable_vis_obs:
            self.obs_dim = vis_obs_dim
        else:
            self.obs_dim = state_obs_dim

        if self.enable_vis_obs:
            print(self.encoder)
        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic=False):
        if self.enable_vis_obs:
            observations = self.encoder(observations)

        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, state_obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.enable_vis_obs = cfg_network.get("vis_obs", False)
        if self.enable_vis_obs:
            vis_obs_dim = (9, cfg_network['img_height'], cfg_network['img_width'])
            self.encoder = Encoder(obs_shape=vis_obs_dim,
                                   output_dim=cfg_network["actor_mlp"]['units'][0])
            self.layer_dims = [cfg_network["actor_mlp"]['units'][0]] + cfg_network['actor_mlp']['units'] + [action_dim]
        else:
            self.layer_dims = [state_obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))
            
        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim

        if self.enable_vis_obs:
            self.obs_dim = vis_obs_dim
        else:
            self.obs_dim = state_obs_dim

        if self.enable_vis_obs:
            print(self.encoder)
        print(self.mu_net)
        print(self.logstd)
    
    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False):

        if self.enable_vis_obs:
            obs = self.encoder(obs)

        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp() # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample
    
    def forward_with_dist(self, obs, deterministic = False):

        if self.enable_vis_obs:
            obs = self.encoder(obs)

        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std
        
    def evaluate_actions_log_probs(self, obs, actions):

        if self.enable_vis_obs:
            obs = self.encoder(obs)

        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)