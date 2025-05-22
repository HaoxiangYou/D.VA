import numpy as np

class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle = False, drop_last = False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()
        
        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx]}


class ActorSupervisedDataset:
    def __init__(self, batch_zie, obs, target_actions, action_eps=None, shuffle=True, drop_last=False):
        self.batch_size = batch_zie
        # Here we assume obs, target_action (action_eps if not None) in (batch_size, additional_dim)
        self.obs = obs
        self.target_actions = target_actions
        self.action_eps = action_eps

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index]
        self.target_actions = self.target_actions[index]
        if self.action_eps is not None:
            self.action_eps = self.action_eps[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        batch_sample = {'obs': self.obs[start_idx:end_idx], 'target_actions': self.target_actions[start_idx:end_idx]}
        if self.action_eps is not None:
            batch_sample["action_eps"] = self.action_eps[start_idx:end_idx]
        return batch_sample
    

class ExpertReplayBuffer(object):    
    def __init__(self, state_obs_shape, action_shape, vis_obs_shape=None, max_size=1000000):

        self.max_size = max_size

        self.state_obs = np.zeros((max_size, state_obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, action_shape), dtype=np.float32)
        if vis_obs_shape is not None:
            self.vis_obs = np.zeros((max_size, *vis_obs_shape), dtype=np.uint8)
        self.size = 0
        self.pos = 0

    def __len__(self):
        return self.size
        
    def append(self, control_action_pairs):

        state_obs = control_action_pairs["state_obs"].cpu().numpy()
        actions = control_action_pairs["actions"].cpu().numpy()
        vis_obs = None
        if "vis_obs" in control_action_pairs:
            vis_obs = control_action_pairs["vis_obs"].cpu().numpy()

        batch_size = state_obs.shape[0]
        if self.pos + batch_size > self.max_size:
            batch_size = self.max_size - self.pos 
        self.state_obs[self.pos:self.pos+batch_size] = state_obs[:batch_size]
        self.actions[self.pos:self.pos+batch_size] = actions[:batch_size]
        if vis_obs is not None:
            self.vis_obs[self.pos:self.pos+batch_size] = vis_obs[:batch_size]
        self.size = min(self.size + batch_size, self.max_size)
        self.pos = (self.pos + batch_size) % self.max_size
