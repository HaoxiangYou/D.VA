# License: see [LICENSE, LICENSES/DiffRL/LICENSE]

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from pathlib import Path

import hydra
import numpy as np
import torch
import copy
import pickle
import dm_env
from dm_env import specs
from tensorboardX import SummaryWriter
import envs
from externals.drqv2 import drqv2_utils
from externals.drqv2 import dmc
from externals.drqv2.replay_buffer import ReplayBufferStorage, make_replay_loader
from viewer.video_recorder import VideoRecorder
from utils.common import *
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
import time
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(os.path.join(project_dir, "externals/drqv2"))

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class MakeDMfromdFlex(dm_env.Environment):
    def __init__(self, cfg, eval):
        self.eval = eval
        self.cfg = cfg
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        seeding(cfg["params"]["general"]["seed"])

        self.env =  env_fn(num_envs = 1 if self.eval else cfg["params"]["config"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            img_height = cfg["params"]["config"].get("img_height", 84),\
                            img_width = cfg["params"]["config"].get("img_width", 84),\
                            seed = cfg["params"]["general"]["seed"], \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            no_grad = True)
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.num_actions
        # self.render_size = 256 # fixed due to the data mismatch with TrainVideoRecorder
        # self.camera_id = 0 # render camera id. 
        self.device = cfg["params"]["general"]["device"]
        self.raw_rew = np.zeros((self.env.num_envs)) 
        self.sim_dt = self.env.sim_dt
        if hasattr(self.env, 'observation_spec'):
            self._observation_spec = self._env.observation_spec()
        else:
            self._observation_spec = specs.BoundedArray(self.env.num_vis_obs,
                                                        minimum= 0,
                                                        maximum= 255,
                                                        dtype='uint8',
                                                        name='observation')
        if hasattr(self.env, 'action_spec'):
            self._action_spec = self._env.action_spec()
        else:
            self._action_spec = specs.BoundedArray((self.env.num_actions, ),
                                                   minimum=-1,
                                                   maximum=1,
                                                   dtype='float32',
                                                   name='action')
        self._reward_spec = specs.Array(shape=(), dtype=np.dtype('float32'), name='reward')
        self._discount_spec = specs.BoundedArray(
        shape=(), dtype='float32', minimum=0.0, maximum=1.0, name='discount')
        if hasattr(self.env, 'discount_spec'):
            self._discount_spec = self.env.discount_spec()

    def reset(self, env_ids = None, force_reset = True, enable_vis_obs=True):
        # return stacked observation (9 * width * height)
        self.env.clear_grad()
        obs = self.env.reset(env_ids=env_ids, 
                             force_reset=force_reset,
                             enable_vis_obs=enable_vis_obs)
        vis_obs_batch = (obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")
        return [dm_env.TimeStep(step_type=dm_env.StepType.FIRST, 
                            reward=None,
                            discount=1.0,
                            observation=vis_obs) for vis_obs in vis_obs_batch]
         
    
    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        obs, rew_batch, done_batch, extra_info = self.env.step(actions = torch.tanh(torch.tensor(actions, dtype = torch.float32, device = self.device)), 
                                                               enable_reset = enable_reset, 
                                                               enable_vis_obs = enable_vis_obs)
        del extra_info
        self.raw_rew[:] = rew_batch.detach().clone().cpu().numpy()
        vis_obs_batch = (obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")
        return [dm_env.TimeStep(step_type=dm_env.StepType.MID if not done else dm_env.StepType.LAST,
                            reward=rew.detach().clone().cpu().item(),
                            discount=1.0,
                            observation=vis_obs) for (vis_obs, rew, done) in zip(vis_obs_batch, rew_batch, done_batch)]
    
    def observation_spec(self):
        return self._observation_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def action_spec(self):
        return self._action_spec
    
    def discount_spec(self):
        return self._discount_spec
    
class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.num_envs = cfg["params"]["config"]["num_actors"]
        self.img_height = cfg["params"]["config"].get("img_height", 84)
        self.img_width = cfg["params"]["config"].get("img_width", 84)
        self.if_render = cfg["params"]["general"]["render"]
        drqv2_utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = drqv2_utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._num_episode_finished = 0
        self.time_report = TimeReport()
        self.writer = SummaryWriter(os.path.join(self.work_dir, 'tb'))
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.vis_obs_buffer = torch.zeros(
            (self.num_envs, 9, self.img_height , self.img_width ), device=self.device, dtype=torch.uint8, requires_grad=False)
        self.iter_count = 0
        self.step_count = 0
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device=self.device)
        self._current_episodes = [copy.deepcopy(defaultdict(list)) for _ in range(self.num_envs)]
        self.episode_loss_his = []
        self.episode_length_his = []
        
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.best_policy_loss = np.inf

    def setup(self):
        train_env = MakeDMfromdFlex(self.cfg, False)
        eval_env = MakeDMfromdFlex(self.cfg, True)
        self.env = train_env
        self.train_env = dmc.make_env(train_env, self.cfg)
        self.eval_env = dmc.make_env(eval_env, self.cfg)
        # create replay buffer
        action_spec = specs.BoundedArray((self.train_env.num_actions, ),
                                                   minimum=-1,
                                                   maximum=1,
                                                   dtype='float32',
                                                   name='action')
        self.data_specs = (self.train_env.observation_spec(),
                      action_spec, # self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(self.data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = None 
        if self.if_render:
            self.video_recorder = VideoRecorder(fps=int(1/self.eval_env.sim_dt))

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def save_video(self, joint_qs, save_dir=None, max_video_length=800):
        self.video_recorder.update_save_dir(save_dir)
        frames = self.eval_env.env.render_traj(joint_qs[:max_video_length], recording=True)
        for i in range(frames.shape[1]):
            self.video_recorder.reset()
            for j in range(frames.shape[0]):
                self.video_recorder.append(frames[j, i])
            self.video_recorder.save("eval_traj_{}.mp4".format(i))

    def eval(self, file_name):
        eval_until_episode = drqv2_utils.Until(self.cfg.num_eval_episodes)
        step, episode, total_reward = 0, 0, 0
        save_dir = self.work_dir / "eval" / file_name 
        joint_qs = []
        os.makedirs(save_dir, exist_ok=True)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset(env_ids = None, force_reset = True)
            joint_qs.append(self.eval_env.env.state.joint_q.view(self.eval_env.num_envs, -1).detach().clone())
            while not time_step[0].last():
                with torch.no_grad(), drqv2_utils.eval_mode(self.agent):
                    actions = self.agent.act(time_step[0].observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(actions=actions, 
                                                enable_reset = False, 
                                                enable_vis_obs = True)
                joint_qs.append(self.eval_env.env.state.joint_q.view(self.eval_env.num_envs, -1).detach().clone())
                total_reward += time_step[0].reward
                step += 1
            episode += 1
            if time_step[0].last():
                self.eval_env.reset(force_reset = True, enable_vis_obs=True)

        if self.video_recorder is not None:
            joint_qs = torch.stack(joint_qs)
            self.save_video(joint_qs, save_dir=save_dir)
        return total_reward / episode

    def process_time_steps(self, store_time_steps):
        done_ids = []
        for idx, time_step in enumerate(store_time_steps):
            # store the observation 
            for spec in self.data_specs:
                value = time_step[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episodes[idx][spec.name].append(value)
            if time_step.last():
                done_ids.append(idx)
                # if the current episode ends, we store the episode 
                episode = dict()
                for spec in self.data_specs:
                    value = self._current_episodes[idx][spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
                    # we only save episodes that finished
                self._current_episodes[idx] = copy.deepcopy(defaultdict(list))
                self.replay_storage._store_episode(episode)
        
        self.episode_length += 1
        # Finally, process the reward for logging
        with torch.no_grad():
            self.episode_loss -= torch.tensor(self.train_env.raw_rew, dtype=torch.float32, device=self.device)
            if len(done_ids)>0:
                self.episode_loss_meter.update(self.episode_loss[done_ids])
                self.episode_length_meter.update(self.episode_length[done_ids])
                for done_env_id in done_ids:
                    if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                        print('ep loss error')
                        raise ValueError
                    self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                    self.episode_length_his.append(self.episode_length[done_env_id].item())
                    self.episode_loss[done_env_id] = 0.
                    self.episode_length[done_env_id] = 0
                self.train_env.reset(env_ids=np.array(done_ids, dtype=np.int32), enable_vis_obs=True)
        
        self._global_episode += len(done_ids)
        self._num_episode_finished += len(done_ids)

    def train(self):
        # predicates
        self.start_time = time.time()
        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")
        self.time_report.add_timer("evaluation time") 
        self.time_report.add_timer("env step time")
        self.time_report.add_timer("IO Time")
        self.time_report.add_timer("save snapshot")
        
        self.time_report.start_timer("algorithm")

        train_until_step = drqv2_utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = drqv2_utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = drqv2_utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        save_every_step = drqv2_utils.Every(self.cfg.save_every_frames,
                                      self.cfg.action_repeat)

        actor_step, episode_step, episode_reward = 0, 0, 0
        self.time_report.start_timer("IO Time")
        time_steps = self.train_env.reset(env_ids = None, force_reset = True)
        self.process_time_steps(time_steps)
        self.time_report.end_timer("IO Time")

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.step_count):
                print_info("Start Evaluation with maximum trajectory length:{}".format(torch.max(self.episode_length).item()))
                eval_start_time = time.time()

                self.time_report.start_timer("evaluation time")
                save_dir = os.path.join(self.work_dir, "eval/iter_{}".format(episode_step))
                mean_eval_policy_loss = self.eval(file_name="iter_{}".format(episode_step))
                
                self.save(save_dir=save_dir, filename=self.cfg["params"]["diff_env"]["name"] + "policy_iter{}_reward{:.3f}".format(episode_step, -mean_eval_policy_loss))
                self.save_training_summary(save_dir=save_dir)
                self.time_report.end_timer("evaluation time")
                print_info("Evaluation done in {} seconds".format(time.time()-eval_start_time))

            time_start_epoch = time.time()
            # sample action
            # output action with shape (num_envs, action_size)
            with torch.no_grad(), drqv2_utils.eval_mode(self.agent):
                self.time_report.start_timer("IO Time")
                obs = torch.from_numpy(np.array([time_step.observation for time_step in time_steps], dtype=np.uint8)).to(self.device)
                self.time_report.end_timer("IO Time")
                actions = self.agent.act(obs, 
                                        self.global_step,
                                        eval_mode=False,
                                        time_report = self.time_report)
            
            # try to update the agent
            self.time_report.start_timer("backward simulation")
            if not seed_until_step(self.global_step):
                for i in range(self.global_step, self.global_step + self.num_envs):
                    metrics = self.agent.update(self.replay_iter, i, self.time_report)
                actor_step += 1
                self._num_episode_finished = 0
            self.time_report.end_timer("backward simulation")

            if self.cfg.save_snapshot and save_every_step(episode_step):
                self.time_report.start_timer("save snapshot")
                self.save_snapshot()
                self.time_report.end_timer("save snapshot")
                print_info("Snapshot saved at step {}".format(episode_step))

            # take env step       
            time_start_steps = time.time()
            self.time_report.start_timer("forward simulation")
            self.time_report.start_timer("env step time")
            time_steps = self.train_env.step(actions=actions, 
                                             enable_reset = False, 
                                             enable_vis_obs = True)
            self.time_report.end_timer("env step time")
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("IO Time")
            time_end_steps = time.time()
            episode_reward += np.sum([time_step.reward for time_step in time_steps])
            
            self.process_time_steps(time_steps)
            self.time_report.end_timer("IO Time")
            self.step_count += self.num_envs * self.cfg.action_repeat
            episode_step += 1

            self._global_step += self.num_envs
            
            # logging
            self.time_report.start_timer("IO Time")
            time_elapse = time.time() - self.start_time - self.time_report.timers["evaluation time"].time_total
            if (len(self.episode_loss_his) > 0):
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_episode_length = self.episode_length_meter.get_mean()
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, actor_step)
                if mean_policy_loss < self.best_policy_loss:
                    self.best_policy_loss = mean_policy_loss
                    self.save()
                    print_info("Best policy saved with loss: {:2f}".format(mean_policy_loss))
            else:
                mean_policy_loss = np.inf
                mean_episode_length = 0
            self.time_report.end_timer("IO Time")

            self.writer.flush()
            time_end_epoch = time.time()
            print('iter {}: ep loss {:.2f}, ep len {}, fps episode {:.2f}, fps env steps {:2f}'.format(\
                        episode_step, mean_policy_loss, mean_episode_length, 
                        1 / (time_end_epoch - time_start_epoch), self.cfg.action_repeat * self.num_envs / (time_end_steps - time_start_steps)))
        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.save_training_summary()

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def save_training_summary(self, save_dir = None):
        if save_dir is None:
            save_dir = self.work_dir
        
        time_report = {}
        for timer_name in self.time_report.timers.keys():
            time_report.update({timer_name: self.time_report.timers[timer_name].time_total})

        training_summary = {
        "time_report": time_report,
        "env_step": self.step_count 
        }

        with open(os.path.join(save_dir, "training_summary.pkl"), "wb") as f:
            pickle.dump(training_summary, f)
    
    def save(self, filename = None, save_dir = None):
        if save_dir is None:
            save_dir = self.work_dir
        if filename is None:
            filename = 'best_policy'
        torch.save([self.agent.actor, self.agent.critic, self.agent.critic_target], os.path.join(save_dir, "{}.pt".format(filename)))


@hydra.main(config_path='cfg/drqv2', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()