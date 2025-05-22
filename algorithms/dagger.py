import sys, os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import numpy as np
import torch
import models.actor
import copy
import yaml
import envs
import time
import pickle
from tensorboardX import SummaryWriter
from models.actor import ActorStochasticMLP, ActorDeterministicMLP
from utils.time_report import TimeReport
from utils.common import *
from utils.dataset import ExpertReplayBuffer
from utils.average_meter import AverageMeter
from viewer.video_recorder import VideoRecorder

class Dagger:
    def __init__(self, cfg, teacher_policy, teacher_training_info):
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        self.env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            seed = cfg["params"]["general"]["seed"], \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            no_grad = True)

        self.num_envs = self.env.num_envs
        self.num_state_obs = self.env.num_state_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.enable_vis_obs = cfg["params"]["config"].get("vis_obs", True)
        self.device = cfg["params"]["general"]["device"]

        print('num_envs = ', self.num_envs)
        print('num_actions = ', self.num_actions)
        print('num_state_obs = ', self.num_state_obs)
        if self.enable_vis_obs:
            self.num_vis_obs = self.env.num_vis_obs
            print('num_vis_obs =', self.num_vis_obs)

        # video recorder
        self.video_recorder = VideoRecorder(fps=int(1/self.env.sim_dt))

        self.iter_count = 0
        self.supervised_iter_count = 0

        self.teacher_policy = teacher_policy
        # update teacher policy logs
        self.step_count = teacher_training_info["step_count"]
        self.teacher_training_time = teacher_training_info["training_time"]

        # time report
        self.time_report = TimeReport()
        self.time_report.add_timer("teacher_training")
        self.time_report.timers["teacher_training"].time_total = self.teacher_training_time

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.name = cfg['params']['config']["name"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.learning_starts = cfg["params"]["config"]["learning_starts"]
        self.batch_size = cfg["params"]["config"]["batch_size"]
        self.num_update_per_epoch = cfg["params"]["config"]["num_update_per_epoch"]
        self.supervised_learning_threshold = cfg["params"]["config"]["supervised_loss_threshold"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')

        # saving config files
        if cfg['params']['general']['train']:
            self.log_dir = cfg["params"]["general"]["logdir"] 
            os.makedirs(self.log_dir, exist_ok = True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (cfg['params']['config']['player'].get('determenistic', False) or cfg['params']['config']['player'].get('deterministic', False))

        # create actor network
        self.actor_name = cfg["params"]["network"].get("actor", 'ActorStochasticMLP')
        actor_fn = getattr(models.actor, self.actor_name)
        self.actor = actor_fn(self.num_state_obs, self.num_actions, cfg['params']['network'], device = self.device)

        if cfg['params']['general']['train']:
            self.save('init_policy')

        # replay buffer
        self.state_obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_state_obs), dtype = torch.float32, device = self.device)
        if self.enable_vis_obs:   
            self.vis_obs_buf = torch.zeros((self.steps_num, self.num_envs) + self.num_vis_obs, dtype=torch.uint8, device = self.device)
        self.max_replay_buffer_size = cfg["params"]["config"]["max_replay_buffer_size"]
        if self.enable_vis_obs:
            self.replay_buffer = ExpertReplayBuffer(state_obs_shape=self.num_state_obs, action_shape=self.num_actions, 
                                                    vis_obs_shape=self.num_vis_obs, max_size=self.max_replay_buffer_size)
        else:
            self.replay_buffer = ExpertReplayBuffer(state_obs_shape=self.num_state_obs, 
                                                    action_shape=self.num_actions, max_size=self.max_replay_buffer_size)
    
        # average meter
        self.episode_rewards_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_his = []
        self.episode_rewards_his = []
        self.best_policy_rewards = -np.inf

        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.actor_lr)
    
    def initialize_env(self):
        self.env.reset(enable_vis_obs=self.enable_vis_obs)

    @torch.no_grad()
    def sample_trajectories(self, deterministic = False):
        
        # initial trajectory 
        obs = self.env.initialize_trajectory(enable_vis_obs=self.enable_vis_obs)
        state_obs = obs["state_obs"]
        if self.enable_vis_obs:
            vis_obs = obs["vis_obs"]

        for i in range(self.steps_num):
            self.state_obs_buf[i] = state_obs.clone()
            if self.enable_vis_obs:
                self.vis_obs_buf[i] = vis_obs.clone()
            if self.enable_vis_obs:
                actions = self.actor(vis_obs, deterministic=deterministic)
            else:
                actions = self.actor(state_obs, deterministic=deterministic)

            # step forward
            obs, rew, done, extra_info = self.env.step(torch.tanh(actions), enable_vis_obs=self.enable_vis_obs, enable_reset=True)
            state_obs = obs["state_obs"]
            if self.enable_vis_obs:
                vis_obs = obs["vis_obs"]

            # logging
            self.episode_length += 1
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)
            self.episode_rewards += rew
            if len(done_env_ids) > 0:
                self.episode_rewards_meter.update(self.episode_rewards[done_env_ids])
                self.episode_length_meter.update(self.episode_length[done_env_ids])
                for done_env_id in done_env_ids:
                    self.episode_rewards_his.append(self.episode_rewards[done_env_id].item())
                    self.episode_length_his.append(self.episode_length[done_env_id].item())
                    self.episode_rewards[done_env_id] = 0.
                    self.episode_length[done_env_id] = 0

        trajs = {"state_obs": self.state_obs_buf.clone()}
        if self.enable_vis_obs:
            trajs["vis_obs"] = self.vis_obs_buf.clone()

        self.step_count += self.steps_num * self.num_envs

        return trajs        

    def reshape_trajs(self, trajs):
        for key in trajs.keys():
            trajs[key] = trajs[key].reshape(-1, *trajs[key].shape[2:])
        return trajs
    
    def compute_supervised_loss(self, obs, actions):
        if isinstance(self.actor, ActorStochasticMLP):
            logp = self.actor.evaluate_actions_log_probs(obs, actions)
            return -torch.mean(logp)
        elif isinstance(self.actor, ActorDeterministicMLP):
            return torch.nn.functional.mse_loss(self.actor(obs), actions)

    def train(self):
        self.start_time = time.time()

        # add timer
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("expert correction")
        self.time_report.add_timer("evaluation")
        self.time_report.add_timer("supervised learning")
        self.time_report.add_timer("IO time")

        self.time_report.start_timer("algorithm")
        
        self.episode_rewards = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)

        self.initialize_env()

        for epoch in range(self.max_epochs):

            time_start_epoch = time.time()

            # sample trajectory
            self.time_report.start_timer("forward simulation")
            trajs = self.sample_trajectories()
            trajs = self.reshape_trajs(trajs)
            self.time_report.end_timer("forward simulation")

            # provide expert action
            self.time_report.start_timer("expert correction")
            with torch.no_grad():
                trajs["actions"] = self.teacher_policy(trajs["state_obs"])
            self.time_report.end_timer("expert correction")

            self.time_report.start_timer("IO time")
            self.replay_buffer.append(trajs)
            self.time_report.end_timer("IO time")

            # apply supervised learning
            if len(self.replay_buffer) < self.learning_starts:
                supervised_loss = np.inf
            else:
                self.time_report.start_timer("supervised learning")
                for i_update in range(self.num_update_per_epoch):

                    # learning rate schedule
                    if self.lr_schedule == 'linear':
                        actor_lr = (1e-5 - self.actor_lr) * float(self.supervised_iter_count / (self.max_epochs * self.num_update_per_epoch)) + self.actor_lr
                        for param_group in self.actor_optimizer.param_groups:
                            param_group['lr'] = actor_lr
                    else:
                        actor_lr = self.actor_lr

                    # sample a batch
                    indices = np.random.permutation(len(self.replay_buffer))[:self.batch_size]
                    if self.enable_vis_obs:
                        batch_obs = torch.from_numpy(self.replay_buffer.vis_obs[indices]).to(self.device)
                    else:
                        batch_obs = torch.from_numpy(self.replay_buffer.state_obs[indices]).to(self.device)

                    batch_actions = torch.from_numpy(self.replay_buffer.actions[indices]).to(self.device)
                    
                    # compute loss and optimize
                    loss = self.compute_supervised_loss(batch_obs, batch_actions)
                    self.actor_optimizer.zero_grad()
                    loss.backward()
    
                    self.actor_optimizer.step()
                    supervised_loss = loss.item()

                    time_elapse = time.time() - self.start_time - self.time_report.timers["evaluation"].time_total + self.teacher_training_time
                    # logging
                    self.writer.add_scalar('lr/supervised_iter', self.actor_lr, self.supervised_iter_count)
                    self.writer.add_scalar('supervised_loss/supervised_iter', supervised_loss, self.supervised_iter_count)
                    self.writer.add_scalar('supervised_loss/time', supervised_loss, time_elapse)
                    self.writer.flush()
                    print('supervised iter {}/{}, loss = {:7.6f}'.format(i_update + 1, self.num_update_per_epoch, supervised_loss), end='\r')
                    self.supervised_iter_count += 1

                    # early stop
                    if supervised_loss < self.supervised_learning_threshold:
                        break
                    
                self.time_report.end_timer("supervised learning")

            self.iter_count += 1
            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time - self.time_report.timers["evaluation"].time_total + self.teacher_training_time
            if len(self.episode_length_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_rewards = self.episode_rewards_meter.get_mean()

                if mean_policy_rewards > self.best_policy_rewards:
                    print_info("save best policy with rewards {:.2f}".format(mean_policy_rewards))
                    self.save()
                    self.best_policy_rewards = mean_policy_rewards
                
                self.writer.add_scalar('rewards/step', mean_policy_rewards, self.step_count)
                self.writer.add_scalar('rewards/time', mean_policy_rewards, time_elapse)
                self.writer.add_scalar('rewards/iter', mean_policy_rewards, self.iter_count)
                self.writer.add_scalar('best_policy_rewards/step', self.best_policy_rewards, self.step_count)
                self.writer.add_scalar('best_policy_rewards/iter', self.best_policy_rewards, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_rewards = -np.inf
                mean_episode_length = 0
            
            print('iter {}: ep reward {:.2f}, ep len {:.1f}, supervised loss {:.3f} fps total {:.2f}'.format(\
                    self.iter_count, mean_policy_rewards, mean_episode_length, supervised_loss, self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch)))
            
            self.writer.flush()
            
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                print_info("Start Evaluation with maximum trajectory length:{}".format(self.max_episode_length//5))
                eval_start_time = time.time()
                self.time_report.start_timer("evaluation")
                save_dir = os.path.join(self.log_dir, "eval/iter_{}".format(self.iter_count))
                # evaluate shorter trajectories 
                self.run(self.num_envs, save_dir=save_dir, maximum_eval_length=self.max_episode_length//5)
                self.save(save_dir=save_dir, filename=self.name + "policy_iter{}_reward{:.3f}".format(self.iter_count, mean_policy_rewards))
                self.time_report.end_timer("evaluation")
                self.save_training_summary(save_dir=save_dir)
                print_info("Evaluation done in {} seconds".format(time.time()-eval_start_time))

        self.time_report.end_timer("algorithm")

        self.time_report.report()
        
        self.save_training_summary()

        self.save('final_policy')

        # save reward/length history
        self.episode_rewards_his = np.array(self.episode_rewards_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(open(os.path.join(self.log_dir, 'episode_rewards_his.npy'), 'wb'), self.episode_rewards_his)
        np.save(open(os.path.join(self.log_dir, 'episode_length_his.npy'), 'wb'), self.episode_length_his)

        # evaluate the final policy's performance
        print_info("Start Evaluation for final policy")
        self.run(self.num_envs, save_dir=os.path.join(self.log_dir, "eval/final_policy"))

        self.close()

    def save_training_summary(self, save_dir = None):
        if save_dir is None:
            save_dir = self.log_dir
        
        time_report = {}
        for timer_name in self.time_report.timers.keys():
            time_report.update({timer_name: self.time_report.timers[timer_name].time_total})

        training_summary = {
        "time_report": time_report,
        "env_step": self.step_count 
        }

        with open(os.path.join(save_dir, "training_summary.pkl"), "wb") as f:
            pickle.dump(training_summary, f)

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic=False, maximum_eval_length=None):
        episode_length_his = []
        episode_reward_his = []
        joint_qs = []
        joint_qds = []

        episode_reward = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)

        env = self.env.clone()
        obs = env.reset(enable_vis_obs=self.enable_vis_obs)
        state_obs = obs["state_obs"]
        if self.enable_vis_obs:
            vis_obs = obs["vis_obs"]


        joint_qs.append(env.state.joint_q.view(self.num_envs, -1).detach().clone())
        joint_qds.append(env.state.joint_qd.view(self.num_envs, -1).detach().clone())

        games_cnt = 0
        if maximum_eval_length is None:
            maximum_eval_length = self.max_episode_length

        while games_cnt < num_games:
            if self.enable_vis_obs:
                action = self.actor(vis_obs, deterministic = deterministic)
            else:
                action = self.actor(state_obs, deterministic = deterministic)


            obs, rew, done, _ = env.step(torch.tanh(action), enable_reset=True, enable_vis_obs=self.enable_vis_obs)
            state_obs = obs["state_obs"]
            if self.enable_vis_obs:
                vis_obs = obs["vis_obs"]

            joint_qs.append(env.state.joint_q.view(self.num_envs, -1).detach().clone())
            joint_qds.append(env.state.joint_qd.view(self.num_envs, -1).detach().clone())

            episode_length += 1

            # terminate the environment early during eval
            done = torch.where(episode_length >= maximum_eval_length, torch.ones_like(done), done)
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            episode_reward += rew
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('rewards = {:.2f}, len = {}'.format(episode_reward[done_env_id].item(), episode_length[done_env_id]))
                    episode_reward_his.append(episode_reward[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_reward[done_env_id] = 0.
                    episode_length[done_env_id] = 0
                    games_cnt += 1
        
        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_rewards = np.mean(np.array(episode_reward_his))
 
        return mean_policy_rewards, mean_episode_length, torch.stack(joint_qs), torch.stack(joint_qds)

    @torch.no_grad()
    def run(self, num_games, save_dir=None, maximum_eval_length=None):
        mean_policy_rewards, mean_episode_length, joint_qs, joint_qds = self.evaluate_policy(
            num_games = num_games, deterministic = not self.stochastic_evaluation, maximum_eval_length=maximum_eval_length)
        print_info('mean episode reward = {}, mean episode length = {}'.format(mean_policy_rewards, mean_episode_length))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            self.save_video(joint_qs, save_dir=save_dir)    
            trajs = {"dt":self.env.sim_dt, "joint_q":joint_qs.detach().cpu().numpy(), "joint_qd":joint_qds.detach().cpu().numpy()}
            with open(os.path.join(save_dir, "trajs.pkl"), "wb") as f:
                pickle.dump(trajs, f)

    def save_video(self, joint_qs, save_dir=None, max_video_length=800):
        self.video_recorder.update_save_dir(save_dir)
        frames = self.env.render_traj(joint_qs[:max_video_length], recording=True)
        
        for i in range(frames.shape[1]):
            self.video_recorder.reset()
            for j in range(frames.shape[0]):
                self.video_recorder.append(frames[j, i])
            self.video_recorder.save("eval_traj_{}.mp4".format(i))

    def save(self, filename = None, save_dir = None):
        if save_dir is None:
            save_dir = self.log_dir
        if filename is None:
            filename = 'best_policy'
        torch.save([self.actor], os.path.join(save_dir, "{}.pt".format(filename)))

    def close(self):
        self.video_recorder.stop()
        self.writer.close()
                    