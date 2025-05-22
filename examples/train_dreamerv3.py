import argparse
import functools
import os
import pathlib
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["MUJOCO_GL"] = 'egl' # "osmesa"

import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import externals.dreamerv3.exploration as expl
import externals.dreamerv3.models as models
from externals.dreamerv3 import tools
import externals.dreamerv3.envs.wrappers as wrappers
from externals.dreamerv3.parallel import Parallel, Damy

import dm_env
from dm_env import specs
import envs
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
from utils.common import *
import pickle
import time

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()

class MakeDMfromdFlex(dm_env.Environment):
    def __init__(self, cfg, task):
        self.cfg = cfg
        
        env_fn = getattr(envs, cfg.env_name[task])
        seeding(cfg.seed)
        self.env =  env_fn(num_envs = 1, \
                            device = cfg.device, \
                            img_height = cfg.img_height,\
                            img_width = cfg.img_width,\
                            seed = cfg.seed, \
                            episode_length=cfg.episode_length, \
                            stochastic_init = cfg.stochastic_env, \
                            MM_caching_frequency = cfg.MM_caching_frequency, \
                            no_grad = True)

        self.num_envs = self.env.num_envs
        self.num_actions = self.env.num_actions
        self.device = cfg.device
        self.raw_rew = np.zeros((self.env.num_envs)) 
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
                                                   minimum=-1 * np.ones((self.env.num_actions, )),
                                                   maximum=1 * np.ones((self.env.num_actions, )),
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
        obs_stack = np.squeeze((obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")).transpose(1, 2, 0)
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, 
                            reward=None,
                            discount=1.0,
                            observation=obs_stack[:,:,-3:])
         
    
    def step(self, actions, enable_reset = False, enable_vis_obs = True):
        obs, rew_batch, done_batch, extra_info = self.env.step(actions = torch.tanh(torch.tensor(actions, dtype = torch.float32, device = self.device)), 
                                                               enable_reset = enable_reset, 
                                                               enable_vis_obs = enable_vis_obs)
        del extra_info
        self.raw_rew[:] = rew_batch.detach().clone().cpu().numpy()
        obs_stack = np.squeeze((obs["vis_obs"]).detach().clone().cpu().numpy().astype("uint8")).transpose(1, 2, 0)
        return dm_env.TimeStep(step_type=dm_env.StepType.MID if not done_batch.detach().clone().cpu().item() else dm_env.StepType.LAST,
                            reward=rew_batch.detach().clone().cpu().item(),
                            discount=1.0,
                            observation=obs_stack[:,:,-3:])
    
    def observation_spec(self):
        return self._observation_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def action_spec(self):
        return self._action_spec
    
    def discount_spec(self):
        return self._discount_spec
    
class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True, time_report=None):
        step = self._step
        if training:
            time_report.start_timer("IO time")
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            time_report.end_timer("IO time")
            
            for _ in range(steps):
                time_report.start_timer("IO time")
                data = next(self._dataset)
                time_report.end_timer("IO time")
                self._train(data, time_report)
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            time_report.start_timer("Logger time")
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)
            time_report.end_timer("Logger time")
        if training:
            time_report.start_timer("forward simulation")
        policy_output, state = self._policy(obs, state, training)
        if training:
            time_report.end_timer("forward simulation")
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data, time_report):
        metrics = {}  
        post, context, mets = self._wm._train(data, time_report)
        time_report.start_timer("forward simulation")
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        time_report.end_timer("forward simulation")   
        
        metrics.update(self._task_behavior._train(start, reward, time_report)[-1])

        time_report.start_timer("forward simulation")
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
        time_report.end_timer("forward simulation")

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import externals.dreamerv3.envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
        env.reset()

    elif suite == "dflex":
        import externals.dreamerv3.envs.dmc as dmc
        env = MakeDMfromdFlex(config, task)
        env = dmc.DeepMindControlDflex(
            env, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
        env.reset()

    elif suite == "atari":
        import externals.dreamerv3.envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import externals.dreamerv3.envs.dmlab as dmlab
        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from externals.dreamerv3.envs.memorymaze import MemoryMaze
        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)

    elif suite == "crafter":
        import externals.dreamerv3.envs.crafter as crafter
        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)

    elif suite == "minecraft":
        import externals.dreamerv3.envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env

def main(config):
    time_report = TimeReport()
    time_report.add_timer("algorithm")
    time_report.add_timer("forward simulation")
    time_report.add_timer("dflex step")
    time_report.add_timer("backward simulation")
    time_report.add_timer("actor training")
    time_report.add_timer("critic training")
    time_report.add_timer("world model training")
    time_report.add_timer("evaluation time")
    time_report.add_timer("prefill dataset")  
    time_report.add_timer("IO time")
    time_report.add_timer("Logger time")
    time_report.add_timer("NN Training")

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    time_report_dir = logdir / "time_reports"

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)
    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    time_report.start_timer("algorithm")
    time_report.start_timer("prefill dataset")
    algo_start_time = time.time()

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(obs, reset, state, time_report):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
            is_eval=True,
            time_report=None,
            algo_start_time=algo_start_time,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    
    time_report.end_timer("prefill dataset")
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        time_report.start_timer("Logger time")
        logger.write()
        time_report.end_timer("Logger time")
        time_report.start_timer("evaluation time")
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
                time_report=None,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        time_report.end_timer("evaluation time")
        print("Start training.")

        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            time_report=time_report,
            algo_start_time=algo_start_time,
        )
        time_report.start_timer("IO time")
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        time_report.end_timer("IO time")
        save_training_summary(time_report, agent._step, time_report_dir/f"steps_{agent._step}")
    time_report.end_timer("algorithm")
    time_report.report()
    save_training_summary(time_report, agent._step, time_report_dir/f"steps_{agent._step}")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

def save_training_summary(current_time_report, step_count, save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)
        time_report = {}
        for timer_name in current_time_report.timers.keys():
            time_report.update({timer_name: current_time_report.timers[timer_name].time_total})

        training_summary = {
        "time_report": time_report,
        "env_step": step_count 
        }

        with open(os.path.join(str(save_dir), "training_summary.pkl"), "wb") as f:
            pickle.dump(training_summary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "cfg/dreamerv3/configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
