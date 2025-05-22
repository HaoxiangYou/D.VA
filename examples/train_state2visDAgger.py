import os, sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import torch.nn as nn
import yaml
import pickle
import argparse
import algorithms.shac as shac
import algorithms.dagger as dagger
import shutil
import copy
from utils.common import *

class TeacherPolicy(nn.Module):
    def __init__(self, base_policy, obs_rms=None, device="cuda"):
        super().__init__()
        self.base_policy = copy.deepcopy(base_policy).to(device)
        self.obs_rms = obs_rms
        self.device = device
    
    def forward(self, obs, deterministic=True):
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        actions = self.base_policy(obs, deterministic = deterministic)
        return actions
    
def parse_arguments(description="Testing Args", custom_parameters=[]):
    parser = argparse.ArgumentParser()

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    print("ERROR: default must be specified if using type")
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)
        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    
    args = parser.parse_args()
    
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
    
def get_args(): # TODO: delve into the arguments
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--cfg", "type": str, "default": "./cfg/teacher_student/hopper.yaml",
            "help": "Configuration file for training/playing"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights"},
        {"name": "--logdir", "type": str, "default": "logs/tmp/teacher_student/"},
        {"name": "--teacher_checkpoint_dir", "type": str, "default": None},
        {"name": "--save-interval", "type": int, "default": 0},
        {"name": "--no-time-stamp", "action": "store_true", "default": False,
            "help": "whether not add time stamp at the log path"},
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"}]
    
    # parse arguments
    args = parse_arguments(
        description="Teacher-Student Distillation",
        custom_parameters=custom_parameters)
    
    return args

if __name__ == '__main__':
    args = get_args()

    with open(args.cfg, 'r') as f:
        cfg_student = yaml.load(f, Loader=yaml.SafeLoader)

    if args.play or args.test:
        cfg_student["params"]["config"]["num_actors"] = cfg_student["params"]["config"].get("player", {}).get("num_actors", 1)

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())
    args.logdir = os.path.join(args.logdir , "seed_" + str(args.seed))
    
    args.device = torch.device(args.device)

    vargs = vars(args)

    cfg_student["params"]["general"] = {}
    for key in vargs.keys():
        cfg_student["params"]["general"][key] = vargs[key]

    # copy the visual observation related setting from config to newtwork
    cfg_student["params"]["network"]["vis_obs"] = cfg_student["params"]["config"].get("vis_obs", True)
    cfg_student["params"]["network"]["img_height"] = cfg_student["params"]["config"].get("img_height", 84)
    cfg_student["params"]["network"]["img_width"] = cfg_student["params"]["config"].get("img_width", 84)

    if args.train:

        if args.teacher_checkpoint_dir is None:
            # config teacher policy optimizer
            cfg_teacher_path = os.path.join(os.path.dirname(args.cfg), cfg_student["params"]["config"]["teacher_policy_config_path"])
            with open(cfg_teacher_path, 'r') as f:
                cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_teacher["params"]["general"] = {}
            cfg_teacher["params"]["general"]["logdir"] = os.path.join(cfg_student["params"]["general"]["logdir"], "teacher")
            cfg_teacher["params"]["general"]["seed"] = cfg_student["params"]["general"]["seed"]
            cfg_teacher["params"]["general"]["device"] = cfg_student["params"]["general"]["device"]
            cfg_teacher["params"]["general"]["train"] = cfg_student["params"]["general"]["train"]
            # train teacher policy
            teacher_optimizer = shac.SHAC(cfg_teacher)
            teacher_optimizer.train()

        else:
            with open(os.path.join(args.teacher_checkpoint_dir, "cfg.yaml"), 'r') as f:
                cfg_teacher = yaml.load(f, Loader=yaml.UnsafeLoader)
            cfg_teacher["params"]["general"]["logdir"] = os.path.join(cfg_student["params"]["general"]["logdir"], "teacher")
            cfg_teacher["params"]["general"]["seed"] = cfg_student["params"]["general"]["seed"]
            cfg_teacher["params"]["general"]["device"] = cfg_student["params"]["general"]["device"]
            cfg_teacher["params"]["general"]["train"] = False
            # load the teacher policy from existing one
            teacher_optimizer = shac.SHAC(cfg_teacher)
            teacher_optimizer.load(os.path.join(args.teacher_checkpoint_dir, "final_policy.pt"))
            with open(os.path.join(args.teacher_checkpoint_dir, "training_summary.pkl"), "rb") as f:
                training_summary_dict = pickle.load(f)    
                for key, value in training_summary_dict["time_report"].items():
                    teacher_optimizer.time_report.add_timer(key)
                    teacher_optimizer.time_report.timers[key].time_total = value
                teacher_optimizer.step_count = training_summary_dict["env_step"]

            # copy all the files in the teacher directory to current
            dest_dir = os.path.join(args.logdir, "teacher")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copytree(args.teacher_checkpoint_dir, dest_dir, dirs_exist_ok=True)

        # make teacher policy and training info
        teacher_policy = TeacherPolicy(base_policy=teacher_optimizer.actor, obs_rms=teacher_optimizer.state_obs_rms, device=args.device)
        teacher_training_time = teacher_optimizer.time_report.timers["algorithm"].time_total
        if "evaluation" in teacher_optimizer.time_report.timers:
            teacher_training_time -= teacher_optimizer.time_report.timers["evaluation"].time_total

        # make training logging
        teacher_training_info = {"step_count": teacher_optimizer.step_count,
                        "training_time": teacher_training_time}
        
        # release the memory
        del teacher_optimizer
        torch.cuda.empty_cache()

        student_optimizer = dagger.Dagger(cfg=cfg_student, teacher_policy=teacher_policy, teacher_training_info=teacher_training_info)
        student_optimizer.train()

    else:
        pass


