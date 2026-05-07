import os
import re
import yaml
from termcolor import colored

from alfworld.agents.environment import get_environment


def _resolve_alfworld_path(path):
    if not path:
        return path

    root = os.path.expanduser(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld"))
    path = os.path.expandvars(os.path.expanduser(path))
    path = re.sub(r"\$\{ALFWORLD_DATA:-~/.cache/alfworld\}", root, path)

    return path


def _normalize_alfworld_config(config):
    env_cfg = config.get("env", {})
    dataset_cfg = config.get("dataset", {})

    for key in ("domain_file", "grammar_path"):
        if key in env_cfg:
            env_cfg[key] = _resolve_alfworld_path(env_cfg[key])

    hypertree_cfg = env_cfg.get("hypertree_cfg", {})
    if "data_path" in hypertree_cfg:
        hypertree_cfg["data_path"] = _resolve_alfworld_path(hypertree_cfg["data_path"])

    for key in ("data_path", "eval_id_data_path", "eval_ood_data_path"):
        if key in dataset_cfg:
            dataset_cfg[key] = _resolve_alfworld_path(dataset_cfg[key])

    return config

class ALFWorldEnvWrapper:
    def __init__(self, difficulty="hard"):
        """
        ALFWorld 环境封装器
        :param difficulty: "easy" (简单任务) 或 "hard" (复杂任务筛选)
        """
        self.difficulty = difficulty
        print(colored(f"🌍 正在初始化 ALFWorld 环境 | 当前难度: {self.difficulty.upper()}", "cyan", attrs=['bold']))
        
        config_path = os.path.join("configs", "base_config.yaml")
        
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Cannot find ALFWorld config file: {config_path}")
        
        with open(config_path) as reader:
            self.config = yaml.safe_load(reader)
        self.config = _normalize_alfworld_config(self.config)

        env_type = self.config['env']['type']
        self.env = get_environment(env_type)(self.config, train_eval="train")
        self.env = self.env.init_env(batch_size=1)
        
        self.current_obs = None
        self.current_info = None

    def reset(self):
        """重置环境并获取一个新任务"""
        while True:
            obs, info = self.env.reset()
            task_desc = obs[0]
            
            # 提取任务类型特征
            is_complex = any(keyword in task_desc for keyword in ["clean", "heat", "cool", "two"])
            
            if self.difficulty == "hard" and not is_complex:
                continue # 过滤掉简单的 Pick & Place 任务，继续刷
            else:
                self.current_obs = obs[0]
                self.current_info = info
                break
                
        valid_actions = self.current_info.get('admissible_commands', [[]])[0]
        return self.current_obs, task_desc, valid_actions

    def step(self, action):
        """执行动作并返回结果"""
        obs, scores, dones, infos = self.env.step([action])
        
        self.current_obs = obs[0]
        self.current_info = infos
        
        reward = scores[0]
        done = dones[0]
        
        is_success = infos.get('won', [False])[0] if 'won' in infos else False
        step_reward = 1.0 if is_success else -0.1 
        
        valid_actions = infos.get('admissible_commands', [[]])[0]
        
        trace = {
            "action": action,
            "obs": self.current_obs,
            "pddl_reward": step_reward,
            "is_success": is_success
        }
        
        return self.current_obs, step_reward, done, trace, valid_actions
