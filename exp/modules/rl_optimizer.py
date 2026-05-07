import csv
import os

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text


class UniversalRLOptimizer:
    def __init__(self, config):
        rl_config = config.get("rl", {})
        self.gamma = rl_config.get("gamma", 0.95)
        self.q_alpha = rl_config.get("q_alpha", 0.6)
        print(colored("⚙️训练监控台", "cyan", attrs=["bold"]))

    @staticmethod
    def _reward_profile(trace):
        info = trace.get("info", {}) if isinstance(trace, dict) else {}
        if not isinstance(info, dict):
            return {}
        reward_profile = info.get("reward_profile", {})
        return reward_profile if isinstance(reward_profile, dict) else {}

    def topology_potential(self, trace):
        reward_profile = self._reward_profile(trace)
        if "topology_potential" in reward_profile:
            return float(reward_profile.get("topology_potential") or 0.0)
        return float(trace.get("pddl_reward", 0.0) or 0.0)

    def value_signal(self, trace, terminal_success):
        reward_profile = self._reward_profile(trace)
        if "value_signal" in reward_profile:
            return float(reward_profile.get("value_signal") or 0.0)
        phi = self.topology_potential(trace)
        return phi if terminal_success else (phi - 1.0)

    def compute_discounted_returns(self, episode_traces, is_success):
        """
        EMMA-style value estimates for code tasks:
        - topology_potential φ encodes how far the current edge progressed in the
          causal chain
        - value_signal encodes signed success/failure outcome for localized update
        """
        if not episode_traces:
            return [], []

        returns = []
        audit_trail = []
        for idx, trace in enumerate(episode_traces):
            terminal = idx == len(episode_traces) - 1
            phi = self.topology_potential(trace)
            observed = self.value_signal(trace, terminal and is_success)
            future = 0.0
            if not terminal and idx + 1 < len(episode_traces):
                future = self.gamma * self.value_signal(episode_traces[idx + 1], False)
            g_t = observed + future
            returns.append(g_t)
            audit_trail.append(
                {
                    "topology_potential": round(phi, 4),
                    "value_signal": round(observed, 4),
                    "future_component": round(future, 4),
                    "G_t": round(g_t, 4),
                }
            )
        return returns, audit_trail

    def get_initial_q(self, is_success, topology_potential=None):
        if topology_potential is None:
            return 1.0 if is_success else -1.0
        phi = float(topology_potential)
        phi = max(0.0, min(1.0, phi))
        return phi if is_success else (phi - 1.0)

    def update_q(self, q_old, observed_value):
        surprise = float(observed_value) - float(q_old)
        q_new = float(q_old) + self.q_alpha * surprise
        q_new = max(-1.0, min(1.0, q_new))
        return q_new, {"surprise": round(surprise, 4), "observed_value": round(float(observed_value), 4)}

    def log_episode(self, episode, success, reward, td, task_type="General Task"):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "training_metrics.csv")
        file_exists = os.path.isfile(log_file)

        try:
            with open(log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Episode", "Task_Type", "Success", "Total_Reward", "Avg_TD_Error"])
                writer.writerow([episode + 1, task_type, 1 if success else 0, round(reward, 4), round(td, 4)])
        except Exception as e:
            print(colored(f"⚠️ 训练日志写入失败: {e}", "red"))
