import os
import yaml
from termcolor import colored

from modules.memory import EpisodicMemory
from modules.encoder import IntentEncoder
from modules.retriever import MemoryRetriever
from modules.llm_core import FrozenLLM
from modules.env_wrapper import ALFWorldEnvWrapper
from modules.rl_optimizer import UniversalRLOptimizer

print(colored("🚀 正在初始化 MemRL 连续学习框架 (终极解耦与高水位线更新版)...", "magenta", attrs=['bold']))

config_path = os.path.join('configs', 'memrl_config.yaml')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ 找不到配置文件 {config_path}，请先创建！")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

memory_bank = EpisodicMemory(config)
encoder = IntentEncoder(config)
retriever = MemoryRetriever(config)
llm = FrozenLLM(config)
env = ALFWorldEnvWrapper()
rl_opt = UniversalRLOptimizer(config)

max_steps_per_episode = 30
total_episodes = 1000

for episode in range(total_episodes):
    print(colored(f"\n========== 开始任务 Episode {episode + 1}/{total_episodes} ==========", "magenta", attrs=['bold']))
    
    s_t, task, valid_actions = env.reset()
    print(colored(f"🎯 任务目标: {task}", "cyan"))
    
    # 简单的任务分类
    task_lower = task.lower()
    if "clean" in task_lower or "wash" in task_lower:
        current_task_type = "Clean & Place"
    elif "hot" in task_lower or "heat" in task_lower or "microwave" in task_lower:
        current_task_type = "Heat & Place"
    elif "cool" in task_lower or "fridge" in task_lower:
        current_task_type = "Cool & Place"
    elif "two" in task_lower:
        current_task_type = "Pick Two"
    elif "look at" in task_lower or "examine" in task_lower:
        current_task_type = "Examine"
    else:
        current_task_type = "Pick & Place"
        
    print(colored(f"🏷️ 任务分类: {current_task_type}", "yellow"))
    
    z_t = encoder.encode(task) 
    
    done = False
    episode_traces = []
    episode_rewards = []
    used_memories_timeline = [] 
    
    history_log = [] 
    
    env_physics_rules = (
        "💡 [Physics Rules]:\n"
        "- To 'clean' an object, you must go to a sinkbasin and execute 'clean [obj] with sinkbasin [id]'.\n"
        "- To 'heat', use a microwave and the 'heat' action.\n"
        "- To 'cool', use a fridge and the 'cool' action."
    )
    
    while not done and len(episode_traces) < max_steps_per_episode:
        print("-" * 40)
        m_ctx, used_memory_idx = retriever.retrieve(z_t, memory_bank)
        
        if used_memory_idx is not None:
            current_step_idx = len(episode_traces)
            used_memories_timeline.append({"step": current_step_idx, "idx": used_memory_idx})
            
        recent_history = "\n".join(history_log[-5:]) if history_log else "None (Game just started)"
        
        dynamic_context = (
            f"🎯 [Goal Task]: {task}\n"
            f"{env_physics_rules}\n"
            f"📍 [CURRENT Observation]: {s_t}\n"
            f"📜 [Recent Action History (Last 5 steps)]:\n{recent_history}"
        )
        
        prompt = retriever.assemble_prompt(dynamic_context, m_ctx)
        
        y_t = llm.generate_action(prompt, valid_actions=valid_actions)
        print(colored(f"🤖 Agent 动作: {y_t}", "yellow"))
        
        next_s_t, step_reward, done, trace, next_valid_actions = env.step(y_t)
        
        if 'action' not in trace: trace['action'] = y_t
        if 'obs' not in trace: trace['obs'] = next_s_t
        if 'pddl_reward' not in trace: trace['pddl_reward'] = step_reward

        print(colored(f"👁️ 环境反馈: {next_s_t}", "cyan"))
        
        episode_traces.append(trace)
        episode_rewards.append(step_reward)
        
        history_log.append(f"> Action: {y_t}\n< Feedback: {next_s_t}")
        s_t = next_s_t
        valid_actions = next_valid_actions

    # =========================================================
    # 🔥 强化学习轨迹结算面板 🔥
    # =========================================================
    final_success = episode_traces[-1].get('is_success', False) if episode_traces else False
    
    if episode_traces:
        # 获取势能计算列表和详细账单
        discounted_returns, audit_trail = rl_opt.compute_discounted_returns(episode_traces, final_success)
        
        # 🌟 终极解锁 1：无论是否使用记忆，每局结束强制打印“全局核算账单”！🌟
        print(colored("\n" + "💰"*3 + " [全局节点势能核算账单] " + "💰"*3, "cyan", attrs=['bold']))
        for t in range(len(episode_traces)):
            audit = audit_trail[t]
            print(colored(f"   [步 {t:02d} 物理审计] | {audit['details']}", "yellow"))
            pot_str = f"R_t({audit['R_t']:+.1f}) + γ*min(未来短板:{audit['bottleneck']:+.1f})"
            print(colored(f"            └─ 势能 G_t = {pot_str} = {discounted_returns[t]:+.2f}", "magenta"))
        print(colored("-" * 65, "cyan"))
    
   
        if used_memories_timeline:
            print(colored("\n" + "🏆"*3 + " [历史记忆 Q 值覆盖更新 (最高水位线)] " + "🏆"*3, "green", attrs=['bold']))
            for usage in used_memories_timeline:
                step_idx = usage["step"]
                mem_idx = usage["idx"]
                
                g_t = discounted_returns[step_idx]
            
                if final_success and g_t < 0: g_t = 0.5 
                
                q_old = memory_bank.records[mem_idx]['q']
                
              
                q_new = max(q_old, g_t)
                memory_bank.set_q_value(mem_idx, q_new)
        
                if q_new > q_old:
                    print(colored(f"   [ 突破纪录] 记忆 {mem_idx} | 发现更优执行路径！Q值: {q_old:.2f} -> {q_new:.2f} ", "green", attrs=['bold']))
                elif q_new == q_old and g_t == q_old:
                    print(colored(f"   [ 稳定发挥] 记忆 {mem_idx} | 势能追平历史最高纪录！维持 Q值: {q_old:.2f}", "blue"))
                else:
                    print(colored(f"   [ 记忆保护] 记忆 {mem_idx} | 本局势能 G_t({g_t:.2f}) 未打破历史纪录 Q({q_old:.2f})，免于降级。", "dark_grey"))
            print(colored("-" * 65, "green"))
    
    # 提取记忆并存入记忆库
    if final_success:
        print(colored("\n 完美通关！提炼经验...", "green", attrs=['bold']))
        e_new = llm.summarize_experience(task, episode_traces, final_success)
        init_q = rl_opt.get_initial_q(is_success=True)
        idx = memory_bank.add_memory(z_t, e_new, initial_q=init_q)
        print(colored(f" 存入记忆 [索引 {idx}] (出厂 Q={init_q}): {e_new}", "green"))
        
    else:
        print(colored("\n 任务失败，提炼避坑教训...", "red", attrs=['bold']))
        e_new = llm.summarize_experience(task, episode_traces, final_success)
        init_q = rl_opt.get_initial_q(is_success=False)
        idx = memory_bank.add_memory(z_t, e_new, initial_q=init_q)
        print(colored(f" 存入教训 [索引 {idx}] (出厂 Q={init_q}): {e_new}", "yellow"))

    # 写入训练埋点 CSV 数据
    total_pddl_reward = sum(episode_rewards[:-1]) + (10 if final_success else -5)
    rl_opt.log_episode(episode, final_success, total_pddl_reward, 0.0, task_type=current_task_type)

    memory_bank.save_memory()
    print(colored(f"💾 [自动存档] 记忆库与训练日志已同步至硬盘。", "blue"))

print(colored("\n" + "="*50, "cyan", attrs=['bold']))
print(colored("✅ 训练彻底结束！", "green", attrs=['bold']))