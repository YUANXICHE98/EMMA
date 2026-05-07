import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

st.set_page_config(page_title="MemRL 训练控制台", layout="wide")
st.title("🚀 MemRL 强化学习训练控制台")

# ==========================================
# 1. 数据加载引擎 (已升级对接 CSV 日志)
# ==========================================
MEMORY_FILE = "memrl_memory_dump.json"  # 如果你的记忆库叫别名，请在这里改
RL_LOG_FILE = os.path.join("logs", "training_metrics.csv") # 🔥 变更为读取全新的 CSV 日志

@st.cache_data(ttl=5) # 每 5 秒自动刷新缓存
def load_data():
    df_mem = pd.DataFrame()
    df_log = pd.DataFrame()
    
    # 读取记忆库 JSON
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                mem_data = json.load(f)
                # 兼容不同格式：如果数据包在 'records' 键里则提取，否则直接转 DataFrame
                if isinstance(mem_data, dict) and 'records' in mem_data:
                    df_mem = pd.DataFrame(mem_data['records'])
                else:
                    df_mem = pd.DataFrame(mem_data)
        except Exception as e:
            pass 
            
    # 🔥 读取全新的 RL 训练日志 CSV
    if os.path.exists(RL_LOG_FILE):
        try:
            df_log = pd.read_csv(RL_LOG_FILE)
            
            # 字段名映射适配器：把 CSV 的首字母大写列名，映射回你画图用的老列名
            df_log.rename(columns={
                'Episode': 'episode',
                'Task_Type': 'task_type',
                'Success': 'is_success',
                'Total_Reward': 'pddl_reward',
                'Avg_TD_Error': 'td_error_abs'
            }, inplace=True)
            
        except Exception as e:
            pass
            
    return df_mem, df_log

df_mem, df_log = load_data()

if df_log.empty and df_mem.empty:
    st.warning("⚠️ 暂无训练数据，请先运行 run_episode.py 开始训练！")
    st.stop()

# ==========================================
# 2. 仪表盘布局 (Tabs)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 训练曲线 (Metrics)", 
    "🕸️ 技能雷达 (Capabilities)", 
    "🧠 记忆空间 (Memory Space)",
    "🗂️ 记忆列表 (Raw Memories)"
])

# ------------------------------------------
# Tab 1: 训练曲线 (类似 TensorBoard)
# ------------------------------------------
with tab1:
    st.markdown("### 核心强化学习指标")
    if not df_log.empty:
        window = min(10, len(df_log)) if len(df_log) > 0 else 1
        df_log['success_ema'] = df_log['is_success'].ewm(span=window).mean()
        df_log['reward_ema'] = df_log['pddl_reward'].ewm(span=window).mean()
        df_log['td_error_ema'] = df_log['td_error_abs'].ewm(span=window).mean()

        col1, col2, col3 = st.columns(3)
        
        fig_td = px.line(df_log, x='episode', y=['td_error_abs', 'td_error_ema'], 
                         title="📉 TD 误差 (当前已停用, 预期为 0)",
                         color_discrete_sequence=['rgba(255,0,0,0.2)', 'red'])
        fig_td.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        col1.plotly_chart(fig_td, use_container_width=True)

        fig_reward = px.line(df_log, x='episode', y=['pddl_reward', 'reward_ema'], 
                             title="💰 回合总势能奖励 (Episode Reward)",
                             color_discrete_sequence=['rgba(0,0,255,0.2)', 'blue'])
        fig_reward.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        col2.plotly_chart(fig_reward, use_container_width=True)

        fig_sr = px.line(df_log, x='episode', y=['is_success', 'success_ema'], 
                         title="🎯 任务成功率 (Success Rate)",
                         color_discrete_sequence=['rgba(0,128,0,0.2)', 'green'])
        fig_sr.update_layout(showlegend=False, yaxis=dict(range=[0, 1.1]), margin=dict(l=0, r=0, t=30, b=0))
        col3.plotly_chart(fig_sr, use_container_width=True)
    else:
        st.info("尚无 RL 日志数据。")

# ------------------------------------------
# Tab 2: 技能雷达 (能力倾向分析)
# ------------------------------------------
with tab2:
    st.markdown("### Agent 技能演化状态 (基于真实胜率)")
    if not df_log.empty and 'task_type' in df_log.columns:
        skill_stats = df_log.groupby('task_type')['is_success'].agg(['mean', 'count']).reset_index()
        skill_stats.rename(columns={'mean': 'win_rate', 'count': 'attempts'}, inplace=True)
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=skill_stats['win_rate'],
            theta=skill_stats['task_type'],
            fill='toself',
            name='胜率 (Win Rate)',
            marker=dict(color='magenta')
        ))
        
        fig_radar.update_layout(
            title="🎯 各类任务通关胜率雷达图 (0~100%)", 
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
            showlegend=False
        )
        
        col_radar, col_table = st.columns([1.5, 1])
        with col_radar:
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_table:
            st.markdown("#### 📊 详细任务统计")
            st.dataframe(
                skill_stats.style.format({'win_rate': '{:.1%}'})
                           .background_gradient(cmap='RdYlGn', subset=['win_rate']),
                hide_index=True
            )
            
        st.divider()
        if not df_mem.empty and 'q' in df_mem.columns:
            fig_hist = px.histogram(df_mem, x="q", nbins=20, title="🧠 记忆库 Q 值健康度分布", 
                                    color_discrete_sequence=['teal'])
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("尚无带有任务分类的 RL 日志数据，请先跑几局新版程序收集数据。")

# ------------------------------------------
# Tab 3: 记忆空间 (PCA 降维图)
# ------------------------------------------
with tab3:
    st.markdown("### 🗺️ 意图语义空间聚类")
    if not df_mem.empty and 'z' in df_mem.columns and len(df_mem) >= 2:
        try:
            z_matrix = np.array(df_mem['z'].tolist())
            pca = PCA(n_components=2)
            coords = pca.fit_transform(z_matrix)
            
            plot_df = pd.DataFrame(coords, columns=['PCA_1', 'PCA_2'])
            plot_df['Experience'] = df_mem['e']
            plot_df['Q_Value'] = df_mem['q']
            
            fig_pca = px.scatter(
                plot_df, x='PCA_1', y='PCA_2', color='Q_Value',
                hover_data=['Experience'],
                color_continuous_scale='RdYlGn',
                title="高维意图 2D 投影 (越近越相似)"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.error(f"降维绘图失败，可能是因为有些早期记忆缺少意图向量: {e}")
    else:
        st.warning("记忆数量不足 2 条，无法生成意图空间地图。")

# ------------------------------------------
# Tab 4: 记忆列表 (原始文本)
# ------------------------------------------
with tab4:
    st.markdown("### 🗂️ 记忆库明细")
    if not df_mem.empty and 'e' in df_mem.columns:
        display_df = df_mem[['q', 'e']].copy()
        display_df.rename(columns={'q': 'Q值 (权重)', 'e': '经验/教训文本'}, inplace=True)
        
        display_df = display_df.sort_values(by='Q值 (权重)', ascending=False)
        
        st.write(f"当前记忆总数：**{len(display_df)}** 条")
        
        # 用渐变色来直观显示 Q 值的高低
        st.dataframe(
            display_df.style.background_gradient(cmap='RdYlGn', subset=['Q值 (权重)']),
            use_container_width=True,
            height=600
        )
    else:
        st.info("当前记忆库为空，或者数据格式尚未包含经验文本。")