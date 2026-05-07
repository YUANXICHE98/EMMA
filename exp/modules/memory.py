import json
import os
import numpy as np
try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

class EpisodicMemory:
    def __init__(self, config):
        """
        情景记忆库：负责存储、合并和管理意图空间中的所有经验
        """
        self.memory_file = "memrl_memory_dump.json"
        self.records = self._load_memory()
        mem_config = config.get('memory', {})
        
        # 🔥 核心参数：意图合并阈值。
        # 当新任务的 z 与库中某条记忆的 z 相似度大于此值时，判定为同一任务，执行更新而不是新建！
        self.merge_threshold = mem_config.get('merge_threshold', 0.99)
        self.prune_threshold = mem_config.get('prune_threshold', -3.0)
        self.strategy_bonus = mem_config.get('strategy_bonus', 0.4)

        print(colored(f"💾 记忆库已挂载 | 当前容量: {len(self.records)} 条 | 合并阈值: {self.merge_threshold}", "cyan", attrs=['bold']))

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def save_memory(self):
        save_data = []
        for r in self.records:
            payload = {
                'z': r['z'].tolist() if isinstance(r['z'], np.ndarray) else r['z'],
                'e': r['e'],
                'q': r['q']
            }
            if 's' in r:
                payload['s'] = r['s']
            if 'meta' in r:
                payload['meta'] = r['meta']
            save_data.append(payload)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)

    def set_q_value(self, idx, q_new):
        """提供给 RLOptimizer 专门更新 Q 值的接口"""
        if 0 <= idx < len(self.records):
            self.records[idx]['q'] = q_new

    def apply_localized_feedback(self, idx, q_new, final_success, forgetting_enabled=True):
        if not (0 <= idx < len(self.records)):
            return
        record = self.records[idx]
        record['q'] = q_new
        structured = record.get('s', {}) if isinstance(record, dict) else {}
        meta = record.get('meta', {}) if isinstance(record, dict) else {}
        level = str((structured or {}).get('memory_level', '')).strip()
        if final_success:
            if level == "strategy":
                structured['trusted'] = True
                record['s'] = structured
            return

        if level == "tactical":
            if structured.get('outcome') == 'success':
                structured['value_bias'] = 'cautionary_recheck'
                structured['trusted'] = False
            if forgetting_enabled and q_new <= self.prune_threshold:
                meta['pruned'] = True
                structured['value_bias'] = 'suppressed_after_negative_transfer'
            record['meta'] = meta
            record['s'] = structured
        elif level == "strategy":
            structured['value_bias'] = 'cautionary_recheck'
            structured['trusted'] = False
            record['s'] = structured

    def _cosine_similarity(self, v1, v2):
        if v1 is None or v2 is None: return -1.0
        v1, v2 = np.array(v1), np.array(v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.dot(v1, v2) / norm if norm != 0 else 0.0

    @staticmethod
    def _record_key(structured):
        if not isinstance(structured, dict):
            return ""
        level = str(structured.get('memory_level', '')).strip()
        if level == "strategy":
            return f"strategy::{structured.get('strategy_key', '')}"
        return f"tactical::{structured.get('edge_key', '')}"

    def _merge_or_append_record(self, z_np, e, initial_q, meta, structured):
        target_key = self._record_key(structured)
        success_flag = bool((meta or {}).get('success', False))
        best_sim = -1.0
        best_idx = -1

        for i, record in enumerate(self.records):
            record_structured = record.get('s', {}) if isinstance(record, dict) else {}
            if target_key and self._record_key(record_structured) != target_key:
                continue
            sim = self.cosine_similarity(z_np, record['z'])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= self.merge_threshold:
            print(colored(f"🔄 [记忆进化] 命中同一结构边 (相似度 {best_sim:.2f} >= {self.merge_threshold})", "magenta"))
            old_z = np.array(self.records[best_idx]['z'])
            updated_z = 0.7 * old_z + 0.3 * z_np
            self.records[best_idx]['z'] = updated_z.tolist()
            self.records[best_idx]['e'] = e
            if structured is not None:
                existing = self.records[best_idx].get('s', {}) or {}
                if structured.get('memory_level') == 'strategy':
                    parents = set(existing.get('parent_edge_keys', []) or [])
                    parents.update(structured.get('parent_edge_keys', []) or [])
                    structured['parent_edge_keys'] = sorted(parents)
                    structured['consolidation_count'] = len(structured['parent_edge_keys'])
                self.records[best_idx]['s'] = structured
            if meta is not None:
                merged_meta = self.records[best_idx].get('meta', {}) or {}
                merged_meta.update(meta)
                merged_meta['success_count'] = int(merged_meta.get('success_count', 0)) + (1 if success_flag else 0)
                merged_meta['failure_count'] = int(merged_meta.get('failure_count', 0)) + (0 if success_flag else 1)
                self.records[best_idx]['meta'] = merged_meta
            return best_idx

        print(colored("✨ [开拓新知] 新建结构边节点...", "green"))
        new_meta = dict(meta or {})
        new_meta['success_count'] = 1 if success_flag else 0
        new_meta['failure_count'] = 0 if success_flag else 1
        self.records.append({
            'z': z_np.tolist(),
            'e': e,
            'q': initial_q,
            's': structured or {},
            'meta': new_meta
        })
        return len(self.records) - 1

    def consolidate_strategy(self, z, structured, meta=None):
        if not isinstance(structured, dict):
            return None
        strategy_key = str(structured.get('strategy_key', '')).strip()
        if not strategy_key:
            return None
        z_np = np.array(z)
        rendered = structured.get('rendered_hint', '') or ""
        initial_q = float(structured.get('q_init', 0.0))
        return self._merge_or_append_record(z_np, rendered, initial_q, meta or {}, structured)

    def add_memory(self, z, e, initial_q, meta=None, structured=None):
        """
        🔥 核心升级：带有去重和合并机制的记忆写入逻辑 🔥
        """
        z_np = np.array(z)
        return self._merge_or_append_record(z_np, e, initial_q, meta or {}, structured or {})
            
    # 为了兼容外部可能直接调用 cosine_similarity
    def cosine_similarity(self, v1, v2):
        return self._cosine_similarity(v1, v2)
