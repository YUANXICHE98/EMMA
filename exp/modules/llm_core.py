import os
import time
import json
import re
import ast
try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text
import openai
import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class FrozenLLM:
    @staticmethod
    def _parse_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def __init__(self, config):
        llm_config = config.get('llm', {})
        self.model = llm_config.get('model_name')
        self.temperature = llm_config.get('temperature', 0.1)
        self.action_max_tokens = int(llm_config.get('action_max_tokens', 30))
        self.action_timeout = float(llm_config.get('action_timeout', 45))
        self.summary_timeout = float(llm_config.get('summary_timeout', 60))
        self.responses_reasoning_effort = str(llm_config.get("responses_reasoning_effort", "low") or "low").strip()
        self.responses_text_verbosity = str(llm_config.get("responses_text_verbosity", "low") or "low").strip()
        self.client_max_retries = int(
            os.environ.get("EMMA_OPENAI_MAX_RETRIES")
            or os.environ.get("MEMRL_OPENAI_MAX_RETRIES")
            or os.environ.get("OPENAI_MAX_RETRIES")
            or llm_config.get("max_retries", 0)
        )
        self.protocol = str(llm_config.get('protocol') or self._default_protocol_for_model(self.model)).strip().lower()
        self.generate_action_calls = 0
        self.summarize_calls = 0
        self.last_action_debug = {
            "system_prompt": "",
            "user_prompt": "",
            "raw_response": "",
            "error": "",
        }
        self.last_route_debug = {
            "system_prompt": "",
            "user_prompt": "",
            "raw_response": "",
            "error": "",
        }

        self.api_key = (
            os.environ.get("EMMA_OPENAI_API_KEY")
            or os.environ.get("MEMRL_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or llm_config.get('api_key')
        )
        self.base_url = (
            os.environ.get("EMMA_OPENAI_BASE_URL")
            or os.environ.get("MEMRL_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or llm_config.get('base_url')
        )
        self.trust_env_proxy = self._parse_bool(
            os.environ.get("EMMA_OPENAI_TRUST_ENV_PROXY")
            or os.environ.get("MEMRL_OPENAI_TRUST_ENV_PROXY")
            or os.environ.get("OPENAI_TRUST_ENV_PROXY")
            or llm_config.get("trust_env_proxy"),
            default=False,
        )

        if not self.api_key:
            print(colored("⚠️ 警告: 未能在 config 或环境变量中找到 API Key！", "red", attrs=['bold']))

        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url
        client_args["http_client"] = httpx.Client(trust_env=self.trust_env_proxy)
        client_args["max_retries"] = self.client_max_retries
            
        self.client = OpenAI(**client_args)
        
        openai.api_key = self.api_key
        if self.base_url:
            openai.api_base = self.base_url

        print(colored(f"🧠 LLM 核心已挂载 | 模型: {self.model} ", "cyan", attrs=['bold']))

    @staticmethod
    def _default_protocol_for_model(model):
        name = str(model or "").strip().lower()
        if name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
            return "responses"
        return "chat"

    @staticmethod
    def _extract_responses_text(response):
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()
        try:
            data = response.model_dump()
            chunks = []
            for item in data.get("output", []) or []:
                for content in item.get("content", []) or []:
                    text = content.get("text")
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks).strip()
        except Exception:
            return str(response).strip()

    @staticmethod
    def _responses_incomplete_reason(response):
        try:
            data = response.model_dump()
            details = data.get("incomplete_details") or {}
            return str(details.get("reason") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _chat_message_text(choice):
        message = getattr(choice, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        chunks.append(str(text))
                elif item:
                    chunks.append(str(item))
            return "\n".join(chunks).strip()
        if content:
            return str(content).strip()
        return ""

    @staticmethod
    def _chat_reasoning_text(choice):
        message = getattr(choice, "message", None)
        if message is None:
            return ""
        reasoning_fields = ("reasoning_content", "reasoning", "thinking", "reasoning_text")
        for field in reasoning_fields:
            reasoning = getattr(message, field, None)
            if reasoning:
                return str(reasoning).strip()
        try:
            data = message.model_dump()
            for field in reasoning_fields:
                reasoning = data.get(field)
                if reasoning:
                    return str(reasoning).strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _chat_choice_debug(choice):
        try:
            data = choice.model_dump()
        except Exception:
            try:
                data = json.loads(choice.model_dump_json())
            except Exception:
                return str(choice)
        message = data.get("message") or {}
        return json.dumps(
            {
                "finish_reason": data.get("finish_reason"),
                "message_keys": sorted(message.keys()),
                "content_len": len(str(message.get("content") or "")),
                "reasoning_content_len": len(str(message.get("reasoning_content") or "")),
                "message": message,
            },
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _extract_code_from_reasoning(reasoning_text):
        text = str(reasoning_text or "").strip()
        if not text:
            return ""
        if "```" in text:
            for part in text.split("```"):
                candidate = part.strip()
                if candidate.startswith("python"):
                    candidate = candidate[len("python"):].strip()
                if "def " in candidate and ("import " in candidate or "from " in candidate):
                    return candidate.strip()
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def ")):
                candidate = "\n".join(lines[idx:]).strip()
                if "def " in candidate:
                    return candidate
        return ""

    @staticmethod
    def _clean_code_generation_text(text):
        candidate = str(text or "").strip()
        if not candidate:
            return ""
        fenced = FrozenLLM._extract_code_from_reasoning(candidate)
        if fenced:
            candidate = fenced
        lines = candidate.splitlines()
        start_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("from ", "import ", "def ")):
                start_idx = idx
                break
        if start_idx is not None:
            candidate = "\n".join(lines[start_idx:]).strip()
        cut_markers = (
            "\nExplanation:",
            "\nNote:",
            "\nThe code",
            "\nThis code",
            "\nBut ",
            "\nHowever,",
        )
        for marker in cut_markers:
            pos = candidate.find(marker)
            if pos > 0:
                candidate = candidate[:pos].strip()
        lines = []
        for raw_line in candidate.splitlines():
            stripped = raw_line.lstrip()
            if stripped.startswith(("from ", "import ", "def ", "class ", "@")):
                lines.append(stripped)
            else:
                lines.append(raw_line.rstrip())
        candidate = "\n".join(lines).strip()
        parsed_prefix = FrozenLLM._longest_valid_python_prefix(candidate)
        if parsed_prefix:
            return parsed_prefix
        return candidate

    @staticmethod
    def _longest_valid_python_prefix(text):
        candidate = str(text or "").strip()
        if not candidate:
            return ""
        lines = candidate.splitlines()
        for end in range(len(lines), 0, -1):
            prefix = "\n".join(lines[:end]).strip()
            if not prefix:
                continue
            try:
                tree = ast.parse(prefix)
            except SyntaxError:
                continue
            if FrozenLLM._looks_like_python_submission(tree):
                return prefix
        return ""

    @staticmethod
    def _looks_like_python_submission(tree):
        if tree is None:
            return False
        has_function = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
        has_import = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
        return has_function and has_import

    @staticmethod
    def _extract_structured_reasoning_answer(reasoning_text):
        text = str(reasoning_text or "").strip()
        if not text:
            return ""
        fields = ("Explanation:", "Exact Answer:", "Confidence:")
        positions = [text.find(field) for field in fields]
        if any(pos < 0 for pos in positions):
            return ""
        start = min(positions)
        candidate = text[start:].strip()
        lines = []
        for raw_line in candidate.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(fields):
                lines.append(line)
                continue
            if lines:
                lines[-1] = f"{lines[-1]} {line}".strip()
            if len(lines) >= 3 and all(line.startswith(field) for line, field in zip(lines[:3], fields)):
                break
        if len(lines) < 3:
            return ""
        if not all(line.startswith(field) for line, field in zip(lines[:3], fields)):
            return ""
        return "\n".join(lines[:3]).strip()

    @staticmethod
    def _extract_action_from_reasoning(reasoning_text):
        text = str(reasoning_text or "").strip()
        if not text:
            return ""
        action_pattern = r"(Action:\s*[A-Za-z_]+\([^)\n]+\))"
        final_pattern = r"(Final Answer:\s*(?:[Vv]ar(?:iable)?\s*)?#\d+)"
        matches = re.findall(action_pattern, text)
        if matches:
            return matches[-1].strip()
        matches = re.findall(final_pattern, text)
        if matches:
            return matches[-1].strip()
        return ""

    def _create_completion(self, *, system_prompt, user_prompt, max_tokens, temperature, timeout, task_type=None):
        if self.protocol == "responses":
            request_tokens = max(128, int(max_tokens))
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=request_tokens,
                reasoning={"effort": self.responses_reasoning_effort},
                text={"verbosity": self.responses_text_verbosity},
                timeout=timeout,
            )
            content = self._extract_responses_text(response)
            if content:
                return content
            incomplete_reason = self._responses_incomplete_reason(response)
            if incomplete_reason == "max_output_tokens":
                retry_tokens = max(256, request_tokens * 2)
                retry_response = self.client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=retry_tokens,
                    reasoning={"effort": self.responses_reasoning_effort},
                    text={"verbosity": self.responses_text_verbosity},
                    timeout=timeout,
                )
                retry_content = self._extract_responses_text(retry_response)
                if retry_content:
                    return retry_content
            debug_payload = ""
            try:
                debug_payload = json.dumps(response.model_dump(), ensure_ascii=False, default=str)
            except Exception:
                debug_payload = str(response)
            raise RuntimeError(f"LLM returned empty responses content: {debug_payload[:2000]}")
        request_tokens = max(128, int(max_tokens))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=request_tokens,
            timeout=timeout
        )
        first_choice = response.choices[0]
        content = self._chat_message_text(first_choice)
        force_retry = False
        if content:
            if task_type == "code_generation":
                cleaned_code = self._clean_code_generation_text(content)
                if cleaned_code and self._longest_valid_python_prefix(cleaned_code):
                    return cleaned_code
                force_retry = True
            else:
                return content
        reasoning_text = self._chat_reasoning_text(first_choice)
        structured_reasoning_answer = self._extract_structured_reasoning_answer(reasoning_text)
        if structured_reasoning_answer:
            return structured_reasoning_answer
        action_from_reasoning = self._extract_action_from_reasoning(reasoning_text)
        if action_from_reasoning:
            return action_from_reasoning
        if task_type == "code_generation":
            code_from_reasoning = self._extract_code_from_reasoning(reasoning_text)
            if code_from_reasoning:
                return code_from_reasoning

        finish_reason = str(getattr(first_choice, "finish_reason", "") or "").strip().lower()
        if finish_reason == "length" or not content or force_retry:
            retry_messages = messages
            if task_type == "code_generation" and force_retry:
                retry_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer included analysis text or invalid Python. "
                            "Retry and output only executable Python code. "
                            "Return the full snippet with imports and the function body, and no explanation."
                        ),
                    }
                ]
            retry_tokens = max(8192 if task_type == "code_generation" else 256, request_tokens * 2)
            retry_response = self.client.chat.completions.create(
                model=self.model,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=retry_tokens,
                timeout=timeout
            )
            retry_choice = retry_response.choices[0]
            retry_content = self._chat_message_text(retry_choice)
            if retry_content:
                if task_type == "code_generation":
                    cleaned_retry_code = self._clean_code_generation_text(retry_content)
                    if cleaned_retry_code and self._longest_valid_python_prefix(cleaned_retry_code):
                        return cleaned_retry_code
                else:
                    return retry_content
            retry_reasoning_text = self._chat_reasoning_text(retry_choice)
            retry_structured_reasoning_answer = self._extract_structured_reasoning_answer(retry_reasoning_text)
            if retry_structured_reasoning_answer:
                return retry_structured_reasoning_answer
            retry_action_from_reasoning = self._extract_action_from_reasoning(retry_reasoning_text)
            if retry_action_from_reasoning:
                return retry_action_from_reasoning
            if task_type == "code_generation":
                retry_code_from_reasoning = self._extract_code_from_reasoning(retry_reasoning_text)
                cleaned_reasoning_code = self._clean_code_generation_text(retry_code_from_reasoning)
                if cleaned_reasoning_code and self._longest_valid_python_prefix(cleaned_reasoning_code):
                    return cleaned_reasoning_code
        if not content:
            raise RuntimeError(f"LLM returned empty chat content: {self._chat_choice_debug(retry_choice)}")
        if task_type == "code_generation":
            raise RuntimeError(f"LLM returned invalid code-generation content: {self._chat_choice_debug(retry_choice if 'retry_choice' in locals() else first_choice)}")
        return content

    def generate_action(self, prompt, valid_actions=None, task_type=None):
        if task_type == "code_generation":
            system_prompt = (
                "You are an expert Python coding agent solving benchmark tasks.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. The current task description and starter header are the source of truth.\n"
                "2. If historical memory appears, use only its abstract structural hints such as input family, transform family, output contract, failure boundary, and value bias.\n"
                "3. Never copy wording, starter code, function bodies, markdown, or formatting from memory.\n"
                "4. Preserve the required imports, function signature, entry point, and output contract for the current task.\n"
                "5. Output only raw Python code, with no explanation and no markdown fences."
            )
        elif task_type == "closed_ended_reasoning":
            system_prompt = (
                "You are solving a closed-ended reasoning benchmark item.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. The user prompt defines the exact required answer format and contract. Follow it exactly.\n"
                "2. Do not output chain-of-thought, hidden scratch work, or any extra sections beyond the required fields.\n"
                "3. If the prompt requires fields such as Explanation, Exact Answer, and Confidence, emit exactly those fields and nothing else.\n"
                "4. The final answer field must contain only the final answer span required by the contract.\n"
                "5. Keep the response concise and fully compliant with the requested format."
            )
        else:
            system_prompt = (
                "You are a Universal AI Meta-Agent capable of solving complex multi-step tasks across diverse environments (e.g., text games, OS desktops, web navigation). "
                "Your core capability is to map highly abstract logic onto concrete environmental actions.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Read the [Historical Memory] (if provided). It contains domain-agnostic meta-rules (e.g., Entity, Prerequisite, Terminal node). You must deduce how these abstract concepts map to the concrete objects in your current observation.\n"
                "2. Analyze the [Current Environment & State] to determine your current position within the logical state machine.\n"
                "3. If [Available Actions] are provided, you MUST select exactly ONE action from the list that best advances the state machine toward the Goal Task.\n"
                "4. Output ONLY the exact action string. NO reasoning, NO quotes, NO conversational filler."
            )
        
        user_prompt = prompt
        
        if valid_actions:
            valid_str = "\n".join([f"- {a}" for a in valid_actions])
            user_prompt += f"\n\n[Available Actions for this step]:\n{valid_str}\n\n"
            user_prompt += "Select exactly ONE action from the [Available Actions] list above based on the logic. Output ONLY the action text."

        #print(colored("\n" + "▼"*20 + " [发送给 LLM 的 User Prompt] " + "▼"*20, "blue"))
        #print(colored(user_prompt, "white"))
        #print(colored("▲"*75 + "\n", "blue"))

        self.last_action_debug = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "error": "",
        }

        try:
            self.generate_action_calls += 1
            raw_response = self._create_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.action_max_tokens,
                timeout=self.action_timeout,
                task_type=task_type,
            )
            self.last_action_debug["raw_response"] = raw_response
            action = raw_response

            if action.startswith("- "):
                action = action[2:]
            return action.strip()
            
        except Exception as e:
            self.last_action_debug["error"] = str(e)
            fallback = "" if task_type in {"code_generation", "closed_ended_reasoning"} else "look"
            self.last_action_debug["raw_response"] = fallback
            if task_type in {"code_generation", "closed_ended_reasoning"}:
                print(colored(f"⚠️ LLM API 调用彻底失败，返回空并记录错误: {e}", "red"))
            else:
                print(colored(f"⚠️ LLM API 调用彻底失败，执行保命动作 'look': {e}", "red"))
            return fallback

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=15),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def summarize_experience(self, task, traces, is_success):
        condensed_traces = []
        last_action = ""
        for i, t in enumerate(traces):
            if t['action'] != last_action or t.get('is_success', False): 
                condensed_traces.append(f"步 {i+1} | 动作: {t['action']} | 环境反馈: {t['obs'][:50]}... | 惩罚/奖励: {t.get('pddl_reward', -0.1)}")
                last_action = t['action']
                
        trace_text = "\n".join(condensed_traces[-15:]) 
        
        status = "✅ 成功" if is_success else "❌ 失败"

    
        if is_success:
            mistake_instruction = "分析轨迹中是否存在【严重的】绕路或死锁（例如：连续3次以上在毫无关联的节点打转，或者反复拿起放下同一物品）。🚨 极度重要警告：Agent 在未知环境中的初步试错和正常的寻路探测（1-3步内）是绝对合法的探索行为，绝不属于冗余！如果策略整体流畅、没有明显的降智死循环，你必须且只能输出“无”！绝不准吹毛求疵！"
        else:
            mistake_instruction = "一针见血地指出导致任务失败的致命逻辑断层（例如：遗漏了哪个前置状态？或在哪个节点卡死了？）。"

        
        system_prompt = f"""你是一名顶级的 AGI 元学习策略分析师。你的任务是从单次任务轨迹中，提取出【完全无视领域限制（Domain-Agnostic）】的纯逻辑状态转移法则。

【当前任务】: {task}
【最终状态】: {status}
【精简轨迹】:
{trace_text}

🚨 终极法则 (CRITICAL)：
1. 【强制抽象】：你【绝对禁止】使用特定环境名词（如：苹果、肥皂、柜子、洗、拿）。必须抽象为逻辑占位符，如：[目标实体]、[工具节点]、[终点容器]、[状态转化]、[获取控制权]、[寻路/定位]。
2. 【禁止抄袭】：你必须严格根据上方【精简轨迹】中真实发生的步骤进行总结！绝对禁止输出毫无意义的套话模板！

请严格按照以下 Markdown 格式输出：

🎯 **元状态法则 (Meta State-Transition Rule)**: [基于本次轨迹，用极度抽象的语言总结通关/失败的核心逻辑]
🔑 **抽象逻辑链 (Abstract Logic Chain)**: [基于本次轨迹，提取 3-5 步核心逻辑流。格式如：定位[目标实体] -> 动作B -> 动作C]
💀 **策略缺陷与冗余 (Strategic Flaws & Inefficiencies)**: [{mistake_instruction}]
"""
        
        print(colored("\n" + "▼"*20 + " [发送给 LLM 的 Prompt (经验复盘)] " + "▼"*20, "magenta"))
        print(colored(system_prompt, "white"))
        print(colored("▲"*75 + "\n", "magenta"))

        try:
            self.summarize_calls += 1
            return self._create_completion(
                system_prompt="You are an expert experience summarizer.",
                user_prompt=system_prompt,
                temperature=0.1,
                max_tokens=800,
                timeout=self.summary_timeout,
            )
        except Exception as e:
            print(colored(f"⚠️ API 经验总结超时/失败 (正在重试): {e}", "red"))
            raise e

    def get_api_call_count(self):
        return self.generate_action_calls + self.summarize_calls

    def get_last_action_debug(self):
        return dict(self.last_action_debug)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=False
    )
    def classify_route(self, prompt):
        system_prompt = (
            "You are a strict benchmark routing classifier.\n"
            "Decide whether the current task should stay on the cheap solver or escalate to a stronger solver.\n"
            "Return exactly one token: EASY or HARD.\n"
            "Use HARD only when the task likely requires deep symbolic reasoning, multi-step exact derivation, "
            "or unusually brittle answer precision beyond routine direct solving."
        )
        user_prompt = prompt
        self.last_route_debug = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": "",
            "error": "",
        }
        try:
            self.generate_action_calls += 1
            raw_response = self._create_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=16,
                timeout=30,
            )
            self.last_route_debug["raw_response"] = raw_response
            label = raw_response.strip().upper()
            if "HARD" in label:
                return "HARD"
            return "EASY"
        except Exception as e:
            self.last_route_debug["error"] = str(e)
            self.last_route_debug["raw_response"] = "EASY"
            print(colored(f"⚠️ Route probe failed, defaulting to EASY: {e}", "yellow"))
            return "EASY"

    def get_last_route_debug(self):
        return dict(self.last_route_debug)
