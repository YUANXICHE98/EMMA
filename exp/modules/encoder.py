import os
import json
import random
import time
from urllib import error, request

import numpy as np

class IntentEncoder:
    def __init__(self, config):
        """2. 意图编码器：使用真实的 OpenAI Embedding API"""
        encoder_config = config.get('encoder', {})
        llm_config = config.get('llm', {})

        api_key = (
            os.environ.get("EMMA_EMBEDDING_API_KEY")
            or os.environ.get("MEMRL_EMBEDDING_API_KEY")
            or os.environ.get("EMBEDDING_API_KEY")
            or encoder_config.get('api_key')
            or os.environ.get("EMMA_OPENAI_API_KEY")
            or os.environ.get("MEMRL_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or llm_config.get('api_key')
        )
        base_url = (
            os.environ.get("EMMA_EMBEDDING_BASE_URL")
            or os.environ.get("MEMRL_EMBEDDING_BASE_URL")
            or os.environ.get("EMBEDDING_BASE_URL")
            or encoder_config.get('base_url')
            or os.environ.get("EMMA_OPENAI_BASE_URL")
            or os.environ.get("MEMRL_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or llm_config.get('base_url')
        )

        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = (
            os.environ.get("EMMA_EMBEDDING_MODEL")
            or os.environ.get("MEMRL_EMBEDDING_MODEL")
            or os.environ.get("EMBEDDING_MODEL")
            or encoder_config.get('model_name', 'text-embedding-3-large')
        )
        self.timeout = float(
            os.environ.get("EMMA_EMBEDDING_TIMEOUT")
            or os.environ.get("MEMRL_EMBEDDING_TIMEOUT")
            or os.environ.get("EMBEDDING_TIMEOUT")
            or encoder_config.get('timeout', 45)
        )
        self.max_retries = int(
            os.environ.get("EMMA_EMBEDDING_MAX_RETRIES")
            or os.environ.get("MEMRL_EMBEDDING_MAX_RETRIES")
            or os.environ.get("EMBEDDING_MAX_RETRIES")
            or encoder_config.get('max_retries', 2)
        )
        self.retry_backoff_sec = float(
            os.environ.get("EMMA_EMBEDDING_RETRY_BACKOFF_SEC")
            or os.environ.get("MEMRL_EMBEDDING_RETRY_BACKOFF_SEC")
            or os.environ.get("EMBEDDING_RETRY_BACKOFF_SEC")
            or encoder_config.get('retry_backoff_sec', 2.0)
        )
        self.retry_max_backoff_sec = float(
            os.environ.get("EMMA_EMBEDDING_RETRY_MAX_BACKOFF_SEC")
            or os.environ.get("MEMRL_EMBEDDING_RETRY_MAX_BACKOFF_SEC")
            or os.environ.get("EMBEDDING_RETRY_MAX_BACKOFF_SEC")
            or encoder_config.get('retry_max_backoff_sec', 20.0)
        )
        self.trust_env_proxy = self._parse_bool(
            os.environ.get("EMMA_EMBEDDING_TRUST_ENV_PROXY")
            or os.environ.get("MEMRL_EMBEDDING_TRUST_ENV_PROXY")
            or os.environ.get("EMBEDDING_TRUST_ENV_PROXY")
            or encoder_config.get("trust_env_proxy"),
            default=False,
        )
        if self.trust_env_proxy:
            self.opener = request.build_opener()
        else:
            self.opener = request.build_opener(request.ProxyHandler({}))
        self.encode_calls = 0

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

    def _retry_delay(self, attempt, exc):
        retry_after = None
        if isinstance(exc, error.HTTPError):
            retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        base = self.retry_backoff_sec * (2 ** max(0, attempt - 1))
        capped = min(self.retry_max_backoff_sec, base)
        return capped + random.uniform(0.0, 0.5)

    def encode(self, s_t):
        """计算当前输入文本的意图向量 z_t"""
        last_error = None
        attempts = max(1, self.max_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                self.encode_calls += 1
                payload = json.dumps(
                    {
                        "input": s_t,
                        "model": self.model,
                    }
                ).encode("utf-8")
                req = request.Request(
                    f"{self.base_url}/embeddings",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with self.opener.open(req, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))

                vec = np.array(body["data"][0]["embedding"], dtype=float)
                norm = np.linalg.norm(vec)
                if norm == 0:
                    return vec
                return vec / norm
            except Exception as exc:
                last_error = exc
                print(f"⚠️ Embedding API 调用失败 (attempt {attempt}/{attempts}): {exc}")
                if attempt < attempts:
                    time.sleep(self._retry_delay(attempt, exc))

        raise RuntimeError(
            "Embedding provider unavailable after retries. Configure a working encoder provider via "
            "EMMA_EMBEDDING_API_KEY / EMMA_EMBEDDING_BASE_URL / encoder.api_key / encoder.base_url."
        ) from last_error

    def get_api_call_count(self):
        return self.encode_calls
