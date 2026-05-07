from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight candidate secondary models against the current VIMSAI/OpenAI-compatible route.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Candidate model ids to test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("environment/hle/results/model_preflight.json"),
        help="Where to save the JSON summary",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["chat", "responses"],
        choices=["chat", "responses"],
        help="OpenAI-compatible protocols to test",
    )
    return parser.parse_args()


def classify_error(exc: Exception) -> tuple[str, str]:
    text = str(exc)
    lower = text.lower()
    if "model_not_found" in lower or "no available channel for model" in lower:
        return "model_not_found", text
    if "model_price_error" in lower or "价格未配置" in text:
        return "model_price_error", text
    if "invalid_request_error" in lower:
        return "invalid_request", text
    if "unauthorized" in lower or "401" in lower:
        return "unauthorized", text
    return "other_error", text


def _parse_bool(value: Any, default: bool = False) -> bool:
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


def _chat_message_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, list):
        chunks: list[str] = []
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


def _chat_reasoning_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    for field in ("reasoning_content", "reasoning", "thinking", "reasoning_text"):
        value = getattr(message, field, None)
        if value:
            return str(value).strip()
    try:
        data = message.model_dump()
    except Exception:
        return ""
    for field in ("reasoning_content", "reasoning", "thinking", "reasoning_text"):
        value = data.get(field)
        if value:
            return str(value).strip()
    return ""


def probe_chat(client: OpenAI, model: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "Return exactly OK."},
        {"role": "user", "content": "Say OK only."},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=8,
            timeout=30,
        )
        choice = response.choices[0]
        content = _chat_message_text(choice)
        reasoning = _chat_reasoning_text(choice)
        if not content:
            return {
                "model": model,
                "protocol": "chat",
                "status": "reasoning_only" if reasoning else "empty_output",
                "raw_output": content,
                "reasoning_output": reasoning,
                "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
            }
        return {
            "model": model,
            "protocol": "chat",
            "status": "chat_ok",
            "raw_output": content,
            "reasoning_output": reasoning,
            "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
        }
    except Exception as exc:
        status, message = classify_error(exc)
        return {
            "model": model,
            "protocol": "chat",
            "status": status,
            "error": message,
        }


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    try:
        data = response.model_dump()
    except Exception:
        try:
            data = json.loads(response.model_dump_json())
        except Exception:
            return str(response).strip()

    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def probe_responses(client: OpenAI, model: str) -> dict[str, Any]:
    try:
        response = client.responses.create(
            model=model,
            input="Say OK only.",
            instructions="Return exactly OK.",
            max_output_tokens=16,
            timeout=30,
        )
        content = _extract_response_text(response)
        if not content:
            return {
                "model": model,
                "protocol": "responses",
                "status": "empty_output",
                "raw_output": content,
            }
        return {
            "model": model,
            "protocol": "responses",
            "status": "responses_ok",
            "raw_output": content,
        }
    except Exception as exc:
        status, message = classify_error(exc)
        return {
            "model": model,
            "protocol": "responses",
            "status": status,
            "error": message,
        }


def probe_model(client: OpenAI, model: str, protocols: list[str]) -> dict[str, Any]:
    probes = []
    if "chat" in protocols:
        probes.append(probe_chat(client, model))
    if "responses" in protocols:
        probes.append(probe_responses(client, model))
    best = next((probe for probe in probes if probe["status"] in {"chat_ok", "responses_ok"}), None)
    return {
        "model": model,
        "best_status": best["status"] if best else probes[0]["status"] if probes else "not_tested",
        "best_protocol": best["protocol"] if best else "",
        "probes": probes,
    }


def main() -> None:
    args = parse_args()
    api_key = (
        os.environ.get("EMMA_OPENAI_API_KEY")
        or os.environ.get("MEMRL_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    base_url = (
        os.environ.get("EMMA_OPENAI_BASE_URL")
        or os.environ.get("MEMRL_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    if not api_key:
        raise RuntimeError("Missing EMMA_OPENAI_API_KEY / MEMRL_OPENAI_API_KEY / OPENAI_API_KEY")

    client_args: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_args["base_url"] = base_url
    trust_env_proxy = _parse_bool(
        os.environ.get("EMMA_OPENAI_TRUST_ENV_PROXY")
        or os.environ.get("MEMRL_OPENAI_TRUST_ENV_PROXY")
        or os.environ.get("OPENAI_TRUST_ENV_PROXY"),
        default=False,
    )
    max_retries = int(
        os.environ.get("EMMA_OPENAI_MAX_RETRIES")
        or os.environ.get("MEMRL_OPENAI_MAX_RETRIES")
        or os.environ.get("OPENAI_MAX_RETRIES")
        or 0
    )
    client_args["http_client"] = httpx.Client(trust_env=trust_env_proxy)
    client_args["max_retries"] = max_retries
    client = OpenAI(**client_args)

    results = [probe_model(client, model, protocols=list(args.protocols)) for model in args.models]
    payload = {
        "base_url": base_url or "",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
