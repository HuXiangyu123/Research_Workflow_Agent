#!/usr/bin/env python3
"""Check whether GPT models configured in .env are callable.

Usage:
    python tests/api/check_env_gpt_models.py
    python tests/api/check_env_gpt_models.py --prompt "Return: ok"
    python tests/api/check_env_gpt_models.py --timeout 20
    python tests/api/check_env_gpt_models.py --skip-gateway-models
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ProbeResult:
    model: str
    source_keys: list[str]
    ok: bool
    elapsed_s: float
    detail: str


@dataclass
class GatewayModelsResult:
    ok: bool
    elapsed_s: float
    models: list[str]
    detail: str


def _collect_gpt_models_from_env(env_path: Path) -> OrderedDict[str, list[str]]:
    values = dotenv_values(env_path)
    model_to_keys: OrderedDict[str, list[str]] = OrderedDict()

    for key, value in values.items():
        if not key:
            continue
        if "model" not in key.lower():
            continue
        model = str(value or "").strip()
        if not model:
            continue
        if "gpt" not in model.lower():
            continue
        if model not in model_to_keys:
            model_to_keys[model] = []
        model_to_keys[model].append(key)

    return model_to_keys


def _extract_content(resp) -> str:
    content = getattr(resp, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(content).strip()


def _probe_model(
    *,
    model: str,
    source_keys: list[str],
    api_key: str,
    base_url: str,
    prompt: str,
    timeout_s: int,
    max_tokens: int,
) -> ProbeResult:
    started = time.perf_counter()
    try:
        kwargs = {
            "model": model,
            "api_key": api_key,
            "timeout": timeout_s,
            "max_retries": 0,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url

        llm = ChatOpenAI(**kwargs)
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = _extract_content(resp)
        elapsed = time.perf_counter() - started
        preview = text.replace("\n", " ")[:120] if text else "<empty response>"
        return ProbeResult(
            model=model,
            source_keys=source_keys,
            ok=True,
            elapsed_s=elapsed,
            detail=preview,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return ProbeResult(
            model=model,
            source_keys=source_keys,
            ok=False,
            elapsed_s=elapsed,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _fetch_gateway_gpt_models(
    *,
    api_key: str,
    base_url: str,
    timeout_s: int,
) -> GatewayModelsResult:
    started = time.perf_counter()

    if not base_url:
        return GatewayModelsResult(
            ok=False,
            elapsed_s=time.perf_counter() - started,
            models=[],
            detail="OPENAI_API_BASE is empty; skipped gateway model inventory.",
        )

    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return GatewayModelsResult(ok=False, elapsed_s=elapsed, models=[], detail=detail)

        payload = resp.json()
        items = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = sorted({
            str(item.get("id", "")).strip()
            for item in items
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        })
        gpt_models = [m for m in model_ids if "gpt" in m.lower()]

        return GatewayModelsResult(
            ok=True,
            elapsed_s=elapsed,
            models=gpt_models,
            detail=f"Fetched {len(gpt_models)} GPT models from gateway.",
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return GatewayModelsResult(
            ok=False,
            elapsed_s=elapsed,
            models=[],
            detail=f"{type(exc).__name__}: {exc}",
        )


def _print_results(results: list[ProbeResult]) -> None:
    print("\nGPT model connectivity check")
    print("-" * 96)
    print(f"{'status':<8} {'elapsed_s':<10} {'model':<28} {'source_keys':<28} detail")
    print("-" * 96)

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        keys = ",".join(result.source_keys)
        print(
            f"{status:<8} {result.elapsed_s:<10.2f} {result.model:<28} {keys:<28} {result.detail}"
        )

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print("-" * 96)
    print(f"Summary: {passed}/{total} models callable")


def _print_gateway_models(result: GatewayModelsResult) -> None:
    print("\nGateway GPT model inventory")
    print("-" * 96)
    status = "PASS" if result.ok else "WARN"
    print(f"status={status} elapsed_s={result.elapsed_s:.2f} detail={result.detail}")
    if result.ok and result.models:
        for model in result.models:
            print(model)
    print("-" * 96)


def _print_model_comparison(
    probe_results: list[ProbeResult],
    gateway_result: GatewayModelsResult | None,
) -> None:
    print("\nConfigured-vs-gateway comparison")
    print("-" * 96)
    print(f"{'model':<34} {'probe':<8} {'in_gateway':<11} source_keys")
    print("-" * 96)

    gateway_set = set(gateway_result.models) if gateway_result and gateway_result.ok else set()
    has_gateway = bool(gateway_set)

    for r in probe_results:
        probe = "PASS" if r.ok else "FAIL"
        if has_gateway:
            in_gateway = "YES" if r.model in gateway_set else "NO"
        else:
            in_gateway = "N/A"
        print(f"{r.model:<34} {probe:<8} {in_gateway:<11} {','.join(r.source_keys)}")

    if has_gateway:
        configured = {r.model for r in probe_results}
        missing = sorted(configured - gateway_set)
        available_not_used = sorted(gateway_set - configured)
        print("-" * 96)
        if missing:
            print("Configured but not listed by gateway:")
            for model in missing:
                print(f"- {model}")
        else:
            print("All configured GPT models are listed by gateway.")

        if available_not_used:
            print("Gateway GPT models not configured in .env:")
            for model in available_not_used:
                print(f"- {model}")
    print("-" * 96)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPT models from .env by making real chat calls.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file (default: .env)",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: ok",
        help="Minimal prompt sent to each model",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max response tokens for the probe request",
    )
    parser.add_argument(
        "--skip-gateway-models",
        action="store_true",
        help="Skip querying gateway /models inventory",
    )
    args = parser.parse_args()

    env_path = (REPO_ROOT / args.env_file).resolve()
    if not env_path.exists():
        print(f"ERROR: env file not found: {env_path}")
        return 1

    # Project rule: always load .env explicitly in scripts/tests that depend on env.
    load_dotenv(env_path)

    gpt_models = _collect_gpt_models_from_env(env_path)
    if not gpt_models:
        print(f"ERROR: no GPT model entries found in {env_path}")
        return 1

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is empty; cannot probe GPT models.")
        return 1

    base_url = os.getenv("OPENAI_API_BASE", "").strip()
    results: list[ProbeResult] = []

    gateway_result: GatewayModelsResult | None = None
    if not args.skip_gateway_models:
        gateway_result = _fetch_gateway_gpt_models(
            api_key=api_key,
            base_url=base_url,
            timeout_s=args.timeout,
        )

    for model, source_keys in gpt_models.items():
        result = _probe_model(
            model=model,
            source_keys=source_keys,
            api_key=api_key,
            base_url=base_url,
            prompt=args.prompt,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
        )
        results.append(result)

    _print_results(results)
    if gateway_result is not None:
        _print_gateway_models(gateway_result)
    _print_model_comparison(results, gateway_result)
    all_ok = all(r.ok for r in results)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
