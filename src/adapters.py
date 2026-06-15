"""통합 추론 어댑터: generate(model, prompt) -> text

백엔드: ollama(HTTP) / huggingface transformers / google-genai
모델 추가 시 MODELS에 항목만 추가하면 runner는 수정 불필요.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

MODELS = {
    "exaone3.5": {"backend": "ollama", "id": "exaone3.5:7.8b"},
    # 크기 사다리 (T2-1): 패밀리·한국어비중 고정, 크기만 변화 — 교란 없는 비교
    "exaone-2.4b": {"backend": "ollama", "id": "exaone3.5:2.4b"},
    "exaone-32b": {"backend": "ollama", "id": "exaone3.5:32b"},
    # qwen3는 thinking 모델 — 경어 효과 측정에 사고 토큰이 끼면 안 되므로 비활성화
    "qwen3": {"backend": "ollama", "id": "qwen3:8b", "extra": {"think": False}},
    "llama3.1": {"backend": "ollama", "id": "llama3.1:8b"},
    "hcx-seed": {"backend": "hf", "id": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B"},
    "gemini": {"backend": "genai", "id": "gemini-2.5-flash"},
}

_HF_CACHE: dict = {}


def generate(model: str, prompt: str, *, max_tokens: int = 256,
             temperature: float = 0.0, seed: int = 42) -> str:
    cfg = MODELS[model]
    backend = cfg["backend"]
    if backend == "ollama":
        return _ollama(cfg, prompt, max_tokens, temperature, seed)
    if backend == "hf":
        return _hf(cfg, prompt, max_tokens, temperature)
    if backend == "genai":
        return _genai(cfg, prompt, max_tokens, temperature)
    raise ValueError(f"unknown backend: {backend}")


def _ollama(cfg, prompt, max_tokens, temperature, seed):
    payload = {
        "model": cfg["id"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
    }
    payload.update(cfg.get("extra", {}))
    try:
        return _ollama_post(payload)
    except urllib.error.HTTPError as e:
        # 구버전 ollama 서버가 think 파라미터를 거부하는 경우 제거 후 1회 재시도
        if "think" in payload and e.code == 400:
            payload.pop("think")
            return _ollama_post(payload)
        raise


def _ollama_post(payload):
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body["response"].strip()


def _hf(cfg, prompt, max_tokens, temperature):
    """HyperCLOVA X SEED 등 Ollama 미지원 모델용. transformers + torch 필요."""
    if cfg["id"] not in _HF_CACHE:
        import torch  # noqa: F401  (지연 임포트 — 미설치 시 여기서만 실패)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["id"])
        mdl = AutoModelForCausalLM.from_pretrained(cfg["id"], torch_dtype="auto", device_map="auto")
        _HF_CACHE[cfg["id"]] = (tok, mdl)
    tok, mdl = _HF_CACHE[cfg["id"]]
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def _genai(cfg, prompt, max_tokens, temperature):
    """Gemini 보조 기준점. GEMINI_API_KEY 환경 변수 필요. google-genai 패키지 사용."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=cfg["id"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (resp.text or "").strip()
