"""Small dependency-light clients for the Volcano Ark APIs.

The official OpenAI-compatible client is supported by the project, but the
REST implementation here keeps the smoke test runnable without installing a
large SDK.  API responses are retained by callers; secrets are never included
in exceptions or logs.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class MissingCredentialError(RuntimeError):
    pass


class ArkHTTPError(RuntimeError):
    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(f"Ark request failed ({status}): {message}")
        self.status = status
        self.body = body


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class ArkHTTPClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        user_agent: str = "navgen-pipeline/0.1",
    ) -> None:
        self.base_url = (base_url or os.getenv("ARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("ARK_API_KEY")
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    def _require_key(self) -> str:
        if not self.api_key:
            raise MissingCredentialError(
                "ARK_API_KEY is not set; use --mode offline or inject the key through a secret manager"
            )
        return self.api_key

    def request(self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None) -> Any:
        key = self._require_key()
        url = self.base_url + (path if path.startswith("/") else "/" + path)
        body: Optional[bytes] = None
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
                if not raw:
                    return {}
                content_type = response.headers.get("Content-Type", "")
                if "json" in content_type or raw[:1] in (b"{", b"["):
                    return json.loads(raw.decode("utf-8"))
                return raw
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            # Avoid echoing authorization headers or request bodies.
            message = raw[:1000]
            raise ArkHTTPError(exc.code, message, raw) from exc
        except urllib.error.URLError as exc:
            raise ArkHTTPError(0, str(exc.reason)) from exc


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield item
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield item
            yield from _walk_values(item)


def extract_text(response: Any) -> str:
    """Extract text from Responses API variants and SDK-like objects."""

    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    if isinstance(response, Mapping):
        if response.get("output_text"):
            return str(response["output_text"])
        # Responses API returns output[].content[].text in some deployments.
        for item in _walk_values(response):
            if isinstance(item, Mapping) and item.get("type") in {"output_text", "text"} and item.get("text"):
                return str(item["text"])
            if isinstance(item, str) and item.strip().startswith(("{", "[")):
                return item
    # SDK object fallback.
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    return str(text)
    raise ValueError("could not extract text from Ark Responses response")


class ArkResponsesClient:
    def __init__(self, http: Optional[ArkHTTPClient] = None) -> None:
        self.http = http or ArkHTTPClient()
        # Keep lightweight, secret-free call metadata so the pipeline can
        # report token usage and latency without retaining request/response
        # bodies.  ``call_history`` is intentionally public for batch audit
        # code; callers can clear it between runs if desired.
        self.call_history: List[Dict[str, Any]] = []
        self.last_call: Dict[str, Any] = {}
        self.last_usage: Dict[str, Any] = {}

    def generate_text(
        self,
        model: str,
        system: str,
        user_text: str,
        images: Optional[Sequence[str]] = None,
        max_output_tokens: int = 3000,
        thinking: Optional[Mapping[str, Any]] = None,
    ) -> str:
        content: List[Dict[str, Any]] = []
        for image in images or []:
            content.append({"type": "input_image", "image_url": image})
        content.append({"type": "input_text", "text": user_text})
        payload: Dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
            # Seed 2.x enables deep reasoning by default on the Responses
            # endpoint.  For our strict JSON planner/refiner/VLM calls that
            # can consume the entire completion budget before emitting a
            # message (the response then contains only a ``reasoning`` item).
            # Disable it by default; callers can opt in with e.g.
            # ``thinking={"type": "enabled"}`` when a long-form reasoning
            # task is actually desired.
            "thinking": dict(thinking) if thinking is not None else {"type": "disabled"},
        }
        started = time.monotonic()
        try:
            response = self.http.request("POST", "/responses", payload)
            text = extract_text(response)
        except Exception as exc:
            event = {
                "model": model,
                "status": "error",
                "latency_s": round(time.monotonic() - started, 6),
                "error_type": type(exc).__name__,
            }
            self.last_call = event
            self.last_usage = {}
            self.call_history.append(event)
            raise
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        event = {
            "model": model,
            "status": str(response.get("status", "completed")) if isinstance(response, Mapping) else "completed",
            "latency_s": round(time.monotonic() - started, 6),
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
            "response_id": str(response.get("id")) if isinstance(response, Mapping) and response.get("id") else None,
        }
        self.last_call = event
        self.last_usage = dict(event["usage"])
        self.call_history.append(event)
        return text


class SeedanceClient:
    """REST adapter for Ark ``contents/generations/tasks``."""

    def __init__(self, http: Optional[ArkHTTPClient] = None) -> None:
        self.http = http or ArkHTTPClient()

    def create_task(
        self,
        model: str,
        content: Sequence[Mapping[str, Any]],
        duration: int = 8,
        resolution: str = "720p",
        ratio: str = "16:9",
        generate_audio: bool = False,
        seed: Optional[int] = None,
        return_last_frame: bool = False,
        watermark: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Seedance 2.0 accepts integer durations from 4 to 15 seconds (or
        # -1 for service-selected duration).  Fail locally so a batch does
        # not spend a request on an invalid task.
        if int(duration) != -1 and not 4 <= int(duration) <= 15:
            raise ValueError("Seedance 2.0 duration must be an integer in [4, 15] or -1")
        payload: Dict[str, Any] = {
            "model": model,
            "content": [dict(item) for item in content],
            "duration": int(duration),
            "resolution": resolution,
            "ratio": ratio,
            "generate_audio": bool(generate_audio),
            "return_last_frame": bool(return_last_frame),
            "watermark": bool(watermark),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if extra:
            payload.update(dict(extra))
        response = self.http.request("POST", "/contents/generations/tasks", payload)
        if not isinstance(response, Mapping) or not response.get("id"):
            raise ArkHTTPError(200, "create task response did not contain an id", json.dumps(response)[:1000])
        return dict(response)

    def get_task(self, task_id: str) -> Dict[str, Any]:
        response = self.http.request("GET", f"/contents/generations/tasks/{task_id}")
        if not isinstance(response, Mapping):
            raise ArkHTTPError(200, "task response was not JSON", str(response)[:1000])
        return dict(response)

    def wait_for_task(
        self,
        task_id: str,
        poll_s: float = 5.0,
        timeout_s: float = 900.0,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.get_task(task_id)
            status = str(result.get("status", "")).lower()
            if status in {"succeeded", "success", "completed", "failed", "expired", "cancelled"}:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Seedance task {task_id} did not finish within {timeout_s:.0f}s")
            time.sleep(max(0.2, poll_s))

    @staticmethod
    def video_url(task_response: Mapping[str, Any]) -> Optional[str]:
        for key in ("video_url", "url"):
            value = task_response.get(key)
            if isinstance(value, str) and value:
                return value
        content = task_response.get("content")
        if isinstance(content, Mapping):
            for key in ("video_url", "url"):
                value = content.get(key)
                if isinstance(value, str) and value:
                    return value
            video = content.get("video")
            if isinstance(video, Mapping) and isinstance(video.get("url"), str):
                return str(video["url"])
        # Some gateway versions wrap the task in ``data`` or ``output``.
        for key in ("data", "output", "result"):
            nested = task_response.get(key)
            if isinstance(nested, Mapping):
                found = SeedanceClient.video_url(nested)
                if found:
                    return found
        return None

    def download(self, url: str, destination: Path, timeout_s: Optional[float] = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": self.http.user_agent}, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_s or self.http.timeout_s) as response:
            data = response.read()
        if not data:
            raise IOError("downloaded video is empty")
        destination.write_bytes(data)
        return destination


class ArkEmbeddingClient:
    """Optional wrapper around ``volcenginesdkarkruntime``.

    The SDK is intentionally imported lazily.  When it is not installed the
    pipeline uses the deterministic local embedding implementation instead.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("ARK_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key if api_key is not None else os.getenv("ARK_API_KEY")
        if not self.api_key:
            raise MissingCredentialError("ARK_API_KEY is required for ArkEmbeddingClient")
        try:
            from volcenginesdkarkruntime import Ark  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "volcengine-python-sdk[ark] is not installed; use LocalEmbedding or install the live extra"
            ) from exc
        try:
            self.client = Ark(api_key=self.api_key, base_url=self.base_url)
        except TypeError:
            # Older SDK releases only accept api_key and use the regional
            # default endpoint internally.
            self.client = Ark(api_key=self.api_key)

    def create(self, inputs: Sequence[Mapping[str, Any]]) -> Any:
        response = self.client.multimodal_embeddings.create(model=self.model, input=list(inputs))
        return response


def file_to_data_url(path: Path, mime: str = "image/jpeg") -> str:
    """Encode a small local image for an OpenAI-compatible image input."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
