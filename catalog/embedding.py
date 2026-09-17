"""Build neural text embeddings for native task labels.

The catalog scanner stays independent from ML libraries. This module is used by
``build_catalog.py`` only after the read-only scan completes. It turns every
unique ``(scene, native_task)`` pair into one point and preserves catalog record
ids so the browser can reuse all existing filters without recomputing vectors.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EMBEDDING_SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 64
TEXT_PREFIX = "query: "


def aggregate_task_terms(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate path records into scene-specific native task terms."""

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        scene = str(record.get("scene") or "").strip()
        task = str(record.get("native_task") or "").strip()
        record_id = str(record.get("id") or "").strip()
        if scene and task and record_id:
            grouped[(scene, task)].append(record_id)

    points = []
    for (scene, task), record_ids in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        stable_key = f"{scene}\x1f{task}".encode("utf-8")
        points.append(
            {
                "id": hashlib.sha1(stable_key).hexdigest()[:20],
                "scene": scene,
                "task": task,
                "record_ids": sorted(set(record_ids)),
            }
        )
    return points


def _fix_component_signs(coordinates: Any) -> None:
    """Remove the arbitrary sign choice made by SVD implementations in place."""

    import numpy as np

    for index in range(coordinates.shape[1]):
        column = coordinates[:, index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            coordinates[:, index] *= -1


def project_embeddings(embeddings: Any, *, dimensions: int = 2) -> tuple[Any, list[float]]:
    """Project dense normalized embeddings with deterministic centered PCA."""

    import numpy as np

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    rows = matrix.shape[0]
    if rows == 0:
        return np.zeros((0, dimensions), dtype=np.float64), [0.0] * dimensions
    if rows == 1:
        return np.zeros((1, dimensions), dtype=np.float64), [0.0] * dimensions

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    available = min(dimensions, singular_values.shape[0])
    coordinates = left[:, :available] * singular_values[:available]
    total_variance = float((singular_values * singular_values).sum())
    ratios = [
        float(value * value / total_variance) if total_variance else 0.0
        for value in singular_values[:available]
    ]
    if available < dimensions:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimensions - available)))
        ratios.extend([0.0] * (dimensions - available))
    _fix_component_signs(coordinates)
    return coordinates, ratios


def _resolve_device(torch_module: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def encode_task_terms(
    task_terms: list[str],
    *,
    model_name_or_path: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[Any, dict[str, Any]]:
    """Encode native task labels with multilingual E5 mean pooling."""

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the caller's environment
        raise RuntimeError(
            "building neural task embeddings requires PyTorch and Transformers; pass --skip-embedding to omit them"
        ) from exc

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_device = _resolve_device(torch, device)
    cache_path = cache_dir.resolve().as_posix() if cache_dir else None
    load_options = {
        "revision": revision,
        "cache_dir": cache_path,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_options)
    model = AutoModel.from_pretrained(model_name_or_path, **load_options)
    model.eval()
    model.to(resolved_device)

    encoded_batches = []
    with torch.inference_mode():
        for start in range(0, len(task_terms), batch_size):
            texts = [f"{TEXT_PREFIX}{task}" for task in task_terms[start : start + batch_size]]
            tokenized = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {key: value.to(resolved_device) for key, value in tokenized.items()}
            output = model(**tokenized)
            attention_mask = tokenized["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            pooled = functional.normalize(pooled, p=2, dim=1)
            encoded_batches.append(pooled.cpu())

    if encoded_batches:
        embeddings = torch.cat(encoded_batches, dim=0).numpy()
    else:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        embeddings = torch.empty((0, hidden_size), dtype=torch.float32).numpy()
    model_revision = getattr(model.config, "_commit_hash", None) or revision
    metadata = {
        "model_id": model_name_or_path,
        "model_revision": model_revision,
        "embedding_dimension": int(embeddings.shape[1]),
        "device_used": resolved_device,
        "max_length": max_length,
        "text_prefix": TEXT_PREFIX,
        "pooling": "attention_mask_mean_pooling_l2_normalized",
    }
    return embeddings, metadata


def build_embedding(
    records: Iterable[dict[str, Any]],
    *,
    generated_at: str,
    model_name_or_path: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Encode task terms with multilingual E5 and reduce them to two dimensions."""

    points = aggregate_task_terms(records)
    embeddings, model_metadata = encode_task_terms(
        [point["task"] for point in points],
        model_name_or_path=model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
        batch_size=batch_size,
    )
    coordinates, variance_ratios = project_embeddings(embeddings)
    for point, (x_value, y_value) in zip(points, coordinates):
        point["x"] = round(float(x_value), 7)
        point["y"] = round(float(y_value), 7)

    source_digest = hashlib.sha256()
    for point in points:
        source_digest.update(point["scene"].encode("utf-8"))
        source_digest.update(b"\x1f")
        source_digest.update(point["task"].encode("utf-8"))
        source_digest.update(b"\x1e")

    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "generated_at": generated_at,
        "method": "multilingual_e5_mean_pooling_pca",
        "method_label": "multilingual-e5-small 神经文本向量 + PCA",
        "grouping": ["scene", "native_task"],
        "dimensions": 2,
        "reduction": "centered_pca_svd",
        "component_sign_rule": "largest_absolute_coordinate_positive",
        "point_count": len(points),
        "explained_variance_ratio": [round(value, 7) for value in variance_ratios],
        "source_digest": source_digest.hexdigest(),
        **model_metadata,
        "points": points,
    }
