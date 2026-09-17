"""Embedding helpers and deterministic offline fallback."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional for non-image tests
    Image = None  # type: ignore


def _normalise(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [float(x) / norm for x in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    a = _normalise(left)
    b = _normalise(right)
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


class LocalEmbedding:
    """Small deterministic image/text embedding for tests and local pre-QC.

    It is not a semantic model.  Its purpose is to provide a stable contract
    and useful temporal/duplicate signals until the Ark embedding endpoint is
    enabled.
    """

    # 64 luminance cells + 3 x 16 coarse RGB blocks.
    dimension = 112

    def _hash_vector(self, value: str) -> List[float]:
        vector = [0.0] * self.dimension
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        for index in range(self.dimension):
            byte = digest[index % len(digest)]
            vector[index] = (byte / 255.0) * 2.0 - 1.0
        return _normalise(vector)

    def embed_text(self, text: str) -> List[float]:
        return self._hash_vector("text:" + text)

    def embed_image(self, path: Path) -> List[float]:
        if Image is None:
            return self._hash_vector("image:" + str(path) + ":" + str(path.stat().st_size))
        try:
            image = Image.open(path).convert("RGB").resize((8, 8))
        except Exception:
            return self._hash_vector("image:" + str(path) + ":" + str(path.stat().st_size))
        pixels = list(image.getdata())
        vector: List[float] = []
        # 8x8 luminance and coarse RGB summaries.  The resulting 112
        # dimensions are compact but sensitive to both layout and color drift.
        for r, g, b in pixels:
            luminance = (float(r) + float(g) + float(b)) / (3.0 * 255.0)
            # Center features around zero; otherwise two unrelated bright
            # images become almost collinear after normalization and cosine
            # similarity cannot detect scene changes.
            vector.append((luminance - 0.5) * 2.0)
        for channel in range(3):
            for block_y in range(4):
                for block_x in range(4):
                    values: List[float] = []
                    for y in range(block_y * 2, block_y * 2 + 2):
                        for x in range(block_x * 2, block_x * 2 + 2):
                            values.append(pixels[y * 8 + x][channel] / 255.0)
                    vector.append((sum(values) / len(values) - 0.5) * 2.0)
        return _normalise(vector)

    def embed_multimodal(self, inputs: Sequence[Mapping[str, str]]) -> List[float]:
        vectors: List[List[float]] = []
        for item in inputs:
            if item.get("type") == "text":
                vectors.append(self.embed_text(item.get("text", "")))
            elif item.get("type") == "image" and item.get("path"):
                vectors.append(self.embed_image(Path(item["path"])))
        if not vectors:
            return [0.0] * self.dimension
        averaged = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(self.dimension)]
        return _normalise(averaged)

    def temporal_consistency(self, frame_paths: Sequence[Path]) -> float:
        if len(frame_paths) < 2:
            return 1.0
        embeddings = [self.embed_image(path) for path in frame_paths]
        similarities = [cosine_similarity(a, b) for a, b in zip(embeddings, embeddings[1:])]
        return max(0.0, min(1.0, sum(similarities) / len(similarities)))

    def reference_similarity(self, reference: Optional[Path], frames: Sequence[Path]) -> float:
        if reference is None or not frames:
            return 0.8
        ref = self.embed_image(reference)
        values = [cosine_similarity(ref, self.embed_image(frame)) for frame in frames]
        # Map cosine [-1, 1] to a readable [0, 1] score.
        return max(0.0, min(1.0, (sum(values) / len(values) + 1.0) / 2.0))


def nearest_duplicate_pairs(
    paths: Sequence[Path], embedding: Optional[LocalEmbedding] = None, threshold: float = 0.995
) -> List[tuple[str, str, float]]:
    """Find genuinely frozen/duplicate frames.

    Cosine similarity alone is too forgiving for images dominated by a common
    wall or floor.  Pair it with a small grayscale pixel-difference check so a
    slowly moving camera is not mislabeled as a frozen video.
    """

    def pixel_difference(left: Path, right: Path) -> float:
        if Image is None:
            return 1.0 if left.stat().st_size != right.stat().st_size else 0.0
        try:
            first = Image.open(left).convert("L").resize((32, 18))
            second = Image.open(right).convert("L").resize((32, 18))
            values = zip(first.getdata(), second.getdata())
            return sum(abs(int(a) - int(b)) for a, b in values) / (255.0 * 32.0 * 18.0)
        except Exception:
            return 1.0

    embedding = embedding or LocalEmbedding()
    vectors = [(path, embedding.embed_image(path)) for path in paths]
    duplicates: List[tuple[str, str, float]] = []
    for index, (left_path, left) in enumerate(vectors):
        for right_path, right in vectors[index + 1 :]:
            score = cosine_similarity(left, right)
            # At threshold .999 this allows roughly 0.01% normalized pixel
            # change, while exact duplicates always remain detectable.
            difference_limit = max(1e-4, (1.0 - threshold) * 0.1)
            if score >= threshold and pixel_difference(left_path, right_path) <= difference_limit:
                duplicates.append((str(left_path), str(right_path), score))
    return duplicates


def extract_embedding_vectors(response: object) -> List[List[float]]:
    """Normalize common Ark SDK/REST embedding response shapes."""

    if hasattr(response, "model_dump"):
        response = response.model_dump()  # type: ignore[assignment]
    if isinstance(response, Mapping):
        data = response.get("data", response.get("embeddings", []))
    else:
        data = getattr(response, "data", [])
    vectors: List[List[float]] = []
    if isinstance(data, Mapping):
        data = [data]
    if isinstance(data, (list, tuple)):
        for item in data:
            if isinstance(item, Mapping):
                vector = item.get("embedding", item.get("vector"))
            else:
                vector = getattr(item, "embedding", getattr(item, "vector", None))
            if isinstance(vector, (list, tuple)):
                vectors.append([float(value) for value in vector])
    return vectors
