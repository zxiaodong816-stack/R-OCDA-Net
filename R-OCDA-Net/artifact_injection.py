"""Analyze model responses inside synthetic artifact regions.

The module is intentionally model-agnostic. It receives a clean image, an
artifact-corrupted image, and feature tensors or feature-like nested lists.
Model-specific code can then use these helpers to produce attention maps,
feature response maps, and artifact-region statistics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageEnhance


def _as_float_grid(values: Any) -> list[list[float]]:
    """Convert a 2-D tensor/array/list-like object to a Python float grid."""

    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [[float(v) for v in row] for row in values]


def build_artifact_mask(
    clean_image: Image.Image | str | Path,
    artifact_image: Image.Image | str | Path,
    threshold: int = 3,
) -> Image.Image:
    """Return a binary mask of pixels changed by artifact injection."""

    clean = Image.open(clean_image) if isinstance(clean_image, (str, Path)) else clean_image
    artifact = Image.open(artifact_image) if isinstance(artifact_image, (str, Path)) else artifact_image
    clean = clean.convert("RGB")
    artifact = artifact.convert("RGB")
    if clean.size != artifact.size:
        artifact = artifact.resize(clean.size, Image.Resampling.BILINEAR)

    diff = ImageChops.difference(clean, artifact).convert("L")
    mask = Image.new("L", diff.size, 0)
    source = diff.load()
    target = mask.load()
    width, height = diff.size
    for y in range(height):
        for x in range(width):
            target[x, y] = 255 if source[x, y] > threshold else 0
    return mask


def feature_response_map(feature: Any, normalize: bool = True) -> list[list[float]]:
    """Collapse CxHxW or HxW feature data to a 2-D response map."""

    if hasattr(feature, "detach"):
        import torch

        tensor = feature.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor.abs().mean(dim=0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2-D, 3-D, or 4-D feature tensor, got shape {tuple(tensor.shape)}")
        grid = _as_float_grid(tensor)
    else:
        data = feature
        if len(data) > 0 and isinstance(data[0][0], (list, tuple)):
            channels = data
            height = len(channels[0])
            width = len(channels[0][0])
            grid = []
            for y in range(height):
                row = []
                for x in range(width):
                    row.append(sum(abs(float(ch[y][x])) for ch in channels) / len(channels))
                grid.append(row)
        else:
            grid = _as_float_grid(data)

    if not normalize:
        return grid
    flat = [v for row in grid for v in row]
    low = min(flat)
    high = max(flat)
    if high == low:
        return [[0.0 for _ in row] for row in grid]
    return [[(v - low) / (high - low) for v in row] for row in grid]


def response_grid_to_image(response: Any, size: tuple[int, int] | None = None) -> Image.Image:
    """Convert a response map to an 8-bit grayscale image."""

    grid = feature_response_map(response) if not isinstance(response, list) else response
    height = len(grid)
    width = len(grid[0])
    image = Image.new("L", (width, height))
    pixels = image.load()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            pixels[x, y] = max(0, min(255, int(round(float(value) * 255))))
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def overlay_response_on_image(
    base_image: Image.Image | str | Path,
    response: Any,
    output_path: str | Path,
    alpha: float = 0.45,
) -> Path:
    """Overlay a red-yellow response map on the source image."""

    base = Image.open(base_image) if isinstance(base_image, (str, Path)) else base_image
    base = base.convert("RGB")
    gray = response_grid_to_image(response, size=base.size)
    heat = Image.merge("RGB", (gray, ImageEnhance.Brightness(gray).enhance(0.45), Image.new("L", gray.size, 0)))
    overlay = Image.blend(base, heat, alpha=alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def region_response_stats(response: Any, artifact_mask: Image.Image | str | Path) -> dict[str, float | int]:
    """Compute response statistics inside and outside artifact regions."""

    grid = response if isinstance(response, list) else feature_response_map(response)
    height = len(grid)
    width = len(grid[0])
    mask = Image.open(artifact_mask) if isinstance(artifact_mask, (str, Path)) else artifact_mask
    mask = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)

    artifact_values: list[float] = []
    background_values: list[float] = []
    mask_pixels = mask.load()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if mask_pixels[x, y] > 0:
                artifact_values.append(float(value))
            else:
                background_values.append(float(value))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return ordered[idx]

    artifact_mean = mean(artifact_values)
    background_mean = mean(background_values)
    eps = 1e-12
    return {
        "artifact_pixel_count": len(artifact_values),
        "background_pixel_count": len(background_values),
        "artifact_mean": artifact_mean,
        "background_mean": background_mean,
        "artifact_median": percentile(artifact_values, 0.5),
        "background_median": percentile(background_values, 0.5),
        "artifact_p95": percentile(artifact_values, 0.95),
        "background_p95": percentile(background_values, 0.95),
        "artifact_to_background_ratio": artifact_mean / (background_mean + eps),
        "artifact_minus_background": artifact_mean - background_mean,
    }


def write_stats_csv(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Write response statistics rows to CSV."""

    rows = list(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "model",
        "artifact",
        "strength",
        "layer",
        "artifact_mean",
        "background_mean",
        "artifact_to_background_ratio",
        "artifact_minus_background",
        "artifact_pixel_count",
        "background_pixel_count",
        "artifact_median",
        "background_median",
        "artifact_p95",
        "background_p95",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
