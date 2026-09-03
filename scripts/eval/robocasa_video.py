"""Dependency-light RoboCasa observation-to-video conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


def video_frame(observation: dict[str, Any]) -> np.ndarray:
    """Extract one external policy-camera frame without mutating the input."""

    candidates: list[tuple[int, str, np.ndarray]] = []

    def visit(path: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}/{key}", child)
            return
        array = np.asarray(value)
        if array.ndim not in (3, 4):
            return
        lower = path.lower()
        if not any(token in lower for token in ("pixel", "image", "camera")):
            return
        score = 10 if any(
            token in lower for token in ("wrist", "image2", "eye_in_hand")
        ) else 0
        candidates.append((score, path, array))

    visit("observation", observation)
    if not candidates:
        raise RuntimeError("no camera-valued observation leaf available for video")
    _, _, frame = min(candidates, key=lambda item: (item[0], item[1]))
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise RuntimeError(f"video recorder requires batch size one, got {frame.shape}")
        frame = frame[0]
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise RuntimeError(f"unsupported camera frame shape {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        frame = frame * 255.0 if float(np.nanmax(frame)) <= 1.0 else frame
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
