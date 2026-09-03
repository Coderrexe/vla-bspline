"""Small, dependency-light helpers for deterministic evaluation artifacts.

This module deliberately does not import Torch or LeRobot.  The evaluator must
set CUDA determinism environment variables *before* importing Torch, while the
helpers below should remain unit-testable on a CPU-only development machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ACTION_TRACE_PROTOCOL = "executed_numpy_action_v1"


class ExecutedActionTrace:
    """Hash the exact NumPy arrays passed to ``env.step`` in call order.

    Each update is framed by a canonical JSON header containing dtype and shape
    plus an explicit payload length.  This prevents ambiguous concatenations
    and makes shape/dtype changes visible even when raw bytes happen to match.
    The array payload itself is not rounded or normalized: the digest tracks
    the exact contiguous bytes executed by the environment wrapper.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(ACTION_TRACE_PROTOCOL.encode("ascii") + b"\0")
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def update(self, action: Any) -> None:
        array = np.asarray(action)
        if array.dtype.hasobject:
            raise TypeError("object-dtype actions cannot be hashed reproducibly")
        contiguous = np.ascontiguousarray(array)
        header = json.dumps(
            {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        payload = contiguous.tobytes(order="C")
        self._digest.update(struct.pack("<Q", len(header)))
        self._digest.update(header)
        self._digest.update(struct.pack("<Q", len(payload)))
        self._digest.update(payload)
        self._count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def atomic_write_json_new(path: Path, payload: Any) -> None:
    """Atomically publish JSON at a path that must not already exist.

    Data are fsynced into a same-directory temporary file and then installed
    with ``link(2)``.  Unlike ``os.replace``, linking cannot overwrite a result
    created by another process between the initial existence check and publish.
    A crash may leave a hidden temporary file, but never a partial final result.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {path}") from error
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    # Some network filesystems do not implement directory
                    # fsync. The no-clobber link has already published a
                    # complete, fsynced file in that case.
                    pass
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
