from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = False) -> dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))

    report: dict[str, Any] = {
        "python_random_seed": int(seed),
        "numpy_random_seed": int(seed),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "torch_available": False,
        "torch_seed": None,
        "deterministic_torch_requested": bool(deterministic_torch),
        "deterministic_torch_applied": False,
        "limitations": [
            "GPU kernels, data-loader workers, and external solver implementations may still introduce small non-determinism."
        ],
    }

    try:
        import torch
    except ImportError:
        return report

    report["torch_available"] = True
    report["torch_seed"] = int(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            report["deterministic_torch_applied"] = True
        except Exception as exc:  # pragma: no cover - depends on installed torch build.
            report["limitations"].append(f"Could not fully enable deterministic PyTorch algorithms: {exc}")
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    return report
