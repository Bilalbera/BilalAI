"""BilalAI 1.0 - checkpoint kaydetme / yukleme.

Sartname bolum 8:
  - Checkpoint model + optimizer + scheduler + adim/epoch + config + tokenizer
    kimligi + RNG durumlari + veri manifest bilgisi icerir.
  - Yazim atomiktir: once gecici dosyaya yaz, sonra hedefin yerine gecir.
  - En iyi checkpoint korunur; egitime devam ile yalnizca inference ayridir.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_tokenizer_hash(tokenizer_dir: str | Path) -> str:
    """tokenizer.json dosyasinin SHA-256 ozeti (tokenizer kimligi)."""
    tok_path = Path(tokenizer_dir) / "tokenizer.json"
    if not tok_path.exists():
        return "missing"
    return hashlib.sha256(tok_path.read_bytes()).hexdigest()


def capture_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _to_cpu_byte(state: Any) -> Any:
    # RNG durumlari her zaman CPU ByteTensor olmali; checkpoint GPU'ya yuklense bile.
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().to(torch.uint8)
    return state


def restore_rng_states(states: dict[str, Any]) -> None:
    if not states:
        return
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(_to_cpu_byte(states["torch"]))
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_to_cpu_byte(s) for s in states["cuda"]])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: Any,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    global_step: int = 0,
    epoch: int = 0,
    best_validation_loss: float = float("inf"),
    tokenizer_hash: str = "missing",
    rng_states: dict[str, Any] | None = None,
    dataset_manifest: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomik checkpoint yazimi. Egitim durumunu tam olarak saklar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "model_config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "tokenizer_hash": tokenizer_hash,
        "rng_states": rng_states if rng_states is not None else capture_rng_states(),
        "dataset_manifest": dataset_manifest or {},
    }
    if extra:
        payload["extra"] = extra

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomik: yarim dosya hedefin uzerine yazmaz
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Tam checkpoint'i yukler (egitime devam icin)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadi: {path}. Model henuz egitilmemis olabilir."
        )
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_weights(
    model: torch.nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Yalnizca model agirlıklarını yukler (inference icin). Optimizer gerekmez."""
    ckpt = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return ckpt