"""BilalAI 1.0 model ve calisma yapilandirmasi.

Mimari degerleri BilalAI 1.0 Teknik Sartnamesi bolum 2 ile birebir aynidir.
Bu degerler degistirilirse parametre sayisi (118075392) gecerliligini yitirir;
count_parameters.py ve test_architecture.py beklenen degeri bilinclli olarak
guncellenmeden mimari degisikligi kabul edilmez.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


# Teknik sartnamede tanimlanan kesin mimari parametre sayisi.
EXPECTED_PARAM_COUNT = 118_075_392


@dataclass
class BilalAIConfig:
    # --- Kimlik ---
    model_name: str = "BilalAI 1.0"
    model_type: str = "decoder_only_transformer"

    # --- Mimari (sartname bolum 2) ---
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 13
    n_heads: int = 12
    head_dim: int = 64
    ffn_dim: int = 3072
    max_seq_len: int = 1024
    activation: str = "gelu"
    normalization: str = "rmsnorm"
    attention: str = "causal_multi_head_self_attention"
    embedding_tying: bool = True
    linear_bias: bool = True
    learned_position_emb: bool = True
    dropout: float = 0.1
    rms_norm_eps: float = 1e-5

    # --- Egitim varsayilanlari (sartname bolum 7) ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_ratio: float = 0.02
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    precision: str = "bf16"  # bf16 | fp16 | fp32
    seed: int = 42

    # --- Uretim varsayilanlari (sartname bolum 9) ---
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model ({self.d_model}) == n_heads * head_dim "
                f"({self.n_heads} * {self.head_dim}) olmali."
            )
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size ve max_seq_len pozitif olmali.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout [0, 1) araliginda olmali.")

    # --- JSON donusumleri ---
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BilalAIConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BilalAIConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config dosyasi bulunamadi: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    cfg = BilalAIConfig()
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))