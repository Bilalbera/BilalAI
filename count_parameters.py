"""Gercek model nesnesinden benzersiz egitilebilir parametre sayisini hesaplar.

Sartname bolum 11: sayi tahmin edilmez; gercek tensor'lerden sayilir.
Weight tying nedeniyle ayni tensor iki kez sayilmaz.
Beklenen deger 118075392 degilse betik basarisiz olur (exit code 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from config import BilalAIConfig, EXPECTED_PARAM_COUNT
from model import build_model


def count_unique_trainable(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def main() -> int:
    cfg = BilalAIConfig()
    model = build_model(cfg)
    model.eval()

    total = count_unique_trainable(model)

    tok = model.token_embedding.weight.numel()
    pos = model.position_embedding.weight.numel()
    per_layer = count_unique_trainable(model.blocks[0])
    layers = per_layer * cfg.n_layers
    final_norm = model.final_norm.weight.numel()

    lines = [
        f"Model: {cfg.model_name}",
        f"vocab_size={cfg.vocab_size} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"n_heads={cfg.n_heads} head_dim={cfg.head_dim} ffn_dim={cfg.ffn_dim} "
        f"max_seq_len={cfg.max_seq_len}",
        "",
        "Bilesen bazinda parametre sayisi:",
        f"  Token embedding : {tok:>12,}",
        f"  Position emb    : {pos:>12,}",
        f"  Bir katman      : {per_layer:>12,}",
        f"  {cfg.n_layers} katman       : {layers:>12,}",
        f"  Son RMSNorm     : {final_norm:>12,}",
        f"  lm_head         : tied (token embedding ile paylasilir, ayri sayilmaz)",
        "",
        f"TOPLAM (benzersiz, egitilebilir): {total:,}",
        f"Beklenen (sartname)             : {EXPECTED_PARAM_COUNT:,}",
        f"Yuvarlanmis boyut               : {total / 1e6:.1f}M",
    ]
    report = "\n".join(lines)
    print(report)

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameter_report.txt").write_text(report + "\n", encoding="utf-8")

    if total != EXPECTED_PARAM_COUNT:
        print(f"\nHATA: parametre sayisi eslesmiyor ({total} != {EXPECTED_PARAM_COUNT}).", file=sys.stderr)
        return 1
    print("\nDOGRULANDI: parametre sayisi sartname ile eslesiyor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())