"""BilalAI 1.0 - model degerlendirme.

Validation loss olcer, ornek istemlerle cikti uretir, parametre sayisini
dogrular ve model durumunu dogru siniflandirir (sartname bolum 12/20).

Ornek:
  python "BilalAI 1.0/evaluate.py" --checkpoint "BilalAI 1.0/checkpoints/best/bilalai_1_0_best.pt"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BilalAIConfig, EXPECTED_PARAM_COUNT
from inference import BilalAIInference
from train import TokenBlockDataset, evaluate

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = MODEL_DIR / "checkpoints" / "best" / "bilalai_1_0_best.pt"

SAMPLE_PROMPTS = [
    "Merhaba, nasılsın?",
    "Python'da değişken nedir?",
    "2 + 2 kaç eder? İşlemi açıkla.",
    "Bir paragrafı üç cümlede özetle: Güneş, Dünya'daki yaşamın temel enerji kaynağıdır. Bitkiler güneş ışığıyla fotosentez yapar ve oksijen üretir.",
    "Aşağıdaki Python kodundaki hatayı açıkla: print(\"Merhaba\"",
    "Bir döngü örneği yaz ve satırlarını açıkla.",
    "İstanbul hakkında kısa bir paragraf yaz.",
]


def classify_status(global_step: int, val_loss: float | None) -> str:
    if global_step == 0:
        return "A) Mimari hazir - model henuz egitilmedi (rastgele agirlıklar)."
    if global_step < 1000:
        return (f"B) Egitim suruyor - {global_step} adim tamamlandi. "
                "Model henuz olgunlasmadi; cikti kalitesi sinirlidir.")
    return f"C) Egitilmis model - {global_step} adim. Kalite degerlendirme kumesiyle olculmelidir."


def main() -> int:
    parser = argparse.ArgumentParser(description="BilalAI 1.0 degerlendirme")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        engine = BilalAIInference(args.checkpoint, MODEL_DIR / "tokenizer_data", args.device)
    except FileNotFoundError as e:
        print(f"HATA: {e}")
        return 1

    cfg = engine.config
    device = engine.device
    lines = [
        f"Model: {engine.model_name}",
        f"Checkpoint: {Path(args.checkpoint).name}",
        f"Egitim adimi: {engine.global_step}",
        f"Cihaz: {device}",
        "",
    ]

    n_params = engine.model.num_unique_trainable_parameters()
    lines.append(f"Parametre sayisi: {n_params:,} (beklenen {EXPECTED_PARAM_COUNT:,}) "
                 f"-> {'DOGRU' if n_params == EXPECTED_PARAM_COUNT else 'HATALI'}")

    # Validation loss
    val_loss = None
    try:
        val_ds = TokenBlockDataset("validation", cfg.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "fp32")
        val_loss = evaluate(engine.model, val_loader, device, precision, engine.tokenizer.pad_id)
        lines.append(f"Validation loss: {val_loss:.4f}")
    except FileNotFoundError:
        lines.append("Validation loss: olculemedi (validation verisi yok).")
    lines.append("")

    status = classify_status(engine.global_step, val_loss)
    lines.append("DURUM: " + status)
    lines.append("")
    lines.append("ORNEK CIKTILAR (gozle incelenmeli; tek basina benchmark degildir):")
    lines.append("-" * 60)

    samples = []
    for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
        messages = [{"role": "user", "content": prompt}]
        res = engine.generate(messages=messages, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, seed=args.seed)
        lines.append(f"\n[{i}] Istem: {prompt}")
        lines.append(f"    Cikti: {res['text'] or '(bos)'}")
        lines.append(f"    (stop={res['stop_reason']}, token={res['num_new_tokens']})")
        samples.append({"prompt": prompt, "output": res["text"],
                        "stop_reason": res["stop_reason"]})

    report = "\n".join(lines)
    print(report)

    out_dir = MODEL_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_report.txt").write_text(report + "\n", encoding="utf-8")
    (out_dir / "evaluation_report.json").write_text(
        json.dumps({"model_name": engine.model_name, "global_step": engine.global_step,
                    "param_count": n_params, "validation_loss": val_loss,
                    "status": status, "samples": samples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRapor: {out_dir / 'evaluation_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())