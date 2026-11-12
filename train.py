"""BilalAI 1.0 - veri hazirlama ve causal language modeling egitimi.

Sartname bolum 6-7-8:
  - Veri temizleme, tekrar kontrolu, dokuman düzeyinde train/validation/test ayrimi.
  - 1024 token'lik bloklar (shift icin 1025 uzunlugunda saklanir).
  - AdamW, warmup + cosine, gradient clipping, mixed precision, gradient accumulation.
  - Atomik checkpoint, egitime devam, validation loss ve log kaydi.

Kullanim:
  python "BilalAI 1.0/train.py" --prepare-data
  python "BilalAI 1.0/train.py" --max-steps 100
  python "BilalAI 1.0/train.py" --resume "BilalAI 1.0/checkpoints/latest/bilalai_1_0_latest.pt"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import BilalAIConfig
from model import build_model
from tokenizer import ByteLevelBPETokenizer
from checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_tokenizer_hash,
    capture_rng_states,
    restore_rng_states,
)

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CKPT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = MODEL_DIR / "logs"
LATEST_PATH = CKPT_DIR / "latest" / "bilalai_1_0_latest.pt"
BEST_PATH = CKPT_DIR / "best" / "bilalai_1_0_best.pt"


# ====================================================================== #
# VERI HAZIRLAMA
# ====================================================================== #
def clean_text(text: str) -> str:
    """Kontrol karakterlerini temizle, Turkce karakterleri koru, bosluklari duzenle."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    # Satir sonu ve sekme disindaki kontrol karakterlerini kaldir
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_documents(text: str) -> list[str]:
    """Metni dokumanlara ayir. Ayni konusmanin parcalari birlikte kalir (sizinti onlemi)."""
    text = text.strip()
    if not text:
        return []
    if "<system>" in text:
        parts = re.split(r"(?=<system>)", text)
    elif "<turn_end>" in text:
        parts = re.split(r"(?<=<turn_end>)", text)
    else:
        parts = re.split(r"\n\s*\n", text)
    docs = [p.strip() for p in parts if p.strip()]
    return [d for d in docs if len(d) >= 20]


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def dedup_documents(docs: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    unique: list[str] = []
    removed = 0
    for d in docs:
        key = normalize_for_dedup(d)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(d)
    return unique, removed


def split_documents(docs: list[str], seed: int) -> dict[str, list[str]]:
    n = len(docs)
    ratio = 0.01 if n >= 500 else 0.05  # kucuk veride yeterli dogrulama ornegi birak
    n_val = max(2, int(n * ratio)) if n >= 6 else max(1, n // 5)
    n_test = max(2, int(n * ratio)) if n >= 6 else max(1, n // 5)
    n_val = min(n_val, max(1, n // 3))
    n_test = min(n_test, max(1, n // 3))

    idx = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return {
        "train": [docs[i] for i in train_idx],
        "validation": [docs[i] for i in val_idx],
        "test": [docs[i] for i in test_idx],
    }


def build_token_blocks(docs, tokenizer, seq_len, out_bin: Path, pad_id: int) -> dict:
    """Dokumanlari bos+icerik+eos olarak kodla, 1025'lik bloklara stream et.

    Bellek dostu: tam bloklar diske yazilir, bellekte yalnizca bir blok tutulur.
    """
    block_size = seq_len + 1
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    buffer = array("i")
    num_blocks = 0
    num_real_tokens = 0

    with open(out_bin, "wb") as f:
        for doc in docs:
            ids = tokenizer.encode(doc, add_special_tokens=True)
            num_real_tokens += len(ids)
            buffer.extend(ids)
            while len(buffer) >= block_size:
                f.write(buffer[:block_size].tobytes())
                num_blocks += 1
                del buffer[:block_size]
        if len(buffer) > 0:
            while len(buffer) < block_size:
                buffer.append(pad_id)
            f.write(buffer.tobytes())
            num_blocks += 1

    meta = {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "seq_len": seq_len,
        "num_tokens": num_blocks * block_size,
        "num_real_tokens": num_real_tokens,
        "pad_id": pad_id,
        "dtype": "int32",
    }
    out_bin.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def prepare_data(cfg: BilalAIConfig) -> dict:
    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.txt")):
        raise FileNotFoundError(f"Ham veri bulunamadi: {RAW_DIR}. Once data/raw/*.txt ekleyin.")

    sources = []
    all_docs: list[str] = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        docs = extract_documents(cleaned)
        all_docs.extend(docs)
        sources.append({
            "file": path.name,
            "language": "tr",
            "license": "original-self-authored (CC0)",
            "num_documents": len(docs),
            "num_chars": len(cleaned),
        })

    all_docs, removed = dedup_documents(all_docs)
    splits = split_documents(all_docs, cfg.seed)

    for split, docs in splits.items():
        split_dir = DATA_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"{split}.txt").write_text("\n\n".join(docs), encoding="utf-8")

    manifest = {
        "model_name": cfg.model_name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "language": "tr",
        "license_note": "Tum seed verisi ozgun ve yazar tarafindan olusturulmustur (CC0).",
        "sources": sources,
        "total_documents": len(all_docs),
        "duplicates_removed": removed,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_ratio_note": "Kucuk veride %5/%5 dogrulama/test; buyuk veride %1/%1.",
        "seq_len": cfg.max_seq_len,
        "token_blocks": {},
    }

    tok_path = MODEL_DIR / "tokenizer_data"
    if (tok_path / "tokenizer.json").exists():
        tokenizer = ByteLevelBPETokenizer.load(tok_path)
        for split in ("train", "validation"):
            meta = build_token_blocks(
                splits[split], tokenizer, cfg.max_seq_len,
                PROC_DIR / f"{split}.bin", tokenizer.pad_id,
            )
            manifest["token_blocks"][split] = meta
        manifest["tokenizer_hash"] = get_tokenizer_hash(tok_path)
        print(f"[veri] token bloklari olusturuldu: {manifest['token_blocks']}")
    else:
        print("[veri] tokenizer henuz yok; metin bolmeleri yazildi. "
              "Once tokenizer --train, sonra tekrar --prepare-data calistirin.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[veri] dokuman sayisi: {len(all_docs)} (tekrar silinen: {removed})")
    print(f"[veri] bolmeler: {manifest['split_counts']}")
    print(f"[veri] manifest: {DATA_DIR / 'dataset_manifest.json'}")
    return manifest


# ====================================================================== #
# DATASET
# ====================================================================== #
class TokenBlockDataset(Dataset):
    def __init__(self, split: str, seq_len: int):
        bin_path = PROC_DIR / f"{split}.bin"
        meta_path = PROC_DIR / f"{split}.meta.json"
        if not bin_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Islenmis veri bulunamadi: {bin_path}. Once 'train.py --prepare-data' calistirin."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.seq_len = seq_len
        self.block_size = self.meta["block_size"]
        self.pad_id = self.meta["pad_id"]
        self.data = np.memmap(bin_path, dtype=np.int32, mode="r",
                              shape=(self.meta["num_blocks"], self.block_size))

    def __len__(self) -> int:
        return self.meta["num_blocks"]

    def __getitem__(self, idx: int):
        block = torch.from_numpy(np.array(self.data[idx], dtype=np.int64))
        input_ids = block[:self.seq_len]
        labels = block[1:self.seq_len + 1]
        return input_ids, labels


# ====================================================================== #
# EGITIM YARDIMCILARI
# ====================================================================== #
def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def autocast_ctx(device, precision: str):
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, precision, pad_id) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx(device, precision):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
        mask = labels != pad_id
        ntok = int(mask.sum().item())
        total_loss += float(loss.item())
        total_tokens += ntok
    model.train()
    return total_loss / max(1, total_tokens)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================== #
# EGITIM DONGUSU
# ====================================================================== #
def train(args) -> int:
    cfg = BilalAIConfig()
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.micro_batch is not None:
        cfg.micro_batch_size = args.micro_batch
    if args.grad_accum is not None:
        cfg.grad_accum_steps = args.grad_accum
    if args.precision is not None:
        cfg.precision = args.precision
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision = cfg.precision
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    if device.type == "cpu":
        precision = "fp32"

    tok_dir = MODEL_DIR / "tokenizer_data"
    if not (tok_dir / "tokenizer.json").exists():
        print("HATA: tokenizer bulunamadi. Once 'tokenizer.py --train' calistirin.")
        return 1
    tokenizer = ByteLevelBPETokenizer.load(tok_dir)
    tok_hash = get_tokenizer_hash(tok_dir)

    train_ds = TokenBlockDataset("train", cfg.max_seq_len)
    val_ds = TokenBlockDataset("validation", cfg.max_seq_len)
    if len(train_ds) == 0:
        print("HATA: egitim bloku yok. Veriyi hazirlayin.")
        return 1

    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.micro_batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_model(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
    )

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    if args.max_steps:
        total_steps = args.max_steps
        epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    else:
        epochs = args.epochs
        total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    use_scaler = (precision == "fp16" and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        if ckpt.get("tokenizer_hash") not in (tok_hash, "missing"):
            print("HATA: checkpoint tokenizer kimligi mevcut tokenizer ile eslesmiyor. "
                  "Uyumsuz checkpoint yuklenmedi.")
            return 1
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_validation_loss", float("inf"))
        restore_rng_states(ckpt.get("rng_states", {}))
        print(f"[resume] adim {start_step}, epoch {start_epoch}, best_val {best_val:.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "training_log.jsonl"
    manifest_path = DATA_DIR / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    print(f"[egitim] cihaz={device} precision={precision} "
          f"micro_batch={cfg.micro_batch_size} grad_accum={cfg.grad_accum_steps} "
          f"toplam_adim={total_steps} warmup={warmup_steps}")
    print(f"[egitim] train blok={len(train_ds)} validation blok={len(val_ds)}")

    global_step = start_step
    tokens_seen = 0
    t_start = time.time()
    accum_loss = 0.0
    micro_count = 0
    running_loss = 0.0
    running_count = 0
    stop = False

    def log_record(record: dict):
        record["elapsed_sec"] = round(time.time() - t_start, 2)
        record["tokens_seen"] = tokens_seen
        if device.type == "cuda":
            record["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, epochs):
            for input_ids, labels in train_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast_ctx(device, precision):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
                    loss_scaled = loss / cfg.grad_accum_steps

                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                accum_loss += float(loss.item())
                micro_count += 1
                tokens_seen += int(input_ids.numel())

                if micro_count % cfg.grad_accum_steps == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    step_loss = accum_loss / micro_count
                    running_loss += step_loss
                    running_count += 1
                    accum_loss = 0.0
                    micro_count = 0
                    lr_now = scheduler.get_last_lr()[0]

                    if global_step % args.log_interval == 0 or global_step == 1:
                        elapsed = time.time() - t_start
                        tps = tokens_seen / max(1e-6, elapsed)
                        print(f"[adim {global_step}/{total_steps}] loss={step_loss:.4f} "
                              f"lr={lr_now:.2e} tokens/s={tps:.0f}")
                        log_record({"type": "train", "step": global_step, "epoch": epoch,
                                    "train_loss": step_loss, "lr": lr_now,
                                    "tokens_per_sec": tps})

                    if global_step % args.eval_interval == 0 or global_step == total_steps:
                        val_loss = evaluate(model, val_loader, device, precision, tokenizer.pad_id)
                        print(f"[adim {global_step}] validation_loss={val_loss:.4f}")
                        log_record({"type": "validation", "step": global_step,
                                    "validation_loss": val_loss})
                        if val_loss < best_val:
                            best_val = val_loss
                            save_checkpoint(BEST_PATH, model, cfg, optimizer, scheduler,
                                            global_step, epoch, best_val, tok_hash,
                                            capture_rng_states(), manifest)
                            print(f"[checkpoint] yeni en iyi: {BEST_PATH} (val={best_val:.4f})")

                    if global_step % args.checkpoint_interval == 0:
                        save_checkpoint(LATEST_PATH, model, cfg, optimizer, scheduler,
                                        global_step, epoch, best_val, tok_hash,
                                        capture_rng_states(), manifest)

                    if global_step >= total_steps:
                        stop = True
                        break
            if stop:
                break
    except torch.cuda.OutOfMemoryError:
        print("\nHATA: CUDA bellek yetersiz (OOM). micro-batch veya grad-accum degerlerini "
              "degistirin; --micro-batch 1 deneyin. Sonsuz tekrar yapilmaz.")
        save_checkpoint(LATEST_PATH, model, cfg, optimizer, scheduler,
                        global_step, start_epoch, best_val, tok_hash,
                        capture_rng_states(), manifest)
        return 1

    # Son checkpoint ve ozet
    save_checkpoint(LATEST_PATH, model, cfg, optimizer, scheduler,
                    global_step, start_epoch, best_val, tok_hash,
                    capture_rng_states(), manifest)
    final_val = evaluate(model, val_loader, device, precision, tokenizer.pad_id)
    if final_val < best_val:
        best_val = final_val
        save_checkpoint(BEST_PATH, model, cfg, optimizer, scheduler,
                        global_step, start_epoch, best_val, tok_hash,
                        capture_rng_states(), manifest)

    elapsed = time.time() - t_start
    summary = {
        "model_name": cfg.model_name,
        "status": "egitim_suruyor" if args.max_steps else "egitim_tamamlandi",
        "global_step": global_step,
        "final_validation_loss": final_val,
        "best_validation_loss": best_val,
        "tokens_seen": tokens_seen,
        "elapsed_sec": round(elapsed, 2),
        "device": str(device),
        "precision": precision,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "dataset_manifest_summary": manifest.get("split_counts", {}),
    }
    (LOG_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[egitim] bitti. adim={global_step} final_val={final_val:.4f} best_val={best_val:.4f}")
    print(f"[egitim] ozet: {LOG_DIR / 'training_summary.json'}")
    return 0


# ====================================================================== #
def main() -> int:
    parser = argparse.ArgumentParser(description="BilalAI 1.0 egitim")
    parser.add_argument("--prepare-data", action="store_true", help="veri temizle, bol, bloklara ayir")
    parser.add_argument("--max-steps", type=int, default=None, help="kisa deneme icin adim siniri")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint yolu")
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    args = parser.parse_args()

    cfg = BilalAIConfig()
    if args.prepare_data:
        prepare_data(cfg)
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())