"""BilalAI 1.0 - metin uretimi (inference).

Egitilmis checkpoint'i yukler ve otoregresif token uretir. Dis bir LLM API'sine
baglanmaz; uretim tamamen yerel model agirlıkları ile yapilir.

Sartname bolum 9: temperature, top_k, top_p, repetition_penalty, stop_token_ids,
context_length ayarlari; EOS/turn-end durdurma; 1024 siniri kontrollu yonetimi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config import BilalAIConfig
from model import build_model
from tokenizer import ByteLevelBPETokenizer
from checkpoint import load_checkpoint, get_tokenizer_hash
from chat_format import build_prompt, clean_generated

MODEL_DIR = Path(__file__).resolve().parent


def apply_repetition_penalty(logits: torch.Tensor, token_ids, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or len(token_ids) == 0:
        return logits
    for tid in set(int(t) for t in token_ids):
        if logits[tid] > 0:
            logits[tid] = logits[tid] / penalty
        else:
            logits[tid] = logits[tid] * penalty
    return logits


def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k)[0][-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # nucleus disindaki tokenlari ele (siniri asan ilk token korunur)
        remove = (cumulative - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    return logits


class BilalAIInference:
    def __init__(self, checkpoint_path: str | Path,
                 tokenizer_dir: str | Path | None = None,
                 device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else MODEL_DIR / "tokenizer_data"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint bulunamadi: {self.checkpoint_path}. Model henuz egitilmemis."
            )
        if not (self.tokenizer_dir / "tokenizer.json").exists():
            raise FileNotFoundError(f"Tokenizer bulunamadi: {self.tokenizer_dir}.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = ByteLevelBPETokenizer.load(self.tokenizer_dir)

        ckpt = load_checkpoint(self.checkpoint_path, map_location=self.device)
        cfg = BilalAIConfig.from_dict(ckpt.get("model_config", {}))
        self.config = cfg
        self.model_name = cfg.model_name

        ckpt_hash = ckpt.get("tokenizer_hash", "missing")
        cur_hash = get_tokenizer_hash(self.tokenizer_dir)
        self.tokenizer_match = ckpt_hash in ("missing", cur_hash)
        if not self.tokenizer_match:
            raise ValueError(
                "Checkpoint tokenizer kimligi mevcut tokenizer ile eslesmiyor. "
                "Uyumsuz checkpoint/tokenizer yuklenmedi."
            )

        self.model = build_model(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.global_step = ckpt.get("global_step", 0)
        self.stop_token_ids = {self.tokenizer.eos_id, self.tokenizer.turn_end_id, self.tokenizer.pad_id}
        self.ready = True

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        stop_token_ids: set[int] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        max_new_tokens = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = cfg.temperature if temperature is None else temperature
        top_k = cfg.top_k if top_k is None else top_k
        top_p = cfg.top_p if top_p is None else top_p
        repetition_penalty = cfg.repetition_penalty if repetition_penalty is None else repetition_penalty
        stops = self.stop_token_ids if stop_token_ids is None else set(stop_token_ids)

        if seed is not None:
            torch.manual_seed(seed)

        if messages is not None:
            prompt = build_prompt(messages, add_generation_prompt=True)
        if prompt is None:
            raise ValueError("prompt veya messages verilmeli.")

        ids = self.tokenizer.encode(prompt)
        ids = [self.tokenizer.bos_id] + ids
        truncated = False
        # Girdi + uretim max_seq_len'i asmasin; asarsa bastan kontrollu kirp.
        max_input = max(1, cfg.max_seq_len - max_new_tokens)
        if len(ids) > max_input:
            ids = ids[-max_input:]
            truncated = True

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        stop_reason = "max_new_tokens"

        for _ in range(max_new_tokens):
            if input_ids.size(1) >= cfg.max_seq_len:
                stop_reason = "context_limit"
                break
            logits = self.model(input_ids)
            next_logits = logits[0, -1, :].float().clone()
            next_logits = apply_repetition_penalty(next_logits, generated, repetition_penalty)

            if temperature <= 0:
                next_token = int(next_logits.argmax().item())
            else:
                next_logits = next_logits / temperature
                next_logits = top_k_top_p_filter(next_logits, top_k, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())

            if next_token in stops:
                stop_reason = "eos" if next_token == self.tokenizer.eos_id else "turn_end"
                break

            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.long, device=self.device)],
                dim=1,
            )

        text = clean_generated(self.tokenizer.decode(generated, skip_special_tokens=True))
        return {
            "text": text,
            "token_ids": generated,
            "num_new_tokens": len(generated),
            "stop_reason": stop_reason,
            "truncated": truncated,
            "model_name": self.model_name,
            "global_step": self.global_step,
        }