"""BilalAI 1.0 - kendi kendine yeten byte-level BPE tokenizer.

Saf Python ile yazilmistir; ek kutuphane gerektirmez. Byte-level calistigi icin
Turkce karakterler (c, g, i, I, o, s, u) UTF-8 baytlari uzerinden islenir ve
hicbir asamada bozulmaz.

Sözlük tam 32768'dir:
  - 0..8   : ozel tokenlar
  - 9..264 : 256 tekil bayt
  - 265..  : BPE merge tokenlari
  - kalan  : <reserved_N> dolgu tokenlari (veri kucukse kullanilmaz, vocab sabit kalir)

Ozel token ID'leri sabittir ve special_tokens.json icinde kayitlidir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# --- Sabit ozel token ID'leri (sartname bolum 5) ---
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<user>": 4,
    "<assistant>": 5,
    "<system>": 6,
    "<tool>": 7,
    "<turn_end>": 8,
}
BYTE_OFFSET = 9          # ilk bayt tokeninin ID'si
NUM_BASE = BYTE_OFFSET + 256  # 265: merge tokenlari buradan baslar
TARGET_VOCAB_SIZE = 32768


class ByteLevelBPETokenizer:
    def __init__(self, vocab: list[bytes], merges: list[tuple[int, int]], reserved: int = 0):
        self.vocab = vocab                 # id -> bytes (ozel tokenlar dahil)
        self.merges = merges               # sirali (idA, idB) merge kurallari
        self.reserved = reserved
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.id_to_special = {v: k for k, v in SPECIAL_TOKENS.items()}
        # merge tablosu: (idA, idB) -> (oncelik, yeni_id)
        self.merge_ranks = {pair: (rank, NUM_BASE + rank) for rank, pair in enumerate(merges)}
        self._special_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        )
        # Kisayollar
        self.pad_id = SPECIAL_TOKENS["<pad>"]
        self.unk_id = SPECIAL_TOKENS["<unk>"]
        self.bos_id = SPECIAL_TOKENS["<bos>"]
        self.eos_id = SPECIAL_TOKENS["<eos>"]
        self.user_id = SPECIAL_TOKENS["<user>"]
        self.assistant_id = SPECIAL_TOKENS["<assistant>"]
        self.system_id = SPECIAL_TOKENS["<system>"]
        self.tool_id = SPECIAL_TOKENS["<tool>"]
        self.turn_end_id = SPECIAL_TOKENS["<turn_end>"]

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    def _encode_chunk(self, text: str) -> list[int]:
        ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
        if len(ids) < 2 or not self.merge_ranks:
            return ids
        while len(ids) >= 2:
            best_rank = None
            best_pair = None
            best_new = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                entry = self.merge_ranks.get(pair)
                if entry is not None and (best_rank is None or entry[0] < best_rank):
                    best_rank, best_new = entry
                    best_pair = pair
            if best_pair is None:
                break
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(best_new)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for part in self._special_pattern.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self._encode_chunk(part))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        result = []
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < BYTE_OFFSET:  # ozel token
                if buf:
                    result.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special_tokens:
                    result.append(self.id_to_special.get(i, "<unk>"))
            else:
                if i < len(self.vocab):
                    buf.extend(self.vocab[i])
        if buf:
            result.append(buf.decode("utf-8", errors="replace"))
        return "".join(result)

    # ------------------------------------------------------------------ #
    @classmethod
    def train(cls, texts, vocab_size: int = TARGET_VOCAB_SIZE, min_freq: int = 2,
              verbose: bool = True) -> "ByteLevelBPETokenizer":
        if vocab_size < NUM_BASE:
            raise ValueError(f"vocab_size en az {NUM_BASE} olmali.")

        # Baslangic sözlük: ozel tokenlar + 256 bayt
        vocab: list[bytes] = [t.encode("utf-8") for t in SPECIAL_TOKENS]
        vocab += [bytes([b]) for b in range(256)]
        merges: list[tuple[int, int]] = []

        # Kelime bazli frekans (bosluk sinirlarini korur; Turkce baytlari butun kalir)
        word_freq: Counter = Counter()
        for text in texts:
            for word in text.split():
                ids = tuple(BYTE_OFFSET + b for b in word.encode("utf-8"))
                if ids:
                    word_freq[ids] += 1

        max_merges = vocab_size - NUM_BASE
        if verbose:
            print(f"[tokenizer] {len(word_freq)} benzersiz kelime, hedef vocab {vocab_size}")

        while len(merges) < max_merges:
            pair_counts: Counter = Counter()
            for word, freq in word_freq.items():
                for a, b in zip(word, word[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_freq:
                break
            new_id = len(vocab)
            vocab.append(vocab[best_pair[0]] + vocab[best_pair[1]])
            merges.append(best_pair)

            # Tum kelimelerde bu cifti birlestir
            new_word_freq: Counter = Counter()
            a, b = best_pair
            for word, freq in word_freq.items():
                merged = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_word_freq[tuple(merged)] += freq
            word_freq = new_word_freq

            if verbose and len(merges) % 500 == 0:
                print(f"[tokenizer] {len(merges)} merge, son cift frekansi {best_freq}")

        # Vocab'i tam hedefe tamamla (kullanilmayan rezerv tokenlar)
        reserved = 0
        while len(vocab) < vocab_size:
            vocab.append(f"<reserved_{reserved}>".encode("utf-8"))
            reserved += 1

        if verbose:
            print(f"[tokenizer] egitim bitti: {len(merges)} merge, {reserved} rezerv, vocab {len(vocab)}")
        return cls(vocab, merges, reserved)

    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tokenizer_json = {
            "model": "byte_level_bpe",
            "vocab_size": self.vocab_size,
            "byte_offset": BYTE_OFFSET,
            "num_base": NUM_BASE,
            "reserved": self.reserved,
            "vocab_hex": [v.hex() for v in self.vocab],
            "merges": [list(m) for m in self.merges],
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(tokenizer_json, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "special_tokens.json").write_text(
            json.dumps(self.special_tokens, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "model_name": "BilalAI 1.0",
                    "tokenizer_class": "ByteLevelBPETokenizer",
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens,
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "clean_up_tokenization_spaces": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ByteLevelBPETokenizer":
        directory = Path(directory)
        tok_path = directory / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(
                f"Tokenizer dosyasi bulunamadi: {tok_path}. Once 'tokenizer.py --train' calistirin."
            )
        data = json.loads(tok_path.read_text(encoding="utf-8"))
        vocab = [bytes.fromhex(h) for h in data["vocab_hex"]]
        merges = [tuple(m) for m in data["merges"]]
        reserved = data.get("reserved", 0)
        tok = cls(vocab, merges, reserved)
        if tok.vocab_size != data["vocab_size"]:
            raise ValueError("Tokenizer vocab boyutu tutarsiz.")
        return tok


# ---------------------------------------------------------------------- #
def _read_training_texts(model_dir: Path) -> list[str]:
    """Tokenizer yalnizca train bolumunden egitilir (sartname bolum 5)."""
    train_dir = model_dir / "data" / "train"
    files = sorted(train_dir.glob("*.txt")) if train_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"Egitim metni bulunamadi: {train_dir}. Once 'train.py --prepare-data' calistirin."
        )
    texts = [f.read_text(encoding="utf-8") for f in files]
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description="BilalAI 1.0 byte-level BPE tokenizer")
    parser.add_argument("--train", action="store_true", help="data/train metinlerinden tokenizer egit")
    parser.add_argument("--vocab-size", type=int, default=TARGET_VOCAB_SIZE)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--self-test", action="store_true", help="kucuk bir round-trip testi")
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent
    out_dir = model_dir / "tokenizer_data"

    if args.self_test:
        sample = [
            "Merhaba dünya. Türkçe karakterler: ç ğ ı İ ö ş ü Ç Ğ Ö Ş Ü.",
            "Python'da liste nedir? Liste, sıralı değerleri saklar.",
            "istanbul türkiye'nin en kalabalık şehridir.",
        ] * 20
        tok = ByteLevelBPETokenizer.train(sample, verbose=False)
        text = "Merhaba dünya! Türkçe: ç ğ ı İ ö ş ü. <user> Selam <assistant> Merhaba <turn_end>"
        ids = tok.encode(text)
        back = tok.decode(ids, skip_special_tokens=False)
        assert back == text, f"Round-trip bozuk: {back!r}"
        assert tok.vocab_size == TARGET_VOCAB_SIZE
        print(f"[self-test] OK - vocab {tok.vocab_size}, round-trip korundu, Turkce saglam.")
        return 0

    if args.train:
        texts = _read_training_texts(model_dir)
        tok = ByteLevelBPETokenizer.train(texts, vocab_size=args.vocab_size, min_freq=args.min_freq)
        tok.save(out_dir)
        print(f"[tokenizer] kaydedildi: {out_dir}")
        # Dogrulama
        loaded = ByteLevelBPETokenizer.load(out_dir)
        probe = "Türkçe karakter testi: ç ğ ı İ ö ş ü."
        assert loaded.decode(loaded.encode(probe)) == probe
        print(f"[tokenizer] yukleme ve Turkce round-trip dogrulandi. vocab={loaded.vocab_size}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())