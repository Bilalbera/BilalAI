"""BilalAI 1.0 - terminal sohbeti / tek seferlik uretim.

Ornek:
  python "BilalAI 1.0/generate.py" --checkpoint "BilalAI 1.0/checkpoints/best/bilalai_1_0_best.pt"
  python "BilalAI 1.0/generate.py" --checkpoint ... --prompt "Merhaba, nasılsın?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inference import BilalAIInference
from chat_format import DEFAULT_SYSTEM

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = MODEL_DIR / "checkpoints" / "best" / "bilalai_1_0_best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="BilalAI 1.0 terminal uretim")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--tokenizer-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="tek seferlik uretim")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        engine = BilalAIInference(args.checkpoint, args.tokenizer_dir, args.device)
    except FileNotFoundError as e:
        print(f"HATA: {e}")
        print("Model henuz egitilmemis. Once egitim calistirin (train.py).")
        return 1
    except ValueError as e:
        print(f"HATA: {e}")
        return 1

    print(f"{engine.model_name} | cihaz={engine.device} | egitim adimi={engine.global_step}")
    if engine.global_step == 0:
        print("UYARI: Model henuz egitilmemis (rastgele agirlıklar). Ciktilar anlamli olmayabilir.")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
    )

    if args.prompt is not None:
        messages = [{"role": "system", "content": args.system},
                    {"role": "user", "content": args.prompt}]
        result = engine.generate(messages=messages, **gen_kwargs)
        print(f"\n{engine.model_name}: {result['text']}")
        print(f"[stop={result['stop_reason']} tokens={result['num_new_tokens']}]")
        return 0

    # Etkilesimli sohbet
    messages = [{"role": "system", "content": args.system}]
    print("Sohbet basladi. Cikmak icin /exit, gecmisi sifirlamak icin /clear yazin.\n")
    while True:
        try:
            user = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGorusmek uzere!")
            break
        if not user:
            continue
        if user == "/exit":
            print("Gorusmek uzere!")
            break
        if user == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("[gecmis sifirlandi]")
            continue
        messages.append({"role": "user", "content": user})
        result = engine.generate(messages=messages, **gen_kwargs)
        reply = result["text"] or "(bos cikti)"
        print(f"{engine.model_name}: {reply}")
        if result["truncated"]:
            print("[not] Baglam siniri nedeniyle eski mesajlar kirpildi.")
        messages.append({"role": "assistant", "content": reply})
    return 0


if __name__ == "__main__":
    sys.exit(main())