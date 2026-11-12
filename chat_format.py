"""BilalAI 1.0 - sohbet formati.

Ozel tokenlar tokenizer ile aynidir (sartname bolum 10). Sistem, kullanici,
asistan ve arac mesajlari ayirt edilebilir; arac ciktisi kullanici gibi sunulmaz.
"""

from __future__ import annotations

DEFAULT_SYSTEM = "Sen BilalAI 1.0 adlı bir yapay zekâ asistanısın. Açık, doğru ve faydalı cevap ver."

ROLE_TAGS = {
    "system": "<system>",
    "user": "<user>",
    "assistant": "<assistant>",
    "tool": "<tool>",
}


def build_prompt(messages: list[dict], add_generation_prompt: bool = True,
                 default_system: str = DEFAULT_SYSTEM) -> str:
    """Mesaj listesini modelin egitildigi sohbet formatina cevirir."""
    parts: list[str] = []
    if not any(m.get("role") == "system" for m in messages):
        parts.append(f"<system>\n{default_system}\n")
    for m in messages:
        role = m.get("role", "user")
        tag = ROLE_TAGS.get(role, "<user>")
        content = m.get("content", "")
        parts.append(f"{tag}\n{content}\n")
        if role == "assistant":
            parts.append("<turn_end>\n")
    if add_generation_prompt:
        parts.append("<assistant>\n")
    return "".join(parts)


def clean_generated(text: str) -> str:
    """Uretilen metinden ozel tokenlari temizle; normal metni koru."""
    for tag in ROLE_TAGS.values():
        text = text.replace(tag, "")
    text = text.replace("<turn_end>", "").replace("<eos>", "").replace("<bos>", "")
    text = text.replace("<pad>", "").replace("<unk>", "")
    return text.strip()