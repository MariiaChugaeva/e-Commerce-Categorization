import json
import re
import urllib.request
from typing import Any, List


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
        timeout: float = 180.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.8,
        num_predict: int = 512,
        format: Any = None,
        think: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if system is not None:
            payload["system"] = system
        if format is not None:
            payload["format"] = format

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["response"]


_PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["variants"],
}

_PARAPHRASE_SYSTEM = (
    "You rewrite e-commerce product titles. "
    "Each variant must apply a minor surface change to the original: different color, "
    "different size, different measurement, different quantity, or different material shade. "
    "Keep the product category identical. Do not add or remove words describing the product type. "
    "Each variant must be meaningfully different from the original and from the other variants."
)

_PARAPHRASE_PROMPT = (
    "Original title: {title}\n"
    "Return {n} variants as JSON matching the schema."
)


def _content_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z]{3,}", s.lower())}


def _is_paraphrase(variant: str, original: str, min_overlap: float = 0.5) -> bool:
    ov = _content_tokens(original)
    if not ov:
        return True
    vv = _content_tokens(variant)
    return len(ov & vv) / len(ov) >= min_overlap


def paraphrase_title(
    client: OllamaClient,
    title: str,
    n: int = 5,
    temperature: float = 0.9,
    min_overlap: float = 0.5,
) -> List[str]:
    raw = client.generate(
        _PARAPHRASE_PROMPT.format(title=title, n=n),
        system=_PARAPHRASE_SYSTEM,
        temperature=temperature,
        num_predict=128 * n,
        format=_PARAPHRASE_SCHEMA,
    )
    data = json.loads(raw)
    seen = {title.lower()}
    out: List[str] = []
    for v in data["variants"]:
        v = v.strip()
        key = v.lower()
        if not v or key in seen:
            continue
        if not _is_paraphrase(v, title, min_overlap):
            continue
        seen.add(key)
        out.append(v)
    return out[:n]
