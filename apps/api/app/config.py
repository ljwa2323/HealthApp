from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _candidate_key_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    cwd = Path.cwd()
    return [
        Path(os.environ.get("NVIDIA_API_KEY_FILE", "")),
        cwd / "nvidia_key.txt",
        repo_root / "nvidia_key.txt",
        cwd.parent.parent / "nvidia_key.txt",
    ]


@lru_cache(maxsize=1)
def get_nvidia_api_key() -> str | None:
    env_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if env_key and not env_key.startswith("$"):
        return env_key

    for path in _candidate_key_paths():
        if not path or not path.is_file():
            continue
        value = path.read_text(encoding="utf-8").strip()
        if value and not value.startswith("$"):
            return value
    return None


OCR_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVCF_ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
MAX_INLINE_B64_CHARS = 180_000
