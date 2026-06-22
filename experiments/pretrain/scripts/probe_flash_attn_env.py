import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def torch_major_minor() -> str:
    version = torch.__version__.split("+", 1)[0]
    return ".".join(version.split(".")[:2])


def cuda_major_tag() -> str | None:
    if not torch.version.cuda:
        return None
    return f"cu{torch.version.cuda.split('.', 1)[0]}"


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def top_tags(limit: int) -> list[str]:
    try:
        from packaging import tags
    except Exception:
        return []
    return [str(tag) for _, tag in zip(range(limit), tags.sys_tags())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the current environment for FlashAttention wheel compatibility.")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--tag_limit", type=int, default=12)
    args = parser.parse_args()

    cxx11_abi = bool(torch.compiled_with_cxx11_abi()) if hasattr(torch, "compiled_with_cxx11_abi") else None
    cuda_tag = cuda_major_tag()
    py_tag = python_tag()
    platform_tag = "linux_x86_64" if sys.platform.startswith("linux") and platform.machine() == "x86_64" else None
    candidate_wheel_prefix = None
    if cuda_tag and platform_tag:
        abi_text = "TRUE" if cxx11_abi else "FALSE"
        candidate_wheel_prefix = (
            f"flash_attn-<version>+{cuda_tag}torch{torch_major_minor()}"
            f"cxx11abi{abi_text}-{py_tag}-{py_tag}-{platform_tag}.whl"
        )

    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_major_minor": torch_major_minor(),
        "cuda_major_tag": cuda_tag,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "cxx11_abi": cxx11_abi,
        "python_tag": py_tag,
        "top_wheel_tags": top_tags(args.tag_limit),
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "candidate_wheel_name_pattern": candidate_wheel_prefix,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output_json:
        output = resolve(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
