from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _safe_run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "runs" / "exp_meta.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make repo-root packages importable when this script is executed as `python tools/...`.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Lazy imports so this script can still run even if env is partially broken.
    import ultralytics  # type: ignore
    import torch  # type: ignore

    try:
        import torchvision  # type: ignore

        tv_version = torchvision.__version__
    except Exception as e:
        tv_version = f"ERR:{type(e).__name__}:{e}"

    yolov8_repo = repo_root / "method" / "yolov8"
    commit = "N/A"
    if (yolov8_repo / ".git").exists():
        commit = _safe_run(["git", "-C", str(yolov8_repo), "rev-parse", "HEAD"])

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append(f"time: {now}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"python: {sys.version.replace(chr(10), ' ')}")
    lines.append(f"ultralytics.__version__: {getattr(ultralytics, '__version__', 'N/A')}")
    lines.append(f"ultralytics.__file__: {getattr(ultralytics, '__file__', 'N/A')}")
    lines.append(f"ultralytics_git_commit: {commit}")
    lines.append(f"torch: {getattr(torch, '__version__', 'N/A')}")
    lines.append(f"torchvision: {tv_version}")
    lines.append(f"cuda_available: {torch.cuda.is_available()}")
    lines.append(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        try:
            lines.append(f"gpu_count: {torch.cuda.device_count()}")
            lines.append(f"gpu_name_0: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            lines.append(f"gpu_query_error: ERR:{type(e).__name__}:{e}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()


