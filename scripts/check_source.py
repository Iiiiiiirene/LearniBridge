"""Check source entry points and keep generated/private files out of Git."""

import ast
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(r"/(?:home|Users|media/cfs|data/public)/|\bh\d{8}\b|\bea-[a-z0-9-]+-cpu-\d+\b")
SECRET = re.compile(r"sk-[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".sh"}
GENERATED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".mp4", ".pyc"}
PUBLIC_MEDIA = {
    "assets/results/image.png": 10 * 1024**2,
    "assets/results/video.mp4": 50 * 1024**2,
}


def main():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=False, capture_output=True,
    )
    if result.returncode == 0:
        paths = sorted({ROOT / name.decode() for name in result.stdout.split(b"\0") if name})
    else:
        paths = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    errors = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        public_media_limit = PUBLIC_MEDIA.get(relative.as_posix())
        if (path.is_symlink() or (path.suffix in GENERATED_SUFFIXES and public_media_limit is None)
                or any(part in (".validation", "__pycache__", ".venv", "cache", "runs") for part in relative.parts)
                or path.name == ".env"):
            errors.append(f"Generated/private file: {relative}")
            continue
        if path.stat().st_size > (public_media_limit or 2 * 1024**2):
            errors.append(f"Unexpected large source file: {relative}")
        if path.suffix in TEXT_SUFFIXES or path.name in ("NOTICE", "LICENSE"):
            content = path.read_text(encoding="utf-8")
            if PRIVATE.search(content) or SECRET.search(content):
                errors.append(f"Private path or potential credential: {relative}")
    module = ast.parse((ROOT / "run.py").read_text())
    entries = next(
        ast.literal_eval(node.value) for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ENTRIES" for target in node.targets)
    )
    for model, actions in entries.items():
        for action, relative in actions.items():
            if not (ROOT / model / relative).is_file():
                errors.append(f"Missing {model} {action} entry point: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Source checks passed ({len(paths)} files; all model entry points exist).")


if __name__ == "__main__":
    main()
