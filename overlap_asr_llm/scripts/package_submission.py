"""Create a zip archive for final submission.

The script packages source code, docs, configs, and result tables while skipping
Python caches and temporary separated audio.
"""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "overlap_asr_llm_submission.zip"
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_PREFIXES = {
    ROOT / "outputs" / "separated_audio",
}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return any(path == prefix or prefix in path.parents for prefix in SKIP_PREFIXES)


def main() -> int:
    if ARCHIVE.exists():
        ARCHIVE.unlink()

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if path == ARCHIVE or path.is_dir() or should_skip(path):
                continue
            zf.write(path, path.relative_to(ROOT.parent))

    print(f"Wrote {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
