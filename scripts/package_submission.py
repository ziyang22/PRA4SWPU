"""Package a standalone submission; optionally collect the maintained reference model."""

import argparse
import zipfile
from pathlib import Path

REFERENCE_MODULES = ("data", "utils", "model", "ops")


def package(
    source: Path, archive: Path, reference: bool = False, checkpoint: Path | None = None
) -> None:
    """Bundle custom files, or inject reference dependencies without maintaining copies."""
    if checkpoint is not None and not reference:
        raise ValueError("--checkpoint requires --reference")
    if source.is_symlink():
        raise ValueError("Submission must not contain symlinks")
    for required in ("requirements.txt", "submission/main.py"):
        if not (source / required).is_file():
            raise ValueError(f"Missing {required}")
    paths = sorted(source.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Submission must not contain symlinks")
    members = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] != "submission" and relative.as_posix() != "requirements.txt":
            raise ValueError(f"Unexpected top-level file: {relative}")
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".npy"):
            raise ValueError(f"Remove generated files or data before packaging: {relative}")
        members[relative.as_posix()] = path
    if reference:
        root = Path(__file__).resolve().parents[1]
        injected = {
            **{f"submission/{name}.py": root / f"src/{name}.py" for name in REFERENCE_MODULES},
            "submission/weights/model.pt": checkpoint or root / "weights/baseline.pt",
            **{
                f"submission/csrc/{name}": root / f"src/csrc/{name}"
                for name in ("bindings.cpp", "conv2d.hip")
            },
        }
        for name, path in injected.items():
            if name in members:
                raise ValueError(f"Reference dependency conflicts with submitted file: {name}")
            if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise ValueError(f"Missing or linked reference dependency: {path}")
        members.update(injected)
    for name in members:
        if any(parent.as_posix() in members for parent in Path(name).parents):
            raise ValueError(f"Conflicting file and directory paths: {name}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, path in sorted(members.items()):
            zf.write(path, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("submission_template"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", action="store_true", help="Collect baseline modules and weight"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Reference weight to bundle instead of baseline"
    )
    args = parser.parse_args()
    package(args.source, args.output, args.reference, args.checkpoint)
