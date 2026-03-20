"""Generate managed-pipelines.json from pipeline metadata with managed: true."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from scripts.lib.discovery import get_repo_root

from .pipeline_description import extract_pipeline_description_from_file

OUTPUT_FILENAME = "managed-pipelines.json"
METADATA_FILENAME = "metadata.yaml"
PIPELINE_PY = "pipeline.py"


@dataclass(frozen=True)
class ManagedPipelineEntry:
    """One record in managed-pipelines.json."""

    name: str
    description: str
    path: str
    stability: str


def load_metadata(metadata_path: Path) -> dict | None:
    """Load and return metadata from a metadata.yaml file.

    Args:
        metadata_path: Path to metadata.yaml.

    Returns:
        Parsed metadata dict or None if file is missing or invalid.
    """
    if not metadata_path.is_file():
        return None
    with open(metadata_path) as f:
        return yaml.safe_load(f)


def discover_pipeline_dirs(pipelines_root: Path) -> list[Path]:
    """Discover all directories under pipelines/ that contain both metadata.yaml and pipeline.py.

    Args:
        pipelines_root: Path to the pipelines/ directory.

    Returns:
        List of paths to pipeline directories (each has metadata.yaml and pipeline.py).
    """
    result = []
    for meta_path in pipelines_root.rglob(METADATA_FILENAME):
        dir_path = meta_path.parent
        if (dir_path / PIPELINE_PY).is_file():
            result.append(dir_path)
    return sorted(result)


def collect_managed_pipelines(repo_root: Path) -> list[ManagedPipelineEntry]:
    """Collect all pipelines that have managed: true in their metadata.yaml.

    Args:
        repo_root: Repository root path.

    Returns:
        List of ``ManagedPipelineEntry`` records.
    """
    pipelines_root = repo_root / "pipelines"
    if not pipelines_root.is_dir():
        return []

    result = []
    for dir_path in discover_pipeline_dirs(pipelines_root):
        meta_path = dir_path / METADATA_FILENAME
        metadata = load_metadata(meta_path)
        if not metadata:
            continue
        if metadata.get("managed") is not True:
            continue

        # Relative path from repo root, with forward slashes for JSON
        rel_path = dir_path.relative_to(repo_root)
        path_str = f"{rel_path.as_posix()}/{PIPELINE_PY}"

        pipeline_py = dir_path / PIPELINE_PY
        from_decorator = extract_pipeline_description_from_file(
            pipeline_py,
            function_name=metadata.get("name"),
        )
        yaml_description = metadata.get("description")
        if isinstance(yaml_description, str) and yaml_description.strip():
            description = yaml_description.strip()
        else:
            description = from_decorator or ""

        result.append(
            ManagedPipelineEntry(
                name=metadata.get("name", ""),
                description=description,
                path=path_str,
                stability=metadata.get("stability", "alpha"),
            )
        )

    return result


def main() -> int:
    """CLI entry point. Generate managed-pipelines.json at repo root."""
    parser = argparse.ArgumentParser(
        description="Generate managed-pipelines.json from pipeline metadata (managed: true).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output file path (default: repo root / {OUTPUT_FILENAME})",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    output_path = args.output if args.output is not None else repo_root / OUTPUT_FILENAME

    pipelines = collect_managed_pipelines(repo_root)
    payload = [asdict(entry) for entry in pipelines]
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(pipelines)} pipeline(s) to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
