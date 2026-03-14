from pathlib import Path


def test_no_internal_absolute_paths_in_package_sources():
    """Prevent leaking internal machine/server paths into shipped code."""
    root = Path(__file__).resolve().parents[3]
    pkg = root / "src" / "edgemap"

    forbidden_fragments = [
        "/Users/",
        "/home/",
        "/scratch/",
        "/mnt/",
        "gsmap_data/",
        "cafferychen",
    ]

    offenders: list[tuple[Path, str]] = []
    for path in pkg.rglob("*"):
        if not path.is_file():
            continue
        if "src/edgemap/tests/" in str(path):
            continue
        if path.suffix not in {".py", ".md", ".txt", ".toml", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for frag in forbidden_fragments:
            if frag in text:
                offenders.append((path.relative_to(root), frag))

    assert not offenders, f"Found internal path fragments: {offenders}"
