"""Package code, processed data and reports; verify all archived bytes."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "eeg_bci_pipeline_handoff"


def validate_data():
    """Check all 18 subjects and 36 arrays against metadata before packaging."""
    rows = []
    for dataset, channels, classes in [("2a", 22, 4), ("2b", 3, 2)]:
        for subject in range(1, 10):
            folder = ROOT / "processed" / f"BCICIV_{dataset}" / f"subject_{subject:02d}"
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            assert meta["source"] == "MOABB", folder
            assert meta["split"]["strategy"] == "official_session", folder
            assert meta["preprocessing"]["standardization"]["fit_on"] == "train_only", folder
            assert len(meta["channels"]) == channels, folder
            for split in ("train", "test"):
                with np.load(folder / f"{split}.npz", allow_pickle=False) as data:
                    x, y = data["X"], data["y"]
                    assert x.ndim == 3 and x.shape[1:] == (channels, 1000), folder
                    assert y.shape == (len(x),) and len(x) > 0, folder
                    assert np.isfinite(x).all(), folder
                    assert np.issubdtype(y.dtype, np.integer), folder
                    assert np.isin(y, np.arange(classes)).all(), folder
                    assert list(x.shape) == meta[f"X_{split}_shape"], folder
                    rows.append(dict(dataset=dataset, subject=subject, split=split,
                                     X_shape=list(x.shape), y_shape=list(y.shape)))
    return rows


def selected_files():
    """Include delivery files without environments, caches or preview drafts."""
    pdf = ROOT / "pipeline_summary.pdf"
    if not pdf.exists():
        pdf = ROOT / "reports/pipeline_summary.pdf"
    files = [ROOT / "README.md", ROOT / ".gitignore", ROOT / "requirements.txt", ROOT / "requirements-models.txt",
             pdf, ROOT / "reports/dataset_summary.csv"]
    for directory in ("configs", "datasets", "preprocessing", "splits", "loaders",
                      "models", "scripts", "tests", "processed", "reports/quality", "reports/integration"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    return sorted(files)


def main():
    """Create a timestamped ZIP and SHA256 sidecar without replacing past releases."""
    rows = validate_data()
    print(f"Validated {len(rows)} train/test files across 18 subjects.", flush=True)
    output = ROOT / "handoff"
    output.mkdir(exist_ok=True)
    archive = output / f"{PREFIX}_{datetime.now():%Y%m%d_%H%M%S}.zip"
    hashes = {}
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for directory in ("configs", "datasets", "preprocessing", "splits", "loaders", "scripts", "tests"):
            bundle.writestr(f"{PREFIX}/{directory}/", b"")
        for path in selected_files():
            relative = path.relative_to(ROOT).as_posix()
            if relative == "pipeline_summary.pdf":
                relative = "reports/pipeline_summary.pdf"
            content = path.read_bytes()
            hashes[relative] = hashlib.sha256(content).hexdigest()
            bundle.writestr(f"{PREFIX}/{relative}", content)
        info = {"python": platform.python_version(),
                "packages": {name: version(name) for name in
                             ("numpy", "torch", "mne", "moabb", "matplotlib", "pytest")},
                "validation": rows,
                "note": "Recorded producer versions; not a cross-platform lockfile."}
        try:
            info["packages"]["braindecode"] = version("braindecode")
        except PackageNotFoundError:
            pass
        bundle.writestr(f"{PREFIX}/handoff_info.json", json.dumps(info, indent=2))
        bundle.writestr(f"{PREFIX}/SHA256SUMS.json", json.dumps(hashes, indent=2))
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None, "ZIP integrity check failed"
        for relative, expected in hashes.items():
            actual = hashlib.sha256(bundle.read(f"{PREFIX}/{relative}")).hexdigest()
            assert actual == expected, relative
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Verified {len(hashes)} source files byte-for-byte.")
    print(f"Archive: {archive}")
    print(f"Size: {archive.stat().st_size / 1024 ** 2:.2f} MiB")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
