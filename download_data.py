"""Download publisher source data and the authors' plotting notebook."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
BASE = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs43588-025-00873-y/MediaObjects/"
)
SOURCES = {
    f"source_figure_{i - 1}.zip": BASE + f"43588_2025_873_MOESM{i}_ESM.zip"
    for i in range(3, 8)
}
SOURCES["authors_create_figures.ipynb"] = (
    "https://raw.githubusercontent.com/stefan-woerner/qamoo/"
    "8bddff3a7d49c18ed3c3d0bf0494e35965117131/notebooks/create_figures.ipynb"
)
manifest = []
for name, url in SOURCES.items():
    target = DATA / name
    if not target.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        payload = response.content
        if name.endswith(".zip"):
            with ZipFile(io.BytesIO(payload)) as archive:
                assert archive.testzip() is None
        else:
            assert "cells" in json.loads(payload)
        target.write_bytes(payload)
    payload = target.read_bytes()
    manifest.append({
        "file": name, "url": url, "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    })
    print(name, len(payload), flush=True)
    if name.endswith(".zip"):
        with ZipFile(io.BytesIO(payload)) as archive:
            print(archive.namelist()[:40], flush=True)
            for member in archive.infolist():
                if member.is_dir() or "__MACOSX" in member.filename:
                    continue
                dest = (DATA / name.removesuffix(".zip") / member.filename).resolve()
                assert dest.is_relative_to(DATA.resolve()), member.filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(member))
(DATA / "sources.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
