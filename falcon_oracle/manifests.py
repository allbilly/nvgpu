"""Corpus manifest loading and verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import FalconOracleError
from .state import FalconMemory
from .values import parse_int


@dataclass
class CorpusManifest:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def base_address(self) -> int:
        return parse_int(self.data["base_address"])

    @property
    def entry_address(self) -> int:
        return parse_int(self.data["entry_address"])

    @property
    def image_size(self) -> int:
        return int(self.data["image_size"])

    @property
    def sha256(self) -> str:
        return self.data["sha256"]

    @property
    def dmem_size(self) -> int:
        return int(self.data.get("dmem_size", 4096))

    @property
    def done_pc(self) -> int | None:
        v = self.data.get("done_pc")
        return parse_int(v) if v is not None else None

    @property
    def done_magic_addr(self) -> int | None:
        v = self.data.get("done_magic_dmem")
        return parse_int(v) if v is not None else None

    @property
    def done_magic_value(self) -> int | None:
        v = self.data.get("done_magic_value")
        return parse_int(v) if v is not None else None


def load_manifest(corpus_dir: Path | str) -> CorpusManifest:
    corpus_dir = Path(corpus_dir)
    path = corpus_dir / "image.manifest.json"
    data = json.loads(path.read_text())
    return CorpusManifest(path=path, data=data)


def verify_corpus(corpus_dir: Path | str) -> dict[str, Any]:
    corpus_dir = Path(corpus_dir)
    manifest = load_manifest(corpus_dir)
    image = (corpus_dir / "image.bin").read_bytes()
    digest = hashlib.sha256(image).hexdigest()
    errors: list[str] = []
    if len(image) != manifest.image_size:
        errors.append(f"size mismatch: {len(image)} != {manifest.image_size}")
    if digest != manifest.sha256:
        errors.append(f"sha256 mismatch: {digest} != {manifest.sha256}")
    if not (manifest.base_address <= manifest.entry_address < manifest.base_address + len(image)):
        errors.append("entry outside image")
    if errors:
        raise FalconOracleError("corpus verification failed", details={"errors": errors})
    return {
        "name": manifest.name,
        "sha256": digest,
        "image_size": len(image),
        "ok": True,
    }


def load_image_memory(corpus_dir: Path | str) -> tuple[CorpusManifest, FalconMemory, bytes]:
    corpus_dir = Path(corpus_dir)
    manifest = load_manifest(corpus_dir)
    image = (corpus_dir / "image.bin").read_bytes()
    verify_corpus(corpus_dir)
    imem = FalconMemory(bytearray(image), name="imem", base=manifest.base_address)
    return manifest, imem, image


def load_symbols(corpus_dir: Path | str) -> dict[int, str]:
    path = Path(corpus_dir) / "symbols.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[int, str] = {}
    for key, meta in data.items():
        out[parse_int(key)] = meta.get("symbol") or key
    return out


def load_instruction_manifest(corpus_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(corpus_dir) / "instructions.json"
    data = json.loads(path.read_text())
    return data["instructions"]


def load_initial_dmem(corpus_dir: Path | str, size: int) -> bytearray:
    path = Path(corpus_dir) / "initial_dmem.bin"
    if path.exists():
        data = bytearray(path.read_bytes())
        if len(data) < size:
            data.extend(b"\x00" * (size - len(data)))
        return data[:size]
    return bytearray(size)
