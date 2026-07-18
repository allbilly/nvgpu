"""Trace-guided Palit GTX 770 legacy option-ROM executor (stdlib only)."""

from .rom import RomImage, load_rom, DEFAULT_ROM_PATH, PINNED_CONTAINER_SHA256, PINNED_LEGACY_SHA256

__all__ = [
  "RomImage",
  "load_rom",
  "DEFAULT_ROM_PATH",
  "PINNED_CONTAINER_SHA256",
  "PINNED_LEGACY_SHA256",
]

__version__ = "0.1.0"
