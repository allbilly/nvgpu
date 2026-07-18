"""Structured PCI expansion-ROM parsing for the pinned Palit GK104 image."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from .constants import (
  CODE_TYPE_EFI,
  CODE_TYPE_X86,
  EFI_CONTAINER_OFFSET,
  GK104_DEVICE_ID,
  LEGACY_CONTAINER_OFFSET,
  LEGACY_IMAGE_SIZE,
  NVIDIA_VENDOR_ID,
  PINNED_CONTAINER_SHA256,
  PINNED_LEGACY_SHA256,
)

DEFAULT_ROM_PATH = (
  Path(__file__).resolve().parent.parent
  / "examples_kepler"
  / "Palit.GTX770.4096.131216.rom"
)


class RomError(ValueError):
  """Malformed or identity-mismatched ROM container/image."""


@dataclass(frozen=True)
class PcirInfo:
  offset: int  # relative to image base
  vendor_id: int
  device_id: int
  pcir_length: int
  pcir_revision: int
  class_code: int
  image_length_units: int
  image_length: int
  code_revision: int
  code_type: int
  indicator: int

  @property
  def is_final_image(self) -> bool:
    return bool(self.indicator & 0x80)

  @property
  def is_x86(self) -> bool:
    return self.code_type == CODE_TYPE_X86

  @property
  def is_efi(self) -> bool:
    return self.code_type == CODE_TYPE_EFI


@dataclass(frozen=True)
class RomImage:
  """Immutable legacy option-ROM image and container metadata."""

  container: bytes
  container_sha256: str
  container_path: Optional[Path]
  image_offset: int
  data: bytes
  sha256: str
  checksum: int
  pcir: PcirInfo
  entry_offset: int  # within image; normally 3

  @property
  def size(self) -> int:
    return len(self.data)

  def byte_at(self, offset: int) -> int:
    return self.data[offset]


def sha256_hex(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _parse_pcir(image: bytes, pcir_off: int) -> PcirInfo:
  if pcir_off < 0 or pcir_off + 0x16 > len(image):
    raise RomError(f"PCIR pointer out of range: {pcir_off:#x}")
  if image[pcir_off:pcir_off + 4] != b"PCIR":
    raise RomError(f"PCIR signature missing at {pcir_off:#x}")
  vendor = int.from_bytes(image[pcir_off + 4:pcir_off + 6], "little")
  device = int.from_bytes(image[pcir_off + 6:pcir_off + 8], "little")
  pcir_len = int.from_bytes(image[pcir_off + 0x0A:pcir_off + 0x0C], "little")
  rev = image[pcir_off + 0x0C]
  class_code = int.from_bytes(image[pcir_off + 0x0D:pcir_off + 0x10], "little")
  units = int.from_bytes(image[pcir_off + 0x10:pcir_off + 0x12], "little")
  code_rev = int.from_bytes(image[pcir_off + 0x12:pcir_off + 0x14], "little")
  code_type = image[pcir_off + 0x14]
  indicator = image[pcir_off + 0x15]
  return PcirInfo(
    offset=pcir_off,
    vendor_id=vendor,
    device_id=device,
    pcir_length=pcir_len,
    pcir_revision=rev,
    class_code=class_code,
    image_length_units=units,
    image_length=units * 512,
    code_revision=code_rev,
    code_type=code_type,
    indicator=indicator,
  )


def _validate_image(image: bytes, *, expect_x86: bool = True) -> PcirInfo:
  if len(image) < 0x20:
    raise RomError("image too short")
  if image[0:2] != b"\x55\xaa":
    raise RomError("missing 55 AA ROM signature")
  size_field = image[2]
  if size_field * 512 != len(image):
    raise RomError(
      f"ROM size field {size_field} (*512={size_field * 512}) "
      f"!= image length {len(image)}"
    )
  pcir_off = int.from_bytes(image[0x18:0x1A], "little")
  pcir = _parse_pcir(image, pcir_off)
  if pcir.image_length != len(image):
    raise RomError(
      f"PCIR image length {pcir.image_length:#x} != data length {len(image):#x}"
    )
  if expect_x86 and not pcir.is_x86:
    raise RomError(f"expected x86 code type 0, got {pcir.code_type:#x}")
  if pcir.vendor_id != NVIDIA_VENDOR_ID or pcir.device_id != GK104_DEVICE_ID:
    raise RomError(
      f"PCI identity {pcir.vendor_id:04x}:{pcir.device_id:04x} "
      f"!= {NVIDIA_VENDOR_ID:04x}:{GK104_DEVICE_ID:04x}"
    )
  checksum = sum(image) & 0xFF
  if checksum != 0:
    raise RomError(f"legacy checksum {checksum:#x} != 0 (mod 256)")
  return pcir


def iter_pci_images(container: bytes) -> Iterable[Tuple[int, bytes, PcirInfo]]:
  """Walk PCI expansion-ROM images structurally (not via ambiguous 55 AA scans).

  Starts at offset 0 and advances by each PCIR image_length.  Stops after the
  image whose indicator has the last-image bit set, or when the next header
  would leave the container.
  """
  off = 0
  while off + 0x20 <= len(container):
    if container[off:off + 2] != b"\x55\xaa":
      # NVIDIA containers may pad before the first image (Palit: 0x600).
      # Skip forward in 512-byte strides looking for the next structural header.
      nxt = off + 512
      if nxt >= len(container):
        break
      off = nxt
      continue
    size_field = container[off + 2]
    img_len = size_field * 512
    if img_len <= 0 or off + img_len > len(container):
      raise RomError(f"truncated image at {off:#x}: length {img_len:#x}")
    image = container[off:off + img_len]
    pcir_off = int.from_bytes(image[0x18:0x1A], "little")
    if image[pcir_off:pcir_off + 4] != b"PCIR":
      raise RomError(f"no PCIR at structural image {off:#x}")
    pcir = _parse_pcir(image, pcir_off)
    yield off, image, pcir
    off += img_len
    if pcir.is_final_image:
      break


def select_legacy_image(
  container: bytes,
  *,
  prefer_offset: int = LEGACY_CONTAINER_OFFSET,
) -> Tuple[int, bytes, PcirInfo]:
  """Select the legacy (code type 0) image structurally.

  Prefers the known Palit offset when present; never falls back to an EFI image.
  """
  found: list[Tuple[int, bytes, PcirInfo]] = []
  for off, image, pcir in iter_pci_images(container):
    if pcir.is_x86 and pcir.vendor_id == NVIDIA_VENDOR_ID and pcir.device_id == GK104_DEVICE_ID:
      found.append((off, image, pcir))
  if not found:
    raise RomError("no legacy x86 NVIDIA GK104 image in container")
  for off, image, pcir in found:
    if off == prefer_offset:
      return off, image, pcir
  # Structural fallback: first matching x86 image (still never EFI).
  return found[0]


def load_rom(
  path: Optional[Path | str] = None,
  *,
  require_pinned_hashes: bool = True,
  container: Optional[bytes] = None,
) -> RomImage:
  """Load and validate the pinned Palit legacy ROM image."""
  container_path: Optional[Path]
  if container is None:
    container_path = Path(path) if path is not None else DEFAULT_ROM_PATH
    container = container_path.read_bytes()
  else:
    container_path = Path(path) if path is not None else None

  container_hash = sha256_hex(container)
  if require_pinned_hashes and container_hash != PINNED_CONTAINER_SHA256:
    raise RomError(
      f"container SHA-256 {container_hash} != pinned {PINNED_CONTAINER_SHA256}"
    )

  image_offset, image, _ = select_legacy_image(container)
  if image_offset != LEGACY_CONTAINER_OFFSET or len(image) != LEGACY_IMAGE_SIZE:
    # Still validate, but pin checks fail for non-canonical layouts.
    if require_pinned_hashes:
      raise RomError(
        f"legacy image at {image_offset:#x} size {len(image):#x}; "
        f"expected {LEGACY_CONTAINER_OFFSET:#x}/{LEGACY_IMAGE_SIZE:#x}"
      )

  pcir = _validate_image(image, expect_x86=True)
  image_hash = sha256_hex(image)
  if require_pinned_hashes and image_hash != PINNED_LEGACY_SHA256:
    raise RomError(
      f"legacy SHA-256 {image_hash} != pinned {PINNED_LEGACY_SHA256}"
    )

  # Guard: EFI image must not be selectable as legacy.
  if len(container) > EFI_CONTAINER_OFFSET + 0x20:
    efi_probe = container[EFI_CONTAINER_OFFSET:EFI_CONTAINER_OFFSET + 0x20]
    if efi_probe[0:2] == b"\x55\xaa":
      efi_pcir_off = int.from_bytes(efi_probe[0x18:0x1A], "little")
      # Only inspect; never return EFI.
      try:
        efi_units = container[EFI_CONTAINER_OFFSET + 2]
        efi_img = container[
          EFI_CONTAINER_OFFSET:EFI_CONTAINER_OFFSET + efi_units * 512
        ]
        efi_pcir = _parse_pcir(efi_img, efi_pcir_off)
        if efi_pcir.is_x86:
          raise RomError("EFI region unexpectedly reports x86 code type")
      except RomError:
        raise
      except Exception:
        pass

  return RomImage(
    container=bytes(container),
    container_sha256=container_hash,
    container_path=container_path,
    image_offset=image_offset,
    data=bytes(image),
    sha256=image_hash,
    checksum=sum(image) & 0xFF,
    pcir=pcir,
    entry_offset=0x0003,
  )


def reject_malformed_examples() -> Sequence[str]:
  """Return descriptions of malformed variants that must be rejected (for tests)."""
  return (
    "missing 55 AA",
    "bad checksum",
    "wrong vendor/device",
    "EFI code type selected as legacy",
    "truncated image length",
    "missing PCIR",
  )
