"""Sparse external memory (VRAM/sysmem) model."""

from __future__ import annotations

from ..errors import DeviceModelError


class SparseMemory:
    def __init__(self, *, default: int | None = None, name: str = "vram") -> None:
        self.name = name
        self.default = default  # None => unmapped fault on read
        self._bytes: dict[int, int] = {}

    def write(self, address: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self._bytes[address + i] = b

    def read(self, address: int, size: int) -> bytes:
        out = bytearray()
        for i in range(size):
            addr = address + i
            if addr in self._bytes:
                out.append(self._bytes[addr])
            elif self.default is not None:
                out.append(self.default & 0xFF)
            else:
                raise DeviceModelError(
                    f"unmapped {self.name} read",
                    details={"address": addr},
                )
        return bytes(out)

    def snapshot_hash(self) -> str:
        import hashlib

        items = sorted(self._bytes.items())
        h = hashlib.sha256()
        for addr, val in items:
            h.update(addr.to_bytes(8, "little"))
            h.update(bytes((val,)))
        return h.hexdigest()

    def region_bytes(self, address: int, size: int) -> bytes:
        return self.read(address, size)
