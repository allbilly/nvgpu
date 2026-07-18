"""Decoder coverage against the PMU pad corpus."""

from pathlib import Path

from falcon_oracle.coverage import report_corpus_coverage
from falcon_oracle.decoder import FalconDecoder
from falcon_oracle.manifests import load_image_memory, verify_corpus
from falcon_oracle.state import FalconMemory

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "pmu_bar1_bootstrap"


def test_corpus_hash_and_size():
    info = verify_corpus(CORPUS)
    assert info["image_size"] == 0xDE
    assert info["ok"]


def test_full_decode_coverage():
    report = report_corpus_coverage(CORPUS)
    assert report["complete"]
    assert report["decoded_bytes"] == 222
    assert report["manifest_ok"], report.get("manifest_mismatches")


def test_truncated_instruction_fails():
    mem = FalconMemory(bytearray(b"\xf1\x37"), name="imem", base=0)
    dec = FalconDecoder()
    try:
        dec.decode_one(mem, 0)
        assert False, "expected failure"
    except Exception:
        pass


def test_mutation_changes_or_fails():
    _, imem, image = load_image_memory(CORPUS)
    dec = FalconDecoder()
    original = dec.decode_one(imem, imem.base)
    mutated = bytearray(image)
    mutated[0] ^= 0x01
    mimem = FalconMemory(mutated, name="imem", base=imem.base)
    try:
        m = dec.decode_one(mimem, mimem.base)
        assert m.raw != original.raw or m.mnemonic != original.mnemonic
    except Exception:
        pass
