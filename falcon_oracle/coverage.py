"""Decode coverage reporting."""

from __future__ import annotations

from pathlib import Path

from .decoder import FalconDecoder, coverage_report
from .manifests import load_image_memory, load_instruction_manifest
from .values import parse_int


def report_corpus_coverage(corpus_dir: Path | str) -> dict:
    corpus_dir = Path(corpus_dir)
    manifest, imem, _ = load_image_memory(corpus_dir)
    decoder = FalconDecoder()
    decoded = decoder.decode_all(imem)
    report = coverage_report(imem, decoded)
    report["name"] = manifest.name

    expected = load_instruction_manifest(corpus_dir)
    mismatches = []
    if len(expected) != len(decoded):
        mismatches.append(
            f"instruction count {len(decoded)} != manifest {len(expected)}"
        )
    for exp, got in zip(expected, decoded):
        if parse_int(exp["pc"]) != got.pc:
            mismatches.append(f"pc {exp['pc']} != 0x{got.pc:x}")
            break
        if exp["raw"] != got.raw.hex():
            mismatches.append(
                f"raw at 0x{got.pc:x}: {got.raw.hex()} != {exp['raw']}"
            )
        if exp["mnemonic"] != got.mnemonic:
            # envydis prints mov for movw; accept alias
            if not (exp["mnemonic"] == "mov" and got.mnemonic == "mov"):
                mismatches.append(
                    f"mnemonic at 0x{got.pc:x}: {got.mnemonic} != {exp['mnemonic']}"
                )
        if exp.get("branch_target"):
            if got.branch_target != parse_int(exp["branch_target"]):
                mismatches.append(
                    f"branch at 0x{got.pc:x}: {got.branch_target} != {exp['branch_target']}"
                )
        if exp["size"] != got.size:
            mismatches.append(f"size at 0x{got.pc:x}: {got.size} != {exp['size']}")

    report["manifest_mismatches"] = mismatches
    report["manifest_ok"] = not mismatches and report["complete"]
    return report


def format_coverage(report: dict) -> str:
    lines = [
        report.get("name", "corpus"),
        f"  image bytes:                  {report['decoded_bytes']} / {report['image_bytes']} decoded",
        f"  instruction instances:         {report['instruction_instances']} / {report['instruction_instances']} decoded",
        f"  unique encoding forms:         {report['unique_form_count']} / {report['unique_form_count']} supported",
        f"  unknown opcodes:                {report['unknown_opcodes']}",
        f"  trailing undecoded bytes:        {report['trailing_undecoded_bytes']}",
    ]
    if report.get("manifest_mismatches"):
        lines.append("  manifest mismatches:")
        for m in report["manifest_mismatches"]:
            lines.append(f"    - {m}")
    return "\n".join(lines)
