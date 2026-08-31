#!/usr/bin/env python3
"""Disassemble the GK104 FECS falcon (fuc3) firmware.

Focus: the ctx_xfer_idle loop the FECS spins in at PC 0xa15-0xa26 while
waiting for the graphics engine to go idle (bit 0x2000 of MMIO 0x409c00).

Encoding: https://envytools.readthedocs.io/en/latest/hw/falcon/isa.html
  - Instructions are variable-length: 2, 3, or 4 bytes.
  - First byte high 2 bits = operand size (00=8b,01=16b,10=32b,11=unsized).
  - Labels/offsets in hubgk104.fuc3.h are falcon $pc = byte offset into the
    binary (confirmed: the binary matches the .h u32 array exactly, and the
    `bra #init` at offset 0 targets label 0x039b).

Usage: python3 disasm_fecs.py [start_hex [end_hex]]
"""
import re, struct, sys

FW   = "firmware/gk104/gk104_fecs_code.bin"
HDR  = "ref/linux/drivers/gpu/drm/nouveau/nvkm/engine/gr/fuc/hubgk104.fuc3.h"
SRC  = "ref/linux/drivers/gpu/drm/nouveau/nvkm/engine/gr/fuc/hub.fuc"

REG = [f"$r{i}" for i in range(16)]
SZ  = {0: "b8", 1: "b16", 2: "b32"}

# branch condition codes (low 6 bits of byte1 for f4/f5 bra)
CC = {**{i: f"p{i}" for i in range(8)}, 0x08: "c", 0x09: "o", 0x0a: "s",
      0x0b: "z", 0x0c: "a", 0x0d: "na", 0x0e: "",
      **{0x10+i: f"np{i}" for i in range(8)}, 0x18: "nc", 0x19: "no",
      0x1a: "ns", 0x1b: "nz", 0x1c: "g", 0x1d: "le", 0x1e: "l", 0x1f: "ge"}

# (format, subopcode) -> mnemonic, sized ops
S = {('0',0):'st',
     ('1',0):'add',('1',1):'adc',('1',2):'sub',('1',3):'sbb',('1',4):'shl',
     ('1',5):'shr',('1',7):'sar',('1',8):'ld',('1',0xc):'shlc',('1',0xd):'shrc',
     ('2',0):'add',('2',1):'adc',('2',2):'sub',('2',3):'sbb',
     ('30',1):'st',('30',4):'cmpu',('30',5):'cmps',('30',6):'cmp',
     ('31',4):'cmpu',('31',5):'cmps',('31',6):'cmp',
     ('34',0):'ld',
     ('36',0):'add',('36',1):'adc',('36',2):'sub',('36',3):'sbb',('36',4):'shl',
     ('36',5):'shr',('36',7):'sar',('36',0xc):'shlc',('36',0xd):'shrc',
     ('37',0):'add',('37',1):'adc',('37',2):'sub',('37',3):'sbb',
     ('38',0):'st',('38',4):'cmpu',('38',5):'cmps',('38',6):'cmp',
     ('3a',0):'not',('3a',1):'neg',('3a',2):'mov',('3a',3):'hswap',
     ('3b',0):'add',('3b',1):'adc',('3b',2):'sub',('3b',3):'sbb',('3b',4):'shl',
     ('3b',5):'shr',('3b',7):'sar',('3b',0xc):'shlc',('3b',0xd):'shrc',
     ('3c',0):'add',('3c',1):'adc',('3c',2):'sub',('3c',3):'sbb',('3c',4):'shl',
     ('3c',5):'shr',('3c',7):'sar',('3c',8):'ld',('3c',0xc):'shlc',('3c',0xd):'shrc',
     ('3d',0):'not',('3d',1):'neg',('3d',2):'mov',('3d',3):'hswap',
     ('3d',4):'clear',('3d',5):'setf'}

# (format, subopcode) -> mnemonic, unsized ops
U = {('cx',0):'mulu',('cx',1):'muls',('cx',2):'sext',('cx',3):'extrs',
     ('cx',4):'and',('cx',5):'or',('cx',6):'xor',('cx',7):'extr',('cx',8):'xbit',
     ('cx',0xb):'ins',('cx',0xc):'div',('cx',0xd):'mod',('cx',0xf):'iord',
     ('dx',0):'iowr',('dx',1):'iowrs',
     ('ex',0):'mulu',('ex',1):'muls',('ex',3):'extrs',('ex',4):'and',('ex',5):'or',
     ('ex',6):'xor',('ex',7):'extr',('ex',0xb):'ins',('ex',0xc):'div',('ex',0xd):'mod',
     ('f0',0):'mulu',('f0',1):'muls',('f0',2):'sext',('f0',3):'sethi',('f0',4):'and',
     ('f0',5):'or',('f0',6):'xor',('f0',7):'mov',('f0',9):'bset',('f0',0xa):'bclr',
     ('f0',0xb):'btgl',('f0',0xc):'xbit',
     ('f1',0):'mulu',('f1',1):'muls',('f1',3):'sethi',('f1',4):'and',('f1',5):'or',
     ('f1',6):'xor',('f1',7):'mov',
     ('f2',8):'setp',('f2',0xc):'ccmd',
     ('f8',0):'ret',('f8',1):'iret',('f8',2):'exit',('f8',3):'xdwait',('f8',6):'???',
     ('f8',7):'xcwait',('f8',8):'trap0',('f8',9):'trap1',('f8',0xa):'trap2',('f8',0xb):'trap3',
     ('f9',0):'push',('f9',1):'addsp',('f9',4):'jmp',('f9',5):'call',('f9',8):'itlb',
     ('f9',9):'bset',('f9',0xa):'bclr',('f9',0xb):'btgl',
     ('fa',0):'iowr',('fa',1):'iowrs',('fa',4):'xcld',('fa',5):'xdld',('fa',6):'xdst',('fa',8):'setp',
     ('fc',0):'pop',
     ('fd',0):'mov>sr',('fd',1):'mov<sr',('fd',2):'ptlb',('fd',3):'vtlb',('fd',4):'and',
     ('fd',5):'or',('fd',6):'xor',('fd',9):'bset',('fd',0xa):'bclr',('fd',0xb):'btgl',('fd',0xc):'xbit',
     ('ff',0):'mulu',('ff',1):'muls',('ff',2):'sext',('ff',3):'extrs',('ff',4):'and',
     ('ff',5):'or',('ff',6):'xor',('ff',7):'extr',('ff',8):'xbit',('ff',0xc):'div',
     ('ff',0xd):'mod',('ff',0xf):'iord'}

def parse_labels(path):
    """Return {byte_offset: label_name} from the .h header comments."""
    txt = open(path).read().split("gk104_grhub_code[]")[1]
    out = {}
    for m in re.finditer(r'/\*\s*0x([0-9a-f]+):\s*(.*?)\s*\*/', txt):
        out[int(m.group(1), 16)] = m.group(2)
    return out

def s8(v):  return v - 256 if v & 0x80 else v
def s16(v): return v - 0x10000 if v & 0x8000 else v

def decode(buf, pc):
    """Decode one instruction at byte offset pc. Returns (length, text)."""
    b0 = buf[pc]
    if b0 < 0xc0:  # sized
        sz = b0 >> 6; fmt = b0 & 0x3f
        suf = SZ[sz]
        if fmt < 0x10:    # 0x: O1 R2S R1S I8
            o1 = fmt & 0xf; r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf; i8 = buf[pc+2]
            mn = S.get(('0', o1), f".op0{o1:x}")
            return 3, f"{mn} {suf} {REG[r2]}, {REG[r1]}, 0x{i8:02x}"
        if fmt < 0x20:    # 1x: O1 R1D R2S I8
            o1 = fmt & 0xf; r1 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i8 = buf[pc+2]
            mn = S.get(('1', o1), f".op1{o1:x}")
            return 3, f"{mn} {suf} {REG[r1]}, {REG[r2]}, 0x{i8:02x}"
        if fmt < 0x30:    # 2x: O1 R1D R2S I16
            o1 = fmt & 0xf; r1 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4
            i16 = buf[pc+2] | (buf[pc+3]<<8)
            mn = S.get(('2', o1), f".op2{o1:x}")
            return 4, f"{mn} {suf} {REG[r1]}, {REG[r2]}, 0x{i16:04x}"
        fl = f"{fmt:02x}"
        if fl == '30':
            o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i8 = buf[pc+2]
            return 3, f"{S.get(('30',o2),'?')} {suf} {REG[r2]}, 0x{i8:02x}"
        if fl == '31':
            o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i16 = buf[pc+2]|(buf[pc+3]<<8)
            return 4, f"{S.get(('31',o2),'?')} {suf} {REG[r2]}, 0x{i16:04x}"
        if fl == '34':
            o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i8 = buf[pc+2]
            return 3, f"{S.get(('34',o2),'?')} {suf} {REG[r2]}, 0x{i8:02x}"
        if fl == '36':
            o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i8 = buf[pc+2]
            return 3, f"{S.get(('36',o2),'?')} {suf} {REG[r2]}, 0x{i8:02x}"
        if fl == '37':
            o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i16 = buf[pc+2]|(buf[pc+3]<<8)
            return 4, f"{S.get(('37',o2),'?')} {suf} {REG[r2]}, 0x{i16:04x}"
        if fl in ('38','39','3a','3b'):
            r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf; o3 = buf[pc+2]&0xf; r3 = buf[pc+2]>>4
            mn = S.get((fl, o3), f".op{fl}{o3:x}")
            if fl == '38': return 3, f"{mn} {suf} {REG[r2]}, {REG[r1]}"
            if fl == '39': return 3, f"{mn} {suf} {REG[r1]}, {REG[r2]}"
            if fl == '3a': return 3, f"{mn} {suf} {REG[r2]}, {REG[r1]}"
            return 3, f"{mn} {suf} {REG[r2]}, {REG[r1]}"
        if fl == '3c':
            r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf; o3 = buf[pc+2]&0xf; r3 = buf[pc+2]>>4
            mn = S.get(('3c', o3), f".op3c{o3:x}")
            return 3, f"{mn} {suf} {REG[r3]}, {REG[r1]}, {REG[r2]}"
        if fl == '3d':
            o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4
            return 2, f"{S.get(('3d',o2),'?')} {suf} {REG[r2]}"
        return 1, f".byte 0x{b0:02x}"
    # unsized
    hi = f"{b0:02x}"
    if 0xc0 <= b0 <= 0xcf:   # cx: O1 R1D R2S I8
        o1 = b0 & 0xf; r1 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i8 = buf[pc+2]
        mn = U.get(('cx', o1), f".cx{o1:x}")
        if mn == 'iord': return 3, f"iord {REG[r1]}, I[{REG[r2]}]"
        return 3, f"{mn} {REG[r1]}, {REG[r2]}, 0x{i8:02x}"
    if 0xd0 <= b0 <= 0xdf:   # dx: O1 R2S R1S I8
        o1 = b0 & 0xf; r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf; i8 = buf[pc+2]
        mn = U.get(('dx', o1), f".dx{o1:x}")
        if mn in ('iowr','iowrs'): return 3, f"{mn} I[{REG[r2]}], {REG[r1]}"
        return 3, f"{mn} {REG[r2]}, {REG[r1]}, 0x{i8:02x}"
    if 0xe0 <= b0 <= 0xef:   # ex: O1 R1D R2S I16
        o1 = b0 & 0xf; r1 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i16 = buf[pc+2]|(buf[pc+3]<<8)
        return 4, f"{U.get(('ex',o1),'?')} {REG[r1]}, {REG[r2]}, 0x{i16:04x}"
    if b0 in (0xf0, 0xf1):   # O2 R2SD I8/I16
        o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4
        if b0 == 0xf0:
            i8 = buf[pc+2]; mn = U.get(('f0', o2), f".f0{o2:x}")
            if mn == 'sethi': return 3, f"sethi {REG[r2]}, 0x{i8:02x}"
            return 3, f"{mn} {REG[r2]}, 0x{i8:02x}"
        i16 = buf[pc+2]|(buf[pc+3]<<8); mn = U.get(('f1', o2), f".f1{o2:x}")
        return 4, f"{mn} {REG[r2]}, 0x{i16:04x}"
    if b0 == 0xf2:           # O2 R2S I8
        o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4; i8 = buf[pc+2]
        return 3, f"{U.get(('f2',o2),'?')} {REG[r2]}, 0x{i8:02x}"
    if b0 in (0xf4, 0xf5):   # OL I8 / OL I16  (bra/jmp/call/sleep/...)
        ol = buf[pc+1]&0x3f
        if b0 == 0xf4:
            imm = s8(buf[pc+2]); raw = buf[pc+2]; length = 3
        else:
            raw = buf[pc+2]|(buf[pc+3]<<8); imm = s16(raw); length = 4
        if ol <= 0x1f:        # bra (relative, sign-extended)
            tgt = (pc + imm) & 0xffffffff
            cc = CC.get(ol, f"?{ol:02x}")
            return length, f"bra {cc} 0x{tgt:x}" if cc else f"bra 0x{tgt:x}"
        if ol == 0x20:        # jmp (absolute, zero-extended)
            return length, f"jmp 0x{raw:x}"
        if ol == 0x21:        # call (absolute)
            return length, f"call 0x{raw:x}"
        if ol == 0x28:        # sleep
            return length, f"sleep 0x{raw:x}"
        if ol == 0x30:        # add [sp]
            return length, f"addsp 0x{imm:x}"
        if ol in (0x31,0x32,0x33):
            bit = {0x31:'bset',0x32:'bclr',0x33:'btgl'}[ol]
            return length, f"{bit} $flags, 0x{raw:x}"
        if ol == 0x3c:
            return length, f"ccmd 0x{raw:x}"
        return length, f".f{b0&1:x}_{ol:02x} 0x{raw:x}"
    if b0 == 0xf8:           # O2 (no operands, 2 bytes)
        o2 = buf[pc+1]&0xf
        return 2, U.get(('f8', o2), f".f8{o2:x}")
    if b0 == 0xf9:           # O2 R2S (2 bytes)
        o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4
        return 2, f"{U.get(('f9',o2),'?')} {REG[r2]}"
    if b0 == 0xfa:           # O3 R2S R1S
        o3 = buf[pc+2]&0xf; r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf
        return 3, f"{U.get(('fa',o3),'?')} {REG[r2]}, {REG[r1]}"
    if b0 == 0xfc:           # O2 R2D (2 bytes)
        o2 = buf[pc+1]&0xf; r2 = buf[pc+1]>>4
        return 2, f"{U.get(('fc',o2),'?')} {REG[r2]}"
    if b0 == 0xfd:           # O3 R2SD R1S
        o3 = buf[pc+2]&0xf; r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf
        return 3, f"{U.get(('fd',o3),'?')} {REG[r2]}, {REG[r1]}"
    if b0 == 0xff:           # O3 R3D R2S R1S
        o3 = buf[pc+2]&0xf; r3 = buf[pc+2]>>4; r2 = buf[pc+1]>>4; r1 = buf[pc+1]&0xf
        mn = U.get(('ff', o3), f".ff{o3:x}")
        if mn == 'iord': return 3, f"iord {REG[r3]}, I[{REG[r2]}]"
        return 3, f"{mn} {REG[r3]}, {REG[r1]}, {REG[r2]}"
    return 1, f".byte 0x{b0:02x}"

def main():
    start = int(sys.argv[1], 16) if len(sys.argv) > 1 else 0xa00
    end   = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0xa40
    buf = open(FW, "rb").read()
    raw_len = len(buf)
    buf += b"\x00\x00\x00\x00"          # pad so decode never reads OOB
    labels = parse_labels(HDR)
    last = max(labels)                   # last real label = ctx_xfer_done
    # decode sequentially from 0 so instruction boundaries are correct
    pc = 0
    insns = {}
    while pc < last + 4:                 # stop at end of real code (before zero pad)
        length, text = decode(buf, pc)
        insns[pc] = (length, text)
        pc += length
    print(f"GK104 FECS firmware: {len(buf)} bytes, {len(insns)} instructions")
    print(f"Disassembly 0x{start:x}-0x{end:x}  (PC values seen: 0xa15 0xa1c 0xa1f 0xa23 0xa26)")
    print("=" * 72)
    for off in sorted(insns):
        length, text = insns[off]
        if off + length <= start or off >= end:
            continue
        lbl = labels.get(off)
        raw = " ".join(f"{b:02x}" for b in buf[off:off+length])
        marker = " <==" if off in (0xa15, 0xa1c, 0xa1f, 0xa23, 0xa26) else ""
        if lbl:
            print(f"\n{off:#06x} <{lbl}>:")
        print(f"  {off:#06x}: {raw:<16} {text}{marker}")
    # show the matching source
    print("\n" + "=" * 72)
    print("Source: hub.fuc ctx_xfer / ctx_xfer_idle (lines 602-612)")
    print("=" * 72)
    for i, line in enumerate(open(SRC), 1):
        if 602 <= i <= 612:
            print(f"  {i}: {line.rstrip()}")

if __name__ == "__main__":
    main()
