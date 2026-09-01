#!/usr/bin/env python3
"""Generate NV50 (Tesla, sm_12) compute kernel binary for add: out[i] = a[i] + b[i]

Kernel logic:
  tid.x = $r0 & 0xffff
  ntid.x = s[0x2]  (blockDim.x, 16-bit)
  ctaid.x = s[0xc]  (blockIdx.x, 16-bit)
  i = ctaid.x * ntid.x + tid.x
  byte_offset = i * 4
  a[i] = ld g0[byte_offset]
  b[i] = ld g1[byte_offset]
  out[i] = a[i] + b[i]
  st g2[byte_offset], out[i]
  exit

Register allocation:
  $r0  - combined thread ID (implicit)
  $r1l - temp for 16-bit shared loads
  $r3  - ntid.x (32-bit)
  $r4  - ctaid.x (32-bit)
  $r5  - tid.x (32-bit)
  $r6  - global index i
  $r7  - byte offset
  $r8  - a[i]
  $r9  - b[i]
  $r10 - result
"""
import struct, sys

def w(out, c0, c1):
    """Write an 8-byte (64-bit) instruction."""
    out.extend(struct.pack('<II', c0 & 0xFFFFFFFF, c1 & 0xFFFFFFFF))

def gen_kernel():
    out = bytearray()

    # 1. and b32 $r5, $r0, 0xffff  (tid.x = $r0 & 0xffff)
    # emitLogicOp: code[0]=0xd0000000, immediate form
    # emitForm_IMM: code[0] |= 1, setDst(5), setSrc(0,0)=r0, setImmediate(1)=0xffff
    c0 = 0xd0000001 | (5 << 2) | (0 << 9) | ((0xffff & 0x3f) << 16)
    c1 = 3 | ((0xffff >> 6) << 2)
    w(out, c0, c1)

    # 2. ld u16 $r1l, s[0x2]  (ntid.x = blockDim.x from shared mem)
    # code[0]=0x10000001, code[1]=0x40000000 | 0x4000 (u16 size)
    # dst=$r1l (half-reg id=2), offset=0x2/2=1 (scale 1 for 16-bit)
    c0 = 0x10000001 | (2 << 2) | (1 << 9)
    c1 = 0x40004000
    w(out, c0, c1)

    # 3. cvt u32 u16 $r3, $r1l  (zero-extend ntid.x to 32-bit)
    # code[0]=0xa0000000, code[1]=0x04000000 (U32 from U16)
    # emitForm_MAD: code[0] |= 1, setDst(3), setSrc(0,0)=$r1l(id=2)
    c0 = 0xa0000001 | (3 << 2) | (2 << 9)
    c1 = 0x04000000
    w(out, c0, c1)

    # 4. ld u16 $r1l, s[0xc]  (ctaid.x = blockIdx.x from shared mem)
    # offset=0xc/2=6 (scale 1 for 16-bit)
    c0 = 0x10000001 | (2 << 2) | (6 << 9)
    c1 = 0x40004000
    w(out, c0, c1)

    # 5. cvt u32 u16 $r4, $r1l  (zero-extend ctaid.x to 32-bit)
    c0 = 0xa0000001 | (4 << 2) | (2 << 9)
    c1 = 0x04000000
    w(out, c0, c1)

    # 6. mul b32 $r6, $r4, $r3  (ctaid.x * ntid.x) - long form
    # emitIMUL: code[0]=0x40000000, long: code[1]=0 (U32)
    # emitForm_MAD: code[0] |= 1, setDst(6), setSrc(0,0)=r4, setSrc(1,1)=r3
    # For 32-bit mul long form, need to set size bit in code[1]
    # Looking at g80.c long form: 32-bit needs code[1] |= 0x04000000
    c0 = 0x40000001 | (6 << 2) | (4 << 9) | (3 << 16)
    c1 = 0x04000000  # 32-bit size flag
    w(out, c0, c1)

    # 7. add b32 $r6, $r6, $r5  (i = ctaid.x*ntid.x + tid.x) - long form
    # emitUADD: code[0]=0x20000000, long: code[1]=0x04000000 (b32)
    # emitForm_ADD: code[0] |= 1, setDst(6), setSrc(0,0)=r6, setSrc(1,2)=r5
    c0 = 0x20000001 | (6 << 2) | (6 << 9)
    c1 = 0x04000000 | (5 << 14)
    w(out, c0, c1)

    # 8. shl b32 $r7, $r6, 2  (byte offset = i * 4)
    # emitShift: code[0]=0x30000001, code[1]=0xc0000000 (SHL)
    # For b32: code[1] |= 0x04000000
    # Shift amount is in bits 23-28 of code[1]
    # Actually, looking at emitShift more carefully...
    # The shift amount for immediate shifts uses emitARL if dst is address reg
    # Otherwise: code[0]=0x30000001, code[1]=0xc0000000, b32: code[1]|=0x04000000
    # But the shift amount... let me check the g80.c table
    c0 = 0x30000001 | (7 << 2) | (6 << 9)
    c1 = 0xc0000000 | 0x04000000 | (2 << 23)  # SHL, b32, shift=2
    w(out, c0, c1)

    # 9. ld b32 $r8, g0[$r7]  (load a[i])
    # code[0]=0xd0000001 | (g_idx<<16) | (dst<<2) | (addr_reg<<9)
    # code[1]=0x80000000 | (0x6<<21)  (U32 size)
    c0 = 0xd0000001 | (0 << 16) | (8 << 2) | (7 << 9)
    c1 = 0x80000000 | (0x6 << 21)
    w(out, c0, c1)

    # 10. ld b32 $r9, g1[$r7]  (load b[i])
    c0 = 0xd0000001 | (1 << 16) | (9 << 2) | (7 << 9)
    c1 = 0x80000000 | (0x6 << 21)
    w(out, c0, c1)

    # 11. add b32 $r10, $r8, $r9  (result = a[i] + b[i]) - long form
    c0 = 0x20000001 | (10 << 2) | (8 << 9)
    c1 = 0x04000000 | (9 << 14)
    w(out, c0, c1)

    # 12. st b32 g2[$r7], $r10  (store result)
    # code[0]=0xd0000001 | (g_idx<<16) | (data_reg<<2) | (addr_reg<<9)
    # code[1]=0xa0000000 | (0x6<<21)
    c0 = 0xd0000001 | (2 << 16) | (10 << 2) | (7 << 9)
    c1 = 0xa0000000 | (0x6 << 21)
    w(out, c0, c1)

    # 13. exit
    # code[0]=0x00000001, code[1]=0x00000001
    w(out, 0x00000001, 0x00000001)

    return bytes(out)

if __name__ == '__main__':
    binary = gen_kernel()
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        with open(sys.argv[idx+1], 'wb') as f:
            f.write(binary)
        print(f'Wrote {len(binary)} bytes to {sys.argv[idx+1]}', file=sys.stderr)
    else:
        sys.stdout.buffer.write(binary)

    # Also print hex dump to stderr for debugging
    print(f'Kernel size: {len(binary)} bytes ({len(binary)//8} instructions)', file=sys.stderr)
    for i in range(0, len(binary), 8):
        c0 = struct.unpack_from('<I', binary, i)[0]
        c1 = struct.unpack_from('<I', binary, i+4)[0]
        print(f'  [{i//8:2d}] 0x{c0:08x} 0x{c1:08x}', file=sys.stderr)
