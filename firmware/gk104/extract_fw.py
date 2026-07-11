import pathlib, re, struct
ARRAYS = {
    "hubgk104.fuc3.h": {"gk104_grhub_code": "gk104_fecs_code.bin", "gk104_grhub_data": "gk104_fecs_data.bin"},
    "gpcgk104.fuc3.h": {"gk104_grgpc_code": "gk104_gpccs_code.bin", "gk104_grgpc_data": "gk104_gpccs_data.bin"},
    "gf119.fuc4.h":    {"gf119_pmu_code": "gk104_pmu_code.bin", "gf119_pmu_data": "gk104_pmu_data.bin"},
}
for header, arrays in ARRAYS.items():
    text = pathlib.Path(header).read_text()
    for array_name, output_name in arrays.items():
        m = re.search(rf"static\s+uint32_t\s+{array_name}\s*\[\]\s*=\s*\{{(.*?)\}};", text, flags=re.S)
        if not m: raise RuntimeError(f"no {array_name} in {header}")
        body = re.sub(r"/\*.*?\*/|//[^\n]*", "", m.group(1), flags=re.S)
        words = [int(v,16) for v in re.findall(r"0x[0-9a-fA-F]+", body)]
        with open(output_name,"wb") as o:
            for w in words: o.write(struct.pack("<I", w))
        print(f"{output_name}: {len(words)*4} bytes ({len(words)} words)")
