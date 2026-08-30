#!/usr/bin/env python3
"""Dump the raw `.cact` tensor directory (dtype/shape/offset/nbytes/group/bits)
to a CSV parity fixture for the EMEL cact loader tests.

This intentionally re-implements the raw struct unpacking from
`needle/model/export.py` instead of importing needle's `read_export()`,
because `read_export()` only returns dequantized numpy tensors and discards
the on-disk directory metadata (offset/nbytes/group/bits) that the EMEL
loader is responsible for validating.

Usage:
    /data/needle/.venv/bin/python3 scripts/gen_cact_directory_csv.py \
        /shared/effortless/train/route-w4-qat.cact \
        tests/fixtures/cact/route-w4-qat.directory.csv
"""

import csv
import struct
import sys

TAG = 0x05E12A83
HDR_FMT = "<29If"
HDR_SIZE = struct.calcsize(HDR_FMT)
REC_FMT = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(REC_FMT)
CODEBOOK_LEN = 28


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.cact> <output.csv>", file=sys.stderr)
        return 2

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        data = f.read()

    header_fields = struct.unpack_from(HDR_FMT, data, 0)
    tag = header_fields[0]
    if tag != TAG:
        raise ValueError(f"bad tag: {tag:#x} != {TAG:#x}")
    num_tensors = header_fields[1]
    codebook_len = header_fields[2]
    if codebook_len != CODEBOOK_LEN:
        raise ValueError(f"unexpected codebook_len: {codebook_len} != {CODEBOOK_LEN}")

    directory_offset = HDR_SIZE + CODEBOOK_LEN * 4

    rows = []
    for i in range(num_tensors):
        rec_offset = directory_offset + i * REC_SIZE
        (dtype, ndim, _pad, s0, s1, s2, s3, offset, nbytes, group, bits) = struct.unpack_from(
            REC_FMT, data, rec_offset
        )
        rows.append((i, dtype, ndim, s0, s1, s2, s3, offset, nbytes, group, bits))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "dtype",
                "ndim",
                "shape0",
                "shape1",
                "shape2",
                "shape3",
                "offset",
                "nbytes",
                "group",
                "bits",
            ]
        )
        writer.writerows(rows)

    print(f"wrote {len(rows)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
