#!/usr/bin/env python3
"""Dump a deterministic sample of the `.cact` trailing RAW tokenizer blob
(id/score/type/surface) to a CSV parity fixture for the EMEL needle
tokenizer-loader tests.

This re-implements the raw struct unpacking from `needle/model/export.py`
(`_TK_HDR "<IIIIIBBH"` + n_pieces of `_TK_REC "<fBH"` + surface bytes)
instead of importing needle, so the fixture is generated from the same
byte-level source of truth the EMEL loader validates.

The sample keeps the fixture small: the header ids, the first 16 pieces,
every 256th piece, and the last piece. Surfaces are hex-encoded so the CSV
stays ASCII-clean for multi-byte UTF-8 pieces.

Usage:
    python3 scripts/gen_cact_tokenizer_csv.py \
        tests/models/route-w4-qat.cact \
        tests/fixtures/cact/route-w4-qat.tokenizer.csv
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
DTYPE_RAW = 4
TK_HDR = "<IIIIIBBH"
TK_REC = "<fBH"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.cact> <output.csv>", file=sys.stderr)
        return 2

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        data = f.read()

    header_fields = struct.unpack_from(HDR_FMT, data, 0)
    if header_fields[0] != TAG:
        raise ValueError(f"bad tag: {header_fields[0]:#x} != {TAG:#x}")
    num_tensors = header_fields[1]
    if header_fields[2] != CODEBOOK_LEN:
        raise ValueError(f"unexpected codebook_len: {header_fields[2]}")

    directory_offset = HDR_SIZE + CODEBOOK_LEN * 4
    last_rec = struct.unpack_from(
        REC_FMT, data, directory_offset + (num_tensors - 1) * REC_SIZE
    )
    if last_rec[0] != DTYPE_RAW:
        raise ValueError("last tensor is not the RAW tokenizer blob")
    blob = data[last_rec[7] : last_rec[7] + last_rec[8]]

    n, pad, eos, bos, unk, add_dummy, byte_fb, _pad = struct.unpack_from(
        TK_HDR, blob, 0
    )

    offset = struct.calcsize(TK_HDR)
    rec_size = struct.calcsize(TK_REC)
    pieces = []
    for _ in range(n):
        score, piece_type, surface_len = struct.unpack_from(TK_REC, blob, offset)
        offset += rec_size
        surface = blob[offset : offset + surface_len]
        offset += surface_len
        pieces.append((score, piece_type, surface))
    if offset != len(blob):
        raise ValueError(f"trailing bytes: consumed {offset} of {len(blob)}")

    sample_ids = sorted(
        set([pad, eos, bos, unk])
        | set(range(min(16, n)))
        | set(range(0, n, 256))
        | {n - 1}
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["n_pieces", "pad_id", "eos_id", "bos_id", "unk_id",
             "add_dummy_prefix", "byte_fallback"]
        )
        writer.writerow([n, pad, eos, bos, unk, add_dummy, byte_fb])
        writer.writerow(["id", "score", "type", "surface_hex"])
        for piece_id in sample_ids:
            score, piece_type, surface = pieces[piece_id]
            writer.writerow(
                [piece_id, f"{score:.1f}", piece_type, surface.hex()]
            )

    print(f"wrote {len(sample_ids)} sampled pieces of {n} to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
