#!/usr/bin/env python3
"""Generate the phase-4 JAX logits parity fixture for the native needle graph.

Reads the pinned `.cact` export (the exact QAT operands the EMEL runtime
computes from), inverts the export.py `_tensors()` emission back into the
reference params tree, and runs the maintained JAX reference decode
(`needle.model.decode._forward_cached`) to dump per-step logits.

Run with the needle venv:
    /data/needle/.venv/bin/python3 scripts/gen_needle_logits_fixture.py \
        /shared/effortless/train/route-w4-qat.cact tests/fixtures/cact
"""

import json
import struct
import sys

sys.path.insert(0, "/data/needle")

import numpy as np  # noqa: E402

from needle.model import decode  # noqa: E402
from needle.model.export import read_export, parse_tokenizer_blob, _SP_META_SPACE  # noqa: E402


def rebuild_params(geometry, tensors):
    g = geometry
    L = g["num_layers"]
    orders = g["engram_orders"]
    num_tables = g["num_engram_tables"]
    heads = num_tables // len(orders)
    sub_dim = g["engram_sub_dim"]
    slots = g["engram_slots"]
    sites = g["engram_layers"]

    it = iter(range(g["num_tensors"]))

    def take():
        return np.asarray(tensors[next(it)], np.float32)

    params = {}
    params["embedding"] = {"embedding": take()}

    layer = {k: [] for k in (
        "norm_in", "q", "k", "v", "q_norm", "k_norm", "gate", "out",
        "post_norm", "attn_gate", "pre_hada", "d1", "d2", "d3")}
    for _ in range(L):
        layer["norm_in"].append(take())
        layer["q"].append(take().T)      # exporter stored kernel.T
        layer["k"].append(take().T)
        layer["v"].append(take().T)
        layer["q_norm"].append(take())
        layer["k_norm"].append(take())
        layer["gate"].append(take().T)
        layer["out"].append(take().T)
        layer["post_norm"].append(take())
        layer["attn_gate"].append(take().reshape(()))
        layer["pre_hada"].append(take())
        layer["d1"].append(take())
        layer["d2"].append(take())
        layer["d3"].append(take())

    def st(name):
        return np.stack(layer[name])

    stack = {
        "layers": {"block": {
            "ZCRMSNorm_0": {"scale": st("norm_in")},
            "self_attn": {
                "q_proj": {"kernel": st("q")},
                "k_proj": {"kernel": st("k")},
                "v_proj": {"kernel": st("v")},
                "gate_proj": {"kernel": st("gate")},
                "out_proj": {"kernel": st("out")},
                "q_norm": {"scale": st("q_norm")},
                "k_norm": {"scale": st("k_norm")},
            },
            "hadamard_mlp": {"d1": st("d1"), "d2": st("d2"), "d3": st("d3")},
            "post_attn_norm": {"scale": st("post_norm")},
            "attn_gate": st("attn_gate"),
            "pre_hada_norm": {"scale": st("pre_hada")},
        }},
    }

    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
                 "mhc_b_res"):
        stack[name] = take()
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        phi = take()                       # (L*out_lanes, nC); out_lanes = n (pre/post) or n*n (res)
        nC = phi.shape[1]
        out_lanes = phi.shape[0] // L
        stack[name] = phi.reshape(L, out_lanes, nC).transpose(0, 2, 1)

    params["stack"] = stack

    for s in range(len(sites)):
        tables = take().reshape(num_tables, slots, sub_dim)
        params[f"engrams_{s}"] = {
            "embedding": tables,
            "key_proj": {"kernel": take().T},
            "value_proj": {"kernel": take().T},
            "taps": take(),
        }

    stack["final_norm"] = {"scale": take()}

    # Remaining tensors: head manifest + head weights + tokenizer RAW blob.
    # Heads are not part of the LM logits path; skip to the RAW blob.
    blob = None
    for t in tensors:
        if isinstance(t, (bytes, bytearray)):
            blob = t
    assert blob is not None, "tokenizer RAW blob missing"
    return params, blob, heads


def main():
    cact_path = sys.argv[1] if len(sys.argv) > 1 else "/shared/effortless/train/route-w4-qat.cact"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/cact"

    geometry, tensors = read_export(cact_path)
    params, tok_blob, engram_heads = rebuild_params(geometry, tensors)
    g = geometry

    tok = parse_tokenizer_blob(tok_blob)
    pieces = tok["pieces"]
    piece_to_id = {p: i for i, p in enumerate(pieces)}

    def encode(text):
        # Greedy longest-match SPM encode consistent with RefTokenizer for
        # simple ASCII prompts (parity prompts chosen to round-trip exactly;
        # the C++ side uses its own maintained tokenizer and the fixture stores
        # token ids, so encode differences cannot skew logits parity).
        s = _SP_META_SPACE + text.replace(" ", _SP_META_SPACE)
        ids, i = [], 0
        while i < len(s):
            for j in range(len(s), i, -1):
                cand = s[i:j]
                if cand in piece_to_id:
                    ids.append(piece_to_id[cand])
                    i = j
                    break
            else:
                raise ValueError(f"unencodable piece at {i}: {s[i:]}")
        return ids

    head_dim = g["head_dim"]
    cfg = decode.DecodeCfg(
        d_model=g["d_model"], num_heads=g["num_heads"],
        num_kv_heads=g["num_kv_heads"], num_layers=g["num_layers"],
        attn_dim=g["num_heads"] * head_dim, mhc_lanes=g["mhc_lanes"],
        engram_layers=tuple(g["engram_layers"]),
        engram_orders=tuple(g["engram_orders"]),
        engram_heads=engram_heads, engram_slots=g["engram_slots"],
        kv_window=int(g["kv_window"]),
    )

    from needle.model.architecture import precompute_rope_freqs
    import jax.numpy as jnp

    prompts = ["hello world", "route this request", "effort"]
    decode_steps = 3
    bos = tok["bos_id"]

    def _js(v):
        if isinstance(v, tuple):
            return list(v)
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    manifest = {"cact": cact_path, "geometry": {k: _js(v) for k, v in g.items()},
                "bos": bos, "decode_steps": decode_steps, "cases": []}

    for ci, prompt in enumerate(prompts):
        ids = [bos] + encode(prompt)
        max_len = len(ids) + decode_steps + 1
        cos, sin = precompute_rope_freqs(head_dim, max_len, theta=float(g["rope_theta"]))
        kshape = (cfg.num_layers, 1, cfg.num_kv_heads, max_len, head_dim)
        k_cache = jnp.zeros(kshape, jnp.float32)
        v_cache = jnp.zeros(kshape, jnp.float32)

        tokens = jnp.asarray([ids], jnp.int32)
        hist = jnp.zeros((1, max_len), jnp.int32).at[0, :len(ids)].set(jnp.asarray(ids))
        hist_valid = jnp.zeros((1, max_len), bool).at[0, :len(ids)].set(True)

        step_logits = []
        logits, k_cache, v_cache = decode._forward_cached(
            params, cfg, tokens, k_cache, v_cache, 0, cos, sin,
            hist=hist, hist_valid=hist_valid)
        step_logits.append(np.asarray(logits[0, -1], np.float32))

        pos = len(ids)
        cur = int(np.argmax(step_logits[-1]))
        gen = [cur]
        for _ in range(decode_steps - 1):
            hist = hist.at[0, pos].set(cur)
            hist_valid = hist_valid.at[0, pos].set(True)
            logits, k_cache, v_cache = decode._forward_cached(
                params, cfg, jnp.asarray([[cur]], jnp.int32), k_cache, v_cache,
                pos, cos, sin, hist=hist, hist_valid=hist_valid)
            step_logits.append(np.asarray(logits[0, -1], np.float32))
            pos += 1
            cur = int(np.argmax(step_logits[-1]))
            gen.append(cur)

        bin_name = f"route-w4-qat.logits.case{ci}.bin"
        with open(f"{out_dir}/{bin_name}", "wb") as f:
            for arr in step_logits:
                f.write(struct.pack(f"<{len(arr)}f", *arr.tolist()))
        manifest["cases"].append({
            "prompt": prompt, "prompt_ids": ids, "greedy": gen,
            "steps": len(step_logits), "vocab": int(g["vocab_size"]),
            "file": bin_name,
        })
        print(f"case{ci}: prompt_ids={ids} greedy={gen}")

    with open(f"{out_dir}/route-w4-qat.logits.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(prompts)} cases to {out_dir}")


if __name__ == "__main__":
    main()
