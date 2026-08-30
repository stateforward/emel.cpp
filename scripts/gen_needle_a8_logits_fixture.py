#!/usr/bin/env python3
"""Generate authoritative W4A8 needle logits from the pinned `.cact` model.

The existing ``route-w4-qat.logits`` fixture intentionally exercises the
legacy W4/f32 cached-decode parity route.  This generator instead reconstructs
the exact dequantized CQ4 operands exported in the pinned `.cact`, configures
the deployment quantizer, and calls ``SimpleAttentionNetwork.apply`` with
``quant=True`` for every growing prefill/decode sequence.  That maintained JAX
architecture path applies signed A8 fake quantization at all deployment `_aq`
sites: engram fetched activations before key/value projections, attention input
before q/k/v/gate projections, attention output before out_proj, and final
normalized hidden before tied-embedding logits.  MTP-only `_aq` sites do not
apply to the LM decode path.

Run with the needle venv:
    /data/needle/.venv/bin/python3 scripts/gen_needle_a8_logits_fixture.py \
        /shared/effortless/train/route-w4-qat.cact tests/fixtures/cact
"""

import json
import os
import struct
import sys

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
sys.path.insert(0, "/data/needle")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gen_needle_logits_fixture import rebuild_params  # noqa: E402
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig  # noqa: E402
from needle.model.export import read_export, parse_tokenizer_blob, _SP_META_SPACE  # noqa: E402
from needle.model.quantize import configure_deploy, fake_quant_act  # noqa: E402


def _config(geometry, engram_heads):
    return TransformerConfig(
        vocab_size=geometry["vocab_size"],
        d_model=geometry["d_model"],
        attn_dim=geometry["num_heads"] * geometry["head_dim"],
        num_heads=geometry["num_heads"],
        num_kv_heads=geometry["num_kv_heads"],
        num_layers=geometry["num_layers"],
        max_seq_len=geometry["max_seq_len"],
        rope_theta=float(geometry["rope_theta"]),
        dtype="float32",
        flash=False,
        engram_orders=tuple(geometry["engram_orders"]),
        engram_heads=engram_heads,
        engram_slots=geometry["engram_slots"],
        engram_layers=tuple(geometry["engram_layers"]),
        mhc_lanes=geometry["mhc_lanes"],
        kv_window=int(geometry["kv_window"]),
        kv_bits=int(geometry["kv_bits"]),
        act_bits=8,
        remat=False,
    )


def _encoder(tokenizer):
    pieces = tokenizer["pieces"]
    piece_to_id = {piece: i for i, piece in enumerate(pieces)}

    def encode(text):
        source = _SP_META_SPACE + text.replace(" ", _SP_META_SPACE)
        ids = []
        offset = 0
        while offset < len(source):
            for end in range(len(source), offset, -1):
                candidate = source[offset:end]
                if candidate in piece_to_id:
                    ids.append(piece_to_id[candidate])
                    offset = end
                    break
            else:
                raise ValueError(f"unencodable piece at {offset}: {source[offset:]}")
        return ids

    return encode


def _json_value(value):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main():
    cact_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/shared/effortless/train/route-w4-qat.cact"
    )
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/cact"
    os.makedirs(out_dir, exist_ok=True)

    geometry, tensors = read_export(cact_path)
    params, tokenizer_blob, engram_heads = rebuild_params(geometry, tensors)
    tokenizer = parse_tokenizer_blob(tokenizer_blob)
    encode = _encoder(tokenizer)

    configure_deploy(act_bits=8, kv_bits=8)
    config = _config(geometry, engram_heads)
    model = SimpleAttentionNetwork(config)
    variables = {"params": jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), params)}
    apply_quant = jax.jit(lambda ids: model.apply(variables, ids, quant=True))

    prompts = ["hello world", "route this request", "effort"]
    decode_steps = 3
    bos = tokenizer["bos_id"]
    manifest = {
        "cact": cact_path,
        "parity_contract": "W4A8 deployment parity",
        "reference_path": "SimpleAttentionNetwork.apply(quant=True)",
        "activation_quant": {
            "bits": 8,
            "qmax": 127,
            "group": "full last dimension",
            "rounding": "jax.numpy.round ties-to-even",
            "clamp": [-128, 127],
            "sites": [
                "engram fetched activation before key/value projections",
                "attention input before q/k/v/gate projections",
                "attention output before out_proj",
                "final normalized hidden before tied embedding logits",
            ],
            "non_decode_sites": [
                "MTP concatenate before mtp_combine",
                "MTP final normalized hidden before tied logits",
            ],
        },
        "geometry": {key: _json_value(value) for key, value in geometry.items()},
        "bos": bos,
        "decode_steps": decode_steps,
        "cases": [],
    }

    probes = [
        np.zeros((8,), np.float32),
        np.asarray([-127.0, -126.5, -1.5, -0.5, 0.0, 0.5, 1.5, 126.5, 127.0], np.float32),
        np.asarray([-2.0, -1.0, -0.25, 0.25, 1.0, 2.0], np.float32),
    ]
    manifest["activation_quant"]["probes"] = [
        {
            "input": probe.tolist(),
            "output": np.asarray(fake_quant_act(jnp.asarray(probe)), np.float32).tolist(),
        }
        for probe in probes
    ]

    for case_index, prompt in enumerate(prompts):
        ids = [bos] + encode(prompt)
        generated = []
        step_logits = []
        sequence = list(ids)
        for _ in range(decode_steps):
            logits = np.asarray(
                apply_quant(jnp.asarray([sequence], jnp.int32))[0, -1], np.float32
            )
            step_logits.append(logits)
            token = int(np.argmax(logits))
            generated.append(token)
            sequence.append(token)

        binary_name = f"route-w4-qat-a8.logits.case{case_index}.bin"
        with open(os.path.join(out_dir, binary_name), "wb") as output:
            for logits in step_logits:
                output.write(struct.pack(f"<{len(logits)}f", *logits.tolist()))
        manifest["cases"].append(
            {
                "prompt": prompt,
                "prompt_ids": ids,
                "greedy": generated,
                "steps": len(step_logits),
                "vocab": int(geometry["vocab_size"]),
                "file": binary_name,
            }
        )
        print(f"case{case_index}: prompt_ids={ids} greedy={generated}")

    manifest_path = os.path.join(out_dir, "route-w4-qat-a8.logits.json")
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=1)
        output.write("\n")
    print(f"wrote {len(prompts)} W4A8 cases to {out_dir}")


if __name__ == "__main__":
    main()
