#!/usr/bin/env python3
"""Prepare the needle heldout-accuracy eval input for tools/needle_eval.

Renders each /shared/effortless/train/heldout.jsonl row exactly like the
training renderer (`/data/needle/needle/model/finetune.py` render_example,
prompt part only) and encodes it with the byte-level reference tokenizer
(`needle/model/export.py` RefTokenizer) built FROM THE PINNED .cact BLOB, so
the C++ eval can cross-check emel tokenizer parity row by row.

Output line format (one row per heldout example):
    gold_domain \t gold_effort \t ref_ids_space_separated \t prompt_hex

Run with the needle venv:
    /data/needle/.venv/bin/python3 scripts/gen_needle_heldout_eval.py \
        /shared/effortless/train/heldout.jsonl \
        tests/models/route-w4-qat.cact \
        build/needle_eval/heldout_prompts.tsv
"""
import json
import sys

sys.path.insert(0, "/data/needle")

from needle.model.export import RefTokenizer  # noqa: E402

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOLS_START = "<tools>"
TOOLS_END = "</tools>"


def render_prompt(example):
    tools = example.get("tools", [])
    tools_json = tools if isinstance(tools, str) else json.dumps(
        tools, separators=(",", ":"), ensure_ascii=False)
    system = (example.get("system") or "").strip()
    prefix = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    return (prefix + IM_START + "user\n" + TOOLS_START + tools_json +
            TOOLS_END + "\n" + example["query"] + IM_END + "\n" + IM_START +
            "assistant\n")


def main():
    eval_path, cact_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    tokenizer = RefTokenizer.from_cact(cact_path)
    rows = [json.loads(line) for line in open(eval_path)]
    with open(out_path, "w") as out:
        for row in rows:
            gold = row["answers"][0]["arguments"]
            prompt = render_prompt(row)
            ids = tokenizer.encode(prompt)
            out.write("%s\t%s\t%s\t%s\n" % (
                gold["domain"], gold["effort"],
                " ".join(str(i) for i in ids),
                prompt.encode("utf-8").hex()))
    print("wrote %d rows to %s" % (len(rows), out_path))


if __name__ == "__main__":
    main()
