# contributing

welcome to **emel.cpp** — the project that makes traditionalists seethe.

### AI-forward policy (yes, we’re serious)

> [!IMPORTANT]
> this project strongly encourages **AI-assisted and AI-written code**, but solid human-written
> code is welcome too.

the point is not authorship purity. the point is throughput, clarity, and whether the change meets
the repo's engineering bar.

if AI helped shape the change, say so. if you wrote it by hand and it still clears the rules,
tests, and review bar, that is acceptable too.

AI-written code is acceptable here. human-written code is acceptable here. what gets merged is the
code that is explicit, auditable, deterministic, and aligned with the project rules.

### how this actually works (throughput-first)

- use AI when it helps. it usually will.
- tell us which models and tools you used when AI materially contributed to the change.
- AI review is encouraged and often preferred for deep code-quality passes, but the merge bar is
  still the repo's technical rules and the actual diff in front of us.
- minor style nits are irrelevant. we care about architecture, determinism, and rule compliance, not perfect indentation.

human review is deliberately focused. the real judges are the quality gates, the canonical rules,
and whether the change actually improves the codebase.

### pull request requirements (non-negotiable)

before you hit submit:
- read and follow `docs/rules/sml.rules.md` and `AGENTS.md` like they’re gospel.
- keep changes small and focused.
- run `scripts/quality_gates.sh` (this is the only required gate script; it runs everything).
- add or update tests for any new behavior.
- update documentation when behavior changes.

benchmarks are enforced via snapshots; see `docs/benchmark.md`.

individual gate scripts live in `scripts/`, but you should only invoke them through `scripts/quality_gates.sh`.

### architecture & style rules (the constitution)

full details live in `docs/rules/sml.rules.md`. highlights:
- boost.SML is the *only* orchestration mechanism allowed.
- run-to-completion actor model. no queues, no deferred events, no re-entrancy, no excuses.
- guards must be pure. actions must be bounded and allocation-free during dispatch.
- no llama.cpp or ggml names, patterns, or energy permitted outside the explicitly allowed tool
  paths.
- naming and structure rules come from `AGENTS.md` and `docs/rules/sml.rules.md`. do not invent
  local exceptions.

### final words

we’re not here to debate whether AI can write good code.

we’re here to make the debate irrelevant with clean, explicit, auditable, high-quality C++ shipped
at maximum velocity.

welcome to the rebellion. silicon and all.
