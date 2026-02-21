# contributing

welcome to **emel.cpp** — the project that makes traditionalists seethe.

### AI-first policy (yes, we’re serious)

> [!IMPORTANT]
> this project **only accepts AI-generated code**.

if you show up with beautiful, hand-crafted, artisanal human code that you spent hours perfecting… that’s adorable. we will still have the AI rewrite it before we merge.

don’t take it personally. we’re not rejecting *you*. we’re rejecting the 2025 belief that humans are inherently better at writing production systems code.

human contributions that refuse the AI rewrite will be closed with a polite link back to this file and a gentle “better luck next time.”

we are deliberately building the public counter-example to the tired claim that “AI can’t write real production systems code.” so far the silicon is winning.

(llama.cpp proudly rejects AI-generated code. we proudly reject non-AI-generated code. different vibes.)

### how this actually works (throughput-first)

- use AI. heavily. (this is not optional.)
- tell us exactly which models and tools you used.
- the AI (grok, gemini, or whichever frontier model you choose) will handle deep code-quality review — not the human maintainer. this is how we keep velocity high.
- minor style nits are irrelevant. we care about architecture, determinism, and rule compliance, not perfect indentation.

human review is deliberately light. the real judges are the quality gates + AI analysis.

### pull request requirements (non-negotiable)

before you hit submit:
- read and follow `docs/sml.rules.md` and `AGENTS.md` like they’re gospel.
- keep changes small and focused.
- run `scripts/quality_gates.sh` (this is the only required gate script; it runs everything).
- add or update tests for any new behavior.
- update documentation when behavior changes.

benchmarks are enforced via snapshots; see `docs/benchmark.md`.

individual gate scripts live in `scripts/`, but you should only invoke them through `scripts/quality_gates.sh`.

### architecture & style rules (the constitution)

full details live in `docs/sml.rules.md`. highlights:
- boost.SML is the *only* orchestration mechanism allowed.
- run-to-completion actor model. no queues, no deferred events, no re-entrancy, no excuses.
- guards must be pure. actions must be bounded and allocation-free during dispatch.
- no llama.cpp or ggml names, patterns, or energy permitted.
- `snake_case` for functions/variables, `pascal_case` for types and state names.

### final words

we’re not here to debate whether AI can write good code.

we’re here to make the debate irrelevant with clean, explicit, auditable, high-quality C++ — shipped at maximum velocity.

welcome to the rebellion. silicon and all.
