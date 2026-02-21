# Contributing

Welcome to **emel.cpp** — the project that makes traditionalists seethe.

### AI-First Policy (yes, we’re serious)

> [!IMPORTANT]
> This project **only accepts AI-generated code**.

If you show up with beautiful, hand-crafted, artisanal human code that you spent hours perfecting… that’s adorable. We will still have the AI rewrite it before we merge.

Don’t take it personally. We’re not rejecting *you*. We’re rejecting the 2025 belief that humans are inherently better at writing production systems code.

Human contributions that refuse the AI rewrite will be closed with a polite link back to this file and a gentle “better luck next time.”

We are deliberately building the public counter-example to the tired claim that “AI can’t write real production systems code.” So far the silicon is winning.

(llama.cpp proudly rejects AI-generated code. We proudly reject non-AI-generated code. Different vibes.)

### How this actually works (throughput-first)

- Use AI. Heavily. (This is not optional.)
- Tell us exactly which models and tools you used.
- The AI (Grok, Gemini, or whichever frontier model you choose) will handle deep code-quality review — not the human maintainer. This is how we keep velocity high.
- Minor style nits are irrelevant. We care about architecture, determinism, and rule compliance, not perfect indentation.

Human review is deliberately light. The real judges are the quality gates + AI analysis.

### Pull request requirements (non-negotiable)

Before you hit submit:
- Read and follow `docs/sml.rules.md` and `AGENTS.md` like they’re gospel.
- Keep changes small and focused.
- Run `scripts/quality_gates.sh` (this is the only required gate script; it runs everything).
- Add or update tests for any new behavior.
- Update documentation when behavior changes.

Benchmarks are enforced via snapshots; see `docs/benchmark.md`.

Individual gate scripts live in `scripts/`, but you should only invoke them through `scripts/quality_gates.sh`.

### Architecture & style rules (the constitution)

Full details live in `docs/sml.rules.md`. Highlights:
- Boost.SML is the *only* orchestration mechanism allowed.
- Run-to-completion actor model. No queues, no deferred events, no re-entrancy, no excuses.
- Guards must be pure. Actions must be bounded and allocation-free during dispatch.
- No llama.cpp or ggml names, patterns, or energy permitted.
- `snake_case` for functions/variables, `PascalCase` for types and state names.

### Final words

We’re not here to debate whether AI can write good code.

We’re here to make the debate irrelevant with clean, explicit, auditable, high-quality C++ — shipped at maximum velocity.

Welcome to the rebellion. Silicon and all.
