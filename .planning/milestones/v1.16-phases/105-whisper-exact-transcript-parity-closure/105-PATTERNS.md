# Phase 105: Whisper Exact Transcript Parity Closure - Pattern Map

**Mapped:** 2026-04-27
**Files analyzed:** 7 new/modified/evidence files
**Analogs found:** 6 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Rule-Safe Analog | Match Quality |
|-------------------|------|-----------|--------------------------|---------------|
| `.planning/phases/105-whisper-exact-transcript-parity-closure/105-01-SUMMARY.md` | planning-artifact | transform | `.planning/milestones/v1.15-phases/92-milestone-evidence-validation-and-ledger-closeout/92-01-SUMMARY.md` | role-match |
| `.planning/phases/105-whisper-exact-transcript-parity-closure/105-VERIFICATION.md` | test/evidence | batch, file-I/O | `.planning/milestones/v1.15-phases/92.1-maintained-gguf-contract-parity-and-benchmark-truth-repair/92.1-VERIFICATION.md` | role-match |
| `.planning/phases/105-whisper-exact-transcript-parity-closure/105-VALIDATION.md` | test/evidence | batch, file-I/O | existing `.planning/phases/105-whisper-exact-transcript-parity-closure/105-VALIDATION.md` plus `.planning/milestones/v1.13-phases/75-comparability-verdict-and-single-lane-publication-repair/75-VALIDATION.md` | exact |
| `.planning/ROADMAP.md` | config/ledger | transform | current `.planning/ROADMAP.md` Phase 105/106/107/108 block | partial |
| `.planning/STATE.md` | config/ledger | transform | current `.planning/STATE.md` reopened v1.16 status block | partial, must repair |
| `build/whisper_compare/summary.json` | evidence artifact | batch, file-I/O | current `build/whisper_compare/summary.json` plus `tools/bench/whisper_compare.py` summary writer | exact evidence input |
| `build/whisper_compare/normalized/manifest.json` | evidence artifact | file-I/O | `scripts/bench_whisper_compare.sh` normalizer invocation | exact evidence input |

## Pattern Assignments

### `.planning/phases/105-whisper-exact-transcript-parity-closure/105-01-SUMMARY.md` (planning-artifact, transform)

**Rule-safe analog:** `.planning/milestones/v1.15-phases/92-milestone-evidence-validation-and-ledger-closeout/92-01-SUMMARY.md`

**Frontmatter pattern** (lines 1-7):

```markdown
---
phase: 92
plan: 01
status: complete
completed: 2026-04-23T15:00:00Z
requirements-completed:
```

Reuse the shape, not the requirement contents. For Phase 105, `requirements-completed` must be empty because active traceability maps `PARITY-01` and `CLOSE-01` to Phase 108 (`.planning/REQUIREMENTS.md` lines 20-30).

**Planning-artifact closeout pattern** (lines 27-33, 46-49):

```markdown
## Changes

- Added missing `requirements-completed` frontmatter to the v1.15 phase summaries.
- Added Nyquist-visible `VALIDATION.md` artifacts for Phases 83 through 92.
- Reconciled `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, and `.planning/STATE.md` with
  the finished v1.15 gap-closure phase set.

## Notes

- Phase 92 is a planning-artifact closeout pass; it does not change maintained runtime code.
```

Adapt for Phase 105 as a transition/supersession summary:

```markdown
requirements-completed: []
verification_status: superseded
superseded-by: [107, 108]

Phase 105 does not claim `PARITY-01` or `CLOSE-01`. The current `exact_match`
uses a normalized EMEL GGUF bridge, while direct pinned-artifact parity remains
Phase 108 scope.
```

**Rejected nearby analog:** `.planning/milestones/v1.16-phases/102-whisper-closeout-evidence/102-01-SUMMARY.md` lines 1-5 and 43-58 claimed closeout requirements while accepting `bounded_drift`. That is historical evidence only; the current audit says bounded drift is not sufficient for reopened v1.16.

### `.planning/phases/105-whisper-exact-transcript-parity-closure/105-VERIFICATION.md` (test/evidence, batch + file-I/O)

**Rule-safe analog:** `.planning/milestones/v1.15-phases/92.1-maintained-gguf-contract-parity-and-benchmark-truth-repair/92.1-VERIFICATION.md`

**Requirement evidence table pattern** (lines 16-25):

```markdown
## Requirement Evidence

| Requirement | Evidence | Status |
|-------------|----------|--------|
| SORT-01 | `tools/bench/diarization/sortformer_bench.cpp` publishes maintained metadata... | Passed |
| PRF-02 | `tools/bench/diarization_compare.py` and `tools/bench/bench_main.cpp` keep EMEL and reference lanes split... | Passed |
```

For Phase 105, invert the status truthfully:

```markdown
| Requirement | Evidence | Status |
|-------------|----------|--------|
| `PARITY-01` | Bridge compare reports `exact_match`, but EMEL uses `build/whisper_compare/normalized/whisper-tiny-q8_0-emel.gguf`; direct pinned artifact remains unsatisfied. | not claimed |
| `CLOSE-01` | Full source-backed closeout gates and final audit are Phase 108 scope. | not claimed |
```

**Commands/results pattern** (lines 27-36, 38-56):

```markdown
## Commands

- `python3 tools/bench/diarization_compare.py --output-dir build/diarization_compare_final --emel-runner build/bench_tools_ninja/bench_runner`
- `scripts/quality_gates.sh`

## Results

- The operator-facing compare workflow passed and wrote:
  - `build/diarization_compare_final/raw/emel.jsonl`
  - `build/diarization_compare_final/raw/reference.jsonl`
  - `build/diarization_compare_final/compare_summary.json`
```

Adapt with both bridge and direct-pinned checks from Phase 105 research:

```markdown
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build`
- `EMEL_WHISPER_EMEL_MODEL="$PWD/build/whisper_reference/whisper-tiny-q8_0-whispercpp.gguf" scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build`

## Results

- Bridge/default compare: `exact_match reason=ok`.
- Direct pinned artifact check: record whether it still returns `error reason=emel_lane_error`.
- Do not convert the bridge result into `PARITY-01` completion.
```

### `.planning/phases/105-whisper-exact-transcript-parity-closure/105-VALIDATION.md` (test/evidence, batch + file-I/O)

**Rule-safe analog:** existing `.planning/phases/105-whisper-exact-transcript-parity-closure/105-VALIDATION.md`

**Existing Phase 105 preconditions** (lines 16-22):

```markdown
## Completion Preconditions

- [ ] At least one phase SUMMARY.md exists
- [ ] Phase VERIFICATION.md exists
- [ ] ROADMAP / STATE represent Phase 105 truthfully as transition or supersession work
- [ ] `nyquist_compliant: true` is never set from frontmatter alone
```

Keep this structure. It already matches the Phase 105 transition scope.

**Rule review pattern** (lines 25-33):

```markdown
## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | yes | Must not claim direct pinned-artifact parity when EMEL and reference consume different artifacts; maintained-path claims require source-backed live code tracing. |
| `docs/rules/sml.rules.md` | yes | Phase 105 should not introduce SML action/guard changes; any discovered dispatch allocation remains Phase 107 scope. |
| `docs/rules/cpp.rules.md` | yes | Verification commands must avoid treating build artifacts as source truth without provenance checks. |
```

**Per-task verification map pattern** (lines 60-68):

```markdown
| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 105-01-01 | 01 | 1 | Phase 105 transition | T-105-01 | Phase 105 does not falsely claim PARITY-01/CLOSE-01 completion | artifact | `rg -n "requirements-completed|superseded|PARITY-01|CLOSE-01" .planning/phases/105-whisper-exact-transcript-parity-closure` | yes | pending |
| 105-01-02 | 01 | 1 | Phase 105 transition | T-105-02 | Evidence records bridge exact-match and direct pinned failure separately | bench/artifact | `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` plus direct pinned check | yes | pending |
```

**Completion-state analog:** `.planning/milestones/v1.13-phases/75-comparability-verdict-and-single-lane-publication-repair/75-VALIDATION.md` lines 12-27 and 62-72 show the same sections after they have passed. Only set `nyquist_compliant: true` after the Phase 105 summary, verification, ROADMAP/STATE checks, and bridge/direct evidence are recorded.

### `.planning/ROADMAP.md` (config/ledger, transform)

**Rule-safe analog:** current Phase 106/108 roadmap block.

**Current handoff pattern** (lines 198-205 and 220-230):

```markdown
### Phase 106: Reopened Whisper Evidence Repair

**Goal:** Close audit artifact and planning-state gaps for reopened Phases 103-105.
**Requirements:** REOPEN-01, SPEECH-01.
**Success Criteria**:
1. ROADMAP.md and STATE.md no longer contradict the actual reopened v1.16 readiness state.
2. Phase 103 and Phase 104 have source-backed VERIFICATION and VALIDATION artifacts.
3. Phase 105's superseded or remaining scope is represented truthfully before closeout continues.

### Phase 108: Pinned Whisper Artifact Parity Closeout

**Goal:** Prove exact transcript parity through the maintained EMEL runtime path against the pinned
Phase 99 `whisper.cpp` audio/model pair.
**Requirements:** PARITY-01, CLOSE-01.
```

Planner should update Phase 105 wording to transition/supersession truth, not direct parity completion. Leave `PARITY-01` and `CLOSE-01` mapped to Phase 108 unless the user explicitly remaps requirements.

**Current conflict to repair:** `.planning/ROADMAP.md` lines 187-196 still describe Phase 105 as owning direct exact parity and closeout gates, while `.planning/REQUIREMENTS.md` lines 20-30 maps `PARITY-01` and `CLOSE-01` to Phase 108.

### `.planning/STATE.md` (config/ledger, transform)

**Rule-safe analog:** current state file provides the fields to update, but its Phase 105 claim must be repaired.

**Stale claim to replace** (lines 29-44):

```markdown
Phase: 105
Plan: completed
Status: `v1.16` ARM Whisper GGUF Parity And Performance is reopened...
Phase `105` now closes the parity gap: the compare bench normalizes the pinned whisper.cpp model artifact into a generated
EMEL GGUF...
and reports exact transcript match for the configured short-context greedy parity lane.
```

**Evidence wording to preserve but qualify** (lines 76-82):

```markdown
Current evidence:
  `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` reports
  `exact_match reason=ok`. EMEL emits `[C]` from the generated normalized GGUF...
  The compare summary records source model SHA
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984` and normalized GGUF SHA
  `9b4be1aa866075c0515319730fffbc2248fd51676428eb8a53a4cd9d3e6cefba`.
```

Adapt as:

```markdown
Phase 105 is transition/evidence cleanup. The bridge exact-match is recorded, but it is not direct
pinned-artifact parity. Final `PARITY-01`/`CLOSE-01` closeout remains Phase 108.
```

### `build/whisper_compare/summary.json` (evidence artifact, batch + file-I/O)

**Rule-safe analog:** current summary artifact and compare writer.

**Bridge evidence excerpt** (`build/whisper_compare/summary.json` lines 1-3, 29-50, 52-67):

```json
{
  "compare_group": "whisper/tiny/q8_0/phase99_440hz_16khz_mono",
  "comparison_status": "exact_match",
  "emel": {
    "model_path": ".../build/whisper_compare/normalized/whisper-tiny-q8_0-emel.gguf",
    "model_sha256": "9b4be1aa866075c0515319730fffbc2248fd51676428eb8a53a4cd9d3e6cefba"
  },
  "model_normalization": {
    "normalizer": "tools/bench/whisper_normalize_model.py",
    "source_sha256": "9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984"
  },
  "reason": "ok",
  "reference": {
    "model_sha256": "9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984"
  }
}
```

**Compare classification source** (`tools/bench/whisper_compare.py` lines 235-258):

```python
status = "exact_match"
reason = "ok"
if emel_record.get("record_type") == "error":
  status = "error"
  reason = "emel_lane_error"
elif reference_record.get("record_type") == "error":
  status = "error"
  reason = "reference_lane_error"
elif emel_record.get("output_checksum") != reference_record.get("output_checksum"):
  status = "bounded_drift"
  reason = "transcript_mismatch"
```

Use this artifact as evidence input only. It does not satisfy direct pinned-artifact parity while `model_normalization` is present and EMEL/reference model SHAs differ.

### `build/whisper_compare/normalized/manifest.json` (evidence artifact, file-I/O)

**Rule-safe analog:** `scripts/bench_whisper_compare.sh`

**Normalizer invocation** (lines 81-92):

```bash
model_path="${EMEL_WHISPER_REFERENCE_MODEL:-$ARTIFACT_DIR/whisper-tiny-q8_0-whispercpp.gguf}"
emel_model="${EMEL_WHISPER_EMEL_MODEL:-}"

if [[ -z "$emel_model" ]]; then
  emel_model="$OUTPUT_DIR/normalized/whisper-tiny-q8_0-emel.gguf"
  python3 "$ROOT_DIR/tools/bench/whisper_normalize_model.py" \
    --source "$model_path" \
    --output "$emel_model" \
    --manifest "$OUTPUT_DIR/normalized/manifest.json"
fi
```

This is a bridge-detection pattern, not a maintained-runtime parity pattern.

## Shared Patterns

### Truthful Requirements Ownership

**Source:** `.planning/REQUIREMENTS.md` lines 20-30

```markdown
| Requirement | Phase | Status |
|-------------|-------|--------|
| PARITY-01 | 108 | Pending |
| CLOSE-01 | 108 | Pending |
```

**Apply to:** `105-01-SUMMARY.md`, `105-VERIFICATION.md`, `105-VALIDATION.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`.

### Rule-Safe Closeout Wording

**Source:** `AGENTS.md` lines 319-340 and 353-378

Pattern to apply in prose:

```markdown
Maintained-path parity is unsatisfied until live codepaths trace from the pinned fixture/model
through EMEL-owned loader/runtime/compare entrypoints. Tool-only bridges and generated artifacts
must be identified as bridges, not source-backed closeout proof.
```

### Validation Rule Review

**Source:** existing `105-VALIDATION.md` lines 25-33 and Phase 75 validation lines 19-27.

```markdown
| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | yes | Must not claim direct pinned-artifact parity when EMEL and reference consume different artifacts. |
| `docs/rules/sml.rules.md` | yes | Phase 105 should not introduce SML action/guard changes. |
| `docs/rules/cpp.rules.md` | yes | Verification commands must avoid treating build artifacts as source truth without provenance checks. |
```

### Evidence Pairing

**Source:** Phase 105 research lines 193-205 and current summary artifact lines 45-50.

Always record bridge and direct-pinned evidence separately:

```markdown
- Bridge/default compare: command, status, EMEL model SHA, reference model SHA, `model_normalization`.
- Direct pinned check: command with `EMEL_WHISPER_EMEL_MODEL=...whispercpp.gguf`, status, stderr/error reason.
```

## Rejected Legacy Patterns

| Source | Rejected Pattern | Rule Conflict / Reason |
|--------|------------------|------------------------|
| `.planning/milestones/v1.16-phases/102-whisper-closeout-evidence/102-01-SUMMARY.md` lines 1-5, 43-58 | Claim closeout requirements while recording `bounded_drift`. | Reopened v1.16 requires exact transcript parity; bounded drift is historical evidence only. |
| `.planning/milestones/v1.16-phases/102-whisper-closeout-evidence/102-VERIFICATION.md` lines 17-24, 28-36 | Treat full gate + bounded-drift parity wrapper as milestone-ready. | Latest audit contradicts this for reopened requirements. |
| `.planning/STATE.md` lines 41-44 and 76-82 | Say Phase 105 closes parity because normalized bridge exact-matches. | `AGENTS.md` requires source-backed maintained-path claims; generated bridge and differing SHAs must be disclosed. |
| `src/emel/speech/recognizer/route/token_sequence/actions.hpp` lines 32-42 | Allocate route buffers and child machines inside an SML action. | SML rules forbid heap allocation during dispatch; leave repair to Phase 107 unless remapped. |
| `src/emel/speech/recognizer/route/token_sequence/actions.hpp` lines 78-82 | Hardcode prompt tokens in an action as policy. | Runtime behavior/policy should be explicit in guards/transitions/contracts; Phase 107 owns policy hardening. |
| `src/emel/kernel/whisper/detail.hpp` lines 1000-1008 | Kernel-local timestamp-aware token selection as decode policy evidence. | Higher-level decode policy must be explicit; kernel helpers are not Phase 105 documentation patterns. |

## No Analog Found

| File / Evidence Need | Role | Data Flow | Reason |
|----------------------|------|-----------|--------|
| Persistent direct-pinned failure artifact for Phase 105 | evidence artifact | batch, file-I/O | Research reports the direct pinned check, but no stable Phase 105 `VERIFICATION.md` exists yet to store the command output and stderr. Planner should create it. |

## Metadata

**Analog search scope:** `.planning/phases`, `.planning/milestones`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/REQUIREMENTS.md`, `scripts/bench_whisper_compare.sh`, `tools/bench/whisper_compare.py`, `build/whisper_compare/**`, and rejected source references cited by the audit.

**Files scanned:** 25+ planning/source/evidence files via `rg`, `sed`, and numbered reads.

**Rule filter:** Existing source code was used only as evidence of current behavior and rejected legacy deviations. The reusable patterns are planning/evidence artifact patterns that do not widen runtime claims or introduce SML/C++ implementation changes.

**Pattern extraction date:** 2026-04-27
