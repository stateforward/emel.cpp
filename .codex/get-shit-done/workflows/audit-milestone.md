<purpose>
Verify milestone achieved its definition of done by aggregating phase verifications, checking cross-phase integration, and assessing requirements coverage. Reads existing VERIFICATION.md files (phases already verified during execute-phase), aggregates tech debt and deferred gaps, then spawns integration checker for cross-phase wiring.
</purpose>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
</required_reading>

<available_agent_types>
Valid GSD subagent types (use exact names — do not fall back to 'general-purpose'):
- gsd-integration-checker — Checks cross-phase integration
</available_agent_types>

<process>

## 0. Initialize Milestone Context

```bash
INIT=$(node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" init milestone-op)
if [[ "$INIT" == @file:* ]]; then INIT=$(cat "${INIT#@file:}"); fi
AGENT_SKILLS_CHECKER=$(node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" agent-skills gsd-integration-checker 2>/dev/null)
```

Extract from init JSON: `milestone_version`, `milestone_name`, `phase_count`, `completed_phases`, `commit_docs`.

Resolve integration checker model:
```bash
integration_checker_model=$(node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" resolve-model gsd-integration-checker --raw)
```

## 1. Determine Milestone Scope

```bash
# Get phases in milestone (sorted numerically, handles decimals)
node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" phases list
```

- Parse version from arguments or detect current from ROADMAP.md
- Identify all phase directories in scope
- Cross-check each in-scope phase against ROADMAP.md and STATE.md:
  - If ROADMAP.md still shows the phase unchecked / incomplete, flag `incomplete phase` — blocker
  - If STATE.md says the milestone or phase is still in progress, do not treat reopened closeout
    phases as complete evidence just because a planning artifact exists
- Extract milestone definition of done from ROADMAP.md
- Extract requirements mapped to this milestone from REQUIREMENTS.md

## 2. Read All Phase Verifications

For each phase directory, read SUMMARY.md, VERIFICATION.md, and VALIDATION.md when present:

```bash
# For each phase, use find-phase to resolve the directory (handles archived phases)
PHASE_INFO=$(node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" find-phase 01 --raw)
# Extract directory from JSON, then read VERIFICATION.md from that directory
# Repeat for each phase number from ROADMAP.md
```

From each phase's artifacts, extract:
- **SUMMARY present/missing:** required execution artifact
- **Status:** passed | gaps_found
- **Critical gaps:** (if any — these are blockers)
- **Non-critical gaps:** tech debt, deferred items, warnings
- **Anti-patterns found:** TODOs, stubs, placeholders
- **Requirements coverage:** which requirements satisfied/blocked
- **VALIDATION evidence:** frontmatter, commands, audit trail, rule-compliance notes

If a phase is missing SUMMARY.md, flag it as "unexecuted phase artifact gap" — blocker.
If a phase is missing VERIFICATION.md, flag it as "unverified phase" — blocker.
If ROADMAP.md / STATE.md still show the phase incomplete, treat the phase as incomplete even if
PLAN.md or VALIDATION.md exists.

## 3. Spawn Integration Checker

With phase context collected:

Extract `MILESTONE_REQ_IDS` from REQUIREMENTS.md traceability table — all REQ-IDs assigned to phases in this milestone.

```
Task(
  prompt="<required_reading>
./AGENTS.md
./docs/rules/sml.rules.md
./docs/rules/cpp.rules.md
</required_reading>

Check cross-phase integration and E2E flows.

Phases: {phase_dirs}
Phase exports: {from SUMMARYs}
API routes: {routes created}

Milestone Requirements:
{MILESTONE_REQ_IDS — list each REQ-ID with description and assigned phase}

MUST map each integration finding to affected requirement IDs where applicable.
MUST call out any contradiction where a phase is claimed validated or complete without matching
SUMMARY.md / VERIFICATION.md / roadmap-state evidence.
MUST call out any rule conflict against AGENTS.md, docs/rules/sml.rules.md, or docs/rules/cpp.rules.md
that affects milestone readiness.

Verify cross-phase wiring and E2E user flows.
${AGENT_SKILLS_CHECKER}",
  subagent_type="gsd-integration-checker",
  model="{integration_checker_model}"
)
```

## 4. Collect Results

Combine:
- Phase-level gaps and tech debt (from step 2)
- Integration checker's report (wiring gaps, broken flows)

## 5. Check Requirements Coverage (3-Source Cross-Reference)

MUST cross-reference three independent sources for each requirement:

### 5a. Parse REQUIREMENTS.md Traceability Table

Extract all REQ-IDs mapped to milestone phases from the traceability table:
- Requirement ID, description, assigned phase, current status, checked-off state (`[x]` vs `[ ]`)

### 5b. Parse Phase VERIFICATION.md Requirements Tables

For each phase's VERIFICATION.md, extract the expanded requirements table:
- Requirement | Source Plan | Description | Status | Evidence
- Map each entry back to its REQ-ID

### 5c. Extract SUMMARY.md Frontmatter Cross-Check

For each phase's SUMMARY.md, extract `requirements-completed` from YAML frontmatter:
```bash
for summary in .planning/phases/*-*/*-SUMMARY.md; do
  [ -e "$summary" ] || continue
  node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" summary-extract "$summary" --fields requirements_completed --pick requirements_completed
done
```

### 5d. Status Determination Matrix

For each REQ-ID, determine status using all three sources:

| VERIFICATION.md Status | SUMMARY Frontmatter | REQUIREMENTS.md | → Final Status |
|------------------------|---------------------|-----------------|----------------|
| passed                 | listed              | `[x]`           | **satisfied**  |
| passed                 | listed              | `[ ]`           | **satisfied** (update checkbox) |
| passed                 | missing             | any             | **partial** (verify manually) |
| gaps_found             | any                 | any             | **unsatisfied** |
| missing                | listed              | any             | **partial** (verification gap) |
| missing                | missing             | any             | **unsatisfied** |

### 5e. FAIL Gate and Orphan Detection

**REQUIRED:** Any `unsatisfied` requirement MUST force `gaps_found` status on the milestone audit.

**Orphan detection:** Requirements present in REQUIREMENTS.md traceability table but absent from ALL phase VERIFICATION.md files MUST be flagged as orphaned. Orphaned requirements are treated as `unsatisfied` — they were assigned but never verified by any phase.

## 5.5. Nyquist Compliance Verification

Skip if `workflow.nyquist_validation` is explicitly `false` (absent = enabled).

```bash
NYQUIST_CONFIG=$(node "/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.codex/get-shit-done/bin/gsd-tools.cjs" config-get workflow.nyquist_validation --raw 2>/dev/null)
```

If `false`: skip entirely.

For each phase directory, check `*-VALIDATION.md`. If exists, parse frontmatter
(`nyquist_compliant`, `wave_0_complete`) and verify supporting evidence.

**Never treat frontmatter alone as Nyquist compliance.** A phase can only count as compliant when:
- `SUMMARY.md` exists for the phase
- `VERIFICATION.md` exists for the phase
- ROADMAP.md and STATE.md do not still mark the phase as incomplete / pending work
- VALIDATION.md contains executable verification commands
- VALIDATION.md records rule-compliance evidence or explicitly says no rule violations were found
- VALIDATION.md has no unresolved escalations, manual-only blockers, or contradictory audit notes

Classify per phase:

| Status | Condition |
|--------|-----------|
| COMPLIANT | VALIDATION evidence passes all gates above |
| PARTIAL | VALIDATION.md exists, but `nyquist_compliant: false`, tasks are red/pending, or evidence is incomplete |
| INVALID | VALIDATION.md claims compliance but supporting completion/rule evidence is missing or contradicted |
| MISSING | No VALIDATION.md |

`INVALID` phases MUST force milestone status `gaps_found`.

Add to audit YAML: `nyquist: { compliant_phases, partial_phases, invalid_phases, missing_phases, overall }`

Discovery only — never auto-calls `$gsd-validate-phase`.

## 6. Aggregate into v{version}-MILESTONE-AUDIT.md

Create `.planning/v{version}-v{version}-MILESTONE-AUDIT.md` with:

```yaml
---
milestone: {version}
audited: {timestamp}
status: passed | gaps_found | tech_debt
scores:
  requirements: N/M
  phases: N/M
  integration: N/M
  flows: N/M
gaps:  # Critical blockers
  requirements:
    - id: "{REQ-ID}"
      status: "unsatisfied | partial | orphaned"
      phase: "{assigned phase}"
      claimed_by_plans: ["{plan files that reference this requirement}"]
      completed_by_plans: ["{plan files whose SUMMARY marks it complete}"]
      verification_status: "passed | gaps_found | missing | orphaned"
      evidence: "{specific evidence or lack thereof}"
  integration: [...]
  flows: [...]
  phase_artifacts: [...]
tech_debt:  # Non-critical, deferred
  - phase: 01-auth
    items:
      - "TODO: add rate limiting"
      - "Warning: no password strength validation"
  - phase: 03-dashboard
    items:
      - "Deferred: mobile responsive layout"
---
```

Plus full markdown report with tables for requirements, phases, integration, tech debt.

**Status values:**
- `passed` — all requirements met, no critical gaps, minimal tech debt
- `gaps_found` — critical blockers exist
- `tech_debt` — no blockers but accumulated deferred items need review

Phase-artifact blockers include:
- roadmap/state say the phase is still incomplete
- SUMMARY.md missing
- VERIFICATION.md missing
- VALIDATION.md claims compliance without supporting evidence

## 7. Present Results

Route by status (see `<offer_next>`).

</process>

<offer_next>
Output this markdown directly (not as a code block). Route based on status:

---

**If passed:**

## ✓ Milestone {version} — Audit Passed

**Score:** {N}/{M} requirements satisfied
**Report:** .planning/v{version}-MILESTONE-AUDIT.md

All requirements covered. Cross-phase integration verified. E2E flows complete.

───────────────────────────────────────────────────────────────

## ▶ Next Up

**Complete milestone** — archive and tag

$gsd-complete-milestone {version}

───────────────────────────────────────────────────────────────

---

**If gaps_found:**

## ⚠ Milestone {version} — Gaps Found

**Score:** {N}/{M} requirements satisfied
**Report:** .planning/v{version}-MILESTONE-AUDIT.md

### Unsatisfied Requirements

{For each unsatisfied requirement:}
- **{REQ-ID}: {description}** (Phase {X})
  - {reason}

### Cross-Phase Issues

{For each integration gap:}
- **{from} → {to}:** {issue}

### Broken Flows

{For each flow gap:}
- **{flow name}:** breaks at {step}

### Nyquist Coverage

| Phase | VALIDATION.md | Compliant | Action |
|-------|---------------|-----------|--------|
| {phase} | exists/missing | true/false/partial | `$gsd-validate-phase {N}` |

Phases needing validation: run `$gsd-validate-phase {N}` for each flagged phase.

───────────────────────────────────────────────────────────────

## ▶ Next Up

**Plan gap closure** — create phases to complete milestone

$gsd-plan-milestone-gaps

───────────────────────────────────────────────────────────────

**Also available:**
- cat .planning/v{version}-MILESTONE-AUDIT.md — see full report
- $gsd-complete-milestone {version} — proceed anyway (accept tech debt)

───────────────────────────────────────────────────────────────

---

**If tech_debt (no blockers but accumulated debt):**

## ⚡ Milestone {version} — Tech Debt Review

**Score:** {N}/{M} requirements satisfied
**Report:** .planning/v{version}-MILESTONE-AUDIT.md

All requirements met. No critical blockers. Accumulated tech debt needs review.

### Tech Debt by Phase

{For each phase with debt:}
**Phase {X}: {name}**
- {item 1}
- {item 2}

### Total: {N} items across {M} phases

───────────────────────────────────────────────────────────────

## ▶ Options

**A. Complete milestone** — accept debt, track in backlog

$gsd-complete-milestone {version}

**B. Plan cleanup phase** — address debt before completing

$gsd-plan-milestone-gaps

───────────────────────────────────────────────────────────────
</offer_next>

<success_criteria>
- [ ] Milestone scope identified
- [ ] All phase VERIFICATION.md files read
- [ ] SUMMARY.md `requirements-completed` frontmatter extracted for each phase
- [ ] REQUIREMENTS.md traceability table parsed for all milestone REQ-IDs
- [ ] 3-source cross-reference completed (VERIFICATION + SUMMARY + traceability)
- [ ] Orphaned requirements detected (in traceability but absent from all VERIFICATIONs)
- [ ] Tech debt and deferred gaps aggregated
- [ ] Integration checker spawned with milestone requirement IDs
- [ ] v{version}-MILESTONE-AUDIT.md created with structured requirement gap objects
- [ ] FAIL gate enforced — any unsatisfied requirement forces gaps_found status
- [ ] Nyquist compliance scanned for all milestone phases (if enabled)
- [ ] Missing VALIDATION.md phases flagged with validate-phase suggestion
- [ ] Results presented with actionable next steps
</success_criteria>
