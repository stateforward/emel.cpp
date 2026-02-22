# memory/kv architecture design (draft)

this document defines the kv memory actor (storage semantics), separate from the coordinator.

## role
- own the kv storage layout and per-sequence state.
- provide slot planning helpers and apply/rollback semantics.

## responsibilities
- maintain kv cells and sequence metadata.
- plan slots for plan steps (contiguous/non-contiguous policy).
- apply/rollback step placement.
- implement sequence ops (rm/cp/keep/add/div) and state io.


## open questions
- should this be a pure storage object (no SM) or a minimal SM wrapper?
