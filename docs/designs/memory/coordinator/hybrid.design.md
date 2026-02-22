# memory/coordinator/hybrid architecture design (draft)

this document defines the hybrid-memory coordinator. it composes kv + recurrent coordinators and
exposes a unified lifecycle surface.

## role
- coordinate hybrid memory (kv + recurrent) as a single lifecycle.
- combine statuses into one outcome for generator.

## responsibilities
- run update/maintenance for both kv and recurrent.
- run batch preparation for both memory types.
- combine status codes deterministically (fail-fast, no_update if both no_update).


## open questions
- should hybrid coordinator own both sub-coordinators or accept injected handles?
