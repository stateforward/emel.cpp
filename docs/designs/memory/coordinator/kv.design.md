# memory/coordinator/kv architecture design (draft)

this document defines the kv-specific memory coordinator. it orchestrates kv lifecycle phases and
delegates storage semantics to kv/cache (or a future kv/storage).

## role
- coordinate kv memory maintenance and batch preparation.
- map lifecycle outcomes into status codes for generator.

## responsibilities
- run update/maintenance phase (shift/copy/optimize).
- run batch preparation phase (slot planning using plan steps).
- expose status: success, no_update, failed_prepare, failed_update.


## open questions
- should kv/cache be demoted to storage-only (kv/storage) with no SM?
- where should contiguous-slot planning policy live?
