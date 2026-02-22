# memory/coordinator/recurrent architecture design (draft)

this document defines the recurrent-memory coordinator. it orchestrates recurrent state lifecycle
phases and delegates storage semantics to the recurrent memory actor.

## role
- coordinate recurrent memory maintenance and batch preparation.
- expose status for generator orchestration.

## responsibilities
- run update/maintenance phase (shift/copy/optimize equivalents).
- run batch preparation phase for recurrent memory.
- map statuses: success, no_update, failed_prepare, failed_update.


## open questions
- how should recurrent memory planning interface with batch plans?
