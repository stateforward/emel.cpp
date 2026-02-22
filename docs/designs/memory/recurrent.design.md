# memory/recurrent architecture design (draft)

this document defines recurrent memory storage semantics (non-kv), separate from the coordinator.

## role
- own recurrent state storage for models that require it (e.g., rwkv-like).
- expose sequence/state ops needed for decode orchestration.

## responsibilities
- maintain recurrent state buffers and per-sequence metadata.
- implement sequence ops and state io.
- support batch preparation helpers for recurrent layouts.


## open questions
- what is the minimal recurrent state surface needed for decoding?
