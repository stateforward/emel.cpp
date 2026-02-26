#pragma once

/*
domain doc: kernel events

this header is documentation-only for kernel event contracts and dispatch behavior.
`kernel/events` is not a standalone actor.

the kernel domain uses compile-time typed `op::*` events handled via transition tables in concrete
kernel machines (`kernel/x86_64`, `kernel/aarch64`, `kernel/cuda`, etc). unsupported ops route
through `sml::unexpected_event` at the active variant boundary and can be forwarded by
`kernel::any` fallback policy.

event shape (common):
- destination handle (tensor metadata + data pointer/offset).
- source operand handles (`src0`, `src1`, `src2`; null when unused).
- tensor shape/stride metadata needed by the kernel action.

opcode families:
- arithmetic: `dup`, `add`, `sub`, `mul`, `div`, `sqrt`, `log`, `sum`, `mean`, etc.
- tensor ops: `repeat`, `concat`, `cpy`, `reshape`, `permute`, `transpose`, etc.
- normalization: `norm`, `rms_norm`, `group_norm`, `l2_norm`.
- matrix ops: `mul_mat`, `mul_mat_id`, `out_prod`.
- activations: `soft_max`, `silu_back`, `unary`, `glu`, etc.
- positional/attention: `rope`, `flash_attn_ext`, `add_rel_pos`, etc.
- convolution/pooling: `conv_*`, `im2col*`, `pool_*`, `upscale`.
- recurrent/windowed/sort/linalg/custom and future training ops.

subop notes:
- `op::unary` and `op::glu` carry subop selectors.
- `op::pool_1d` and `op::pool_2d` carry pool type parameters.
- subop resolution is action-local (not separate SML events).
*/

namespace emel::kernel::event {

struct scaffold {};

}  // namespace emel::kernel::event
