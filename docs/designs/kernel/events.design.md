# kernel events architecture design (draft)

this document defines the kernel domain events. each graph op is a compile-time typed
event dispatched to the device via `make_dispatch_table`. unsupported ops naturally hit
`sml::unexpected_event` in the device variant's transition table.

## event shape (common)
all opcode events carry the same operand payload:
- destination handle (tensor metadata + data pointer/offset).
- source operand handles (`src0`, `src1`, `src2`; nullptr when unused).
- tensor shape/stride metadata needed by the kernel.

## scheduling events

### trigger
- `event::schedule` — graph/processor hands a bound graph to kernel/any.
  inputs: bound `graph`, kernel execution policy.

### outcomes
- `events::schedule_done` — outputs written in-place to bound buffers, status.
- `events::schedule_error` — error_out.

## opcode events

kernel/any walks the graph nodes and dispatches one opcode event per node to
`kernel::device::any` via `sml::utility::make_dispatch_table`. the runtime opcode ID
from the graph node selects the compile-time event type.

each device variant's transition table lists the opcodes it supports. unsupported
opcodes hit `sml::unexpected_event` and route to the error state.

### arithmetic
- `op::dup`, `op::add`, `op::add_id`, `op::add1`, `op::acc`.
- `op::sub`, `op::mul`, `op::div`.
- `op::sqr`, `op::sqrt`, `op::log`, `op::sin`, `op::cos`.
- `op::sum`, `op::sum_rows`, `op::cumsum`, `op::mean`.
- `op::argmax`, `op::count_equal`.

### tensor manipulation
- `op::repeat`, `op::repeat_back`, `op::concat`.
- `op::scale`, `op::set`, `op::cpy`, `op::cont`.
- `op::reshape`, `op::view`, `op::permute`, `op::transpose`.
- `op::get_rows`, `op::get_rows_back`, `op::set_rows`.
- `op::diag`, `op::diag_mask_inf`, `op::diag_mask_zero`.
- `op::pad`, `op::pad_reflect_1d`, `op::roll`.
- `op::arange`, `op::fill`, `op::tri`.
- `op::clamp`.

### normalization
- `op::norm`, `op::rms_norm`, `op::rms_norm_back`.
- `op::group_norm`, `op::l2_norm`.

### matrix operations
- `op::mul_mat`, `op::mul_mat_id`, `op::out_prod`.

### activation and softmax
- `op::soft_max`, `op::soft_max_back`, `op::silu_back`.
- `op::leaky_relu`.
- `op::unary` (sub-dispatches: abs, sgn, neg, step, tanh, elu, relu, sigmoid,
  gelu, gelu_erf, gelu_quick, silu, xielu, etc.).
- `op::glu` (sub-dispatches: reglu, geglu, swiglu, swiglu_oai, geglu_erf,
  geglu_quick).

### positional encoding
- `op::rope`, `op::rope_back`.
- `op::timestep_embedding`.

### attention
- `op::flash_attn_ext`, `op::flash_attn_back`.
- `op::get_rel_pos`, `op::add_rel_pos`.

### convolution and pooling
- `op::conv_transpose_1d`, `op::conv_2d`, `op::conv_3d`.
- `op::conv_2d_dw`, `op::conv_transpose_2d`.
- `op::im2col`, `op::im2col_back`, `op::im2col_3d`.
- `op::pool_1d`, `op::pool_2d`, `op::pool_2d_back`.
- `op::upscale`.

### recurrent
- `op::ssm_conv`, `op::ssm_scan`.
- `op::rwkv_wkv6`, `op::rwkv_wkv7`.
- `op::gated_linear_attn`.

### windowed
- `op::win_part`, `op::win_unpart`.

### sorting and selection
- `op::argsort`, `op::top_k`.

### linear algebra
- `op::solve_tri`.

### custom
- `op::map_custom1`, `op::map_custom2`, `op::map_custom3`, `op::custom`.

### training (future)
- `op::cross_entropy_loss`, `op::cross_entropy_loss_back`.
- `op::opt_step_adamw`, `op::opt_step_sgd`.

## subop handling
- `op::unary` and `op::glu` carry a subop field that selects the specific function.
- `op::pool_1d` and `op::pool_2d` carry `pool_type` parameters.
- subop dispatch is handled inside the action, not as separate SML events.

## dispatch table
- `sml::utility::make_dispatch_table` bridges runtime opcode ID to compile-time event
  type.
- opcode ID range: `[op::dup, op::glu]` (excluding sentinels `NONE` and `COUNT`).
- kernel/any validates opcode ID range before indexing the dispatch table.

## notes
- opcode events use `emel::kernel::op` namespace, not `ggml_op` identifiers.
- all opcode events are trivially copyable and allocation-free.
