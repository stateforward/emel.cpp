---
title: kernel operations architecture design
status: draft
---

# kernel operations architecture design

this document defines the kernel domain operations (`ops`). each graph node represents a
mathematical operation that the `graph/processor` dispatches to the hardware kernel.

## operation shape (common)
all op payloads carry the data necessary for the kernel to execute the math:
- destination handle (tensor metadata + data pointer/offset).
- source operand handles (`src0`, `src1`, `src2`; nullptr when unused).
- tensor shape/stride metadata needed by the kernel.
- op-specific parameters (e.g., axis, eps, scale).

## opcode catalog

the `graph/processor` reads the node's op type and dispatches the corresponding compute
instruction to the active hardware kernel. unsupported opcodes on a specific backend return
a fallback error, usually prompting a fallback to CPU.

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

## subop handling
- `op::unary` and `op::glu` carry a subop field that selects the specific function.
- `op::pool_1d` and `op::pool_2d` carry `pool_type` parameters.
- subop dispatch is handled internally by the kernel backend.

## notes
- opcodes use `emel::kernel::op` namespace.
- op payloads are trivially copyable and allocation-free.
