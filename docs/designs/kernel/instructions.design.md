# kernel/instructions architecture design (draft)

this document defines kernel/instructions. it captures the device-specific execution
instructions derived from a graph and kernel policy.

## role
- represent a prepared execution plan for a graph on a specific device.
- hold scratch sizing and scheduling metadata for execution reuse.

## events (draft)
- `event::prepare_instructions` inputs: graph signature (node list + shapes + op types), kernel
  execution policy (thread count, stream handles, scratch limits).
- `events::prepare_instructions_done` outputs: `kernel::instructions` (opaque to callers outside
  the kernel domain).
- `events::prepare_instructions_error` outputs: error_out.

## responsibilities
- encode per-op task counts and work ranges as needed by the kernel.
- record scratch buffer size requirements.
- describe subgraph boundaries for device execution.
- provide an instruction list with stable indices for per-instruction execution.

## instruction shape (draft)
- `kernel::instruction` fields:
  - `opcode` (ggml_op).
  - `subop` (ggml_unary_op / ggml_glu_op / ggml_op_pool when applicable).
  - operand handles (`src0`, `src1`, `src2`) and destination handle.
  - data bindings (buffer pointers/offsets for each operand).
  - tensor shape/stride metadata needed by the kernel.
  - `work_range` (task index range for this instruction).

## opcode list (draft)
- instructions carry an `opcode` drawn from ggml_op. current opcodes include:
  - `GGML_OP_DUP`, `GGML_OP_ADD`, `GGML_OP_ADD_ID`, `GGML_OP_ADD1`, `GGML_OP_ACC`,
    `GGML_OP_SUB`, `GGML_OP_MUL`, `GGML_OP_DIV`, `GGML_OP_SQR`, `GGML_OP_SQRT`, `GGML_OP_LOG`,
    `GGML_OP_SIN`, `GGML_OP_COS`, `GGML_OP_SUM`, `GGML_OP_SUM_ROWS`, `GGML_OP_CUMSUM`,
    `GGML_OP_MEAN`, `GGML_OP_ARGMAX`, `GGML_OP_COUNT_EQUAL`, `GGML_OP_REPEAT`,
    `GGML_OP_REPEAT_BACK`, `GGML_OP_CONCAT`, `GGML_OP_SILU_BACK`, `GGML_OP_NORM`,
    `GGML_OP_RMS_NORM`, `GGML_OP_RMS_NORM_BACK`, `GGML_OP_GROUP_NORM`, `GGML_OP_L2_NORM`,
    `GGML_OP_MUL_MAT`, `GGML_OP_MUL_MAT_ID`, `GGML_OP_OUT_PROD`, `GGML_OP_SCALE`, `GGML_OP_SET`,
    `GGML_OP_CPY`, `GGML_OP_CONT`, `GGML_OP_RESHAPE`, `GGML_OP_VIEW`, `GGML_OP_PERMUTE`,
    `GGML_OP_TRANSPOSE`, `GGML_OP_GET_ROWS`, `GGML_OP_GET_ROWS_BACK`, `GGML_OP_SET_ROWS`,
    `GGML_OP_DIAG`, `GGML_OP_DIAG_MASK_INF`, `GGML_OP_DIAG_MASK_ZERO`, `GGML_OP_SOFT_MAX`,
    `GGML_OP_SOFT_MAX_BACK`, `GGML_OP_ROPE`, `GGML_OP_ROPE_BACK`, `GGML_OP_CLAMP`,
    `GGML_OP_CONV_TRANSPOSE_1D`, `GGML_OP_IM2COL`, `GGML_OP_IM2COL_BACK`, `GGML_OP_IM2COL_3D`,
    `GGML_OP_CONV_2D`, `GGML_OP_CONV_3D`, `GGML_OP_CONV_2D_DW`, `GGML_OP_CONV_TRANSPOSE_2D`,
    `GGML_OP_POOL_1D`, `GGML_OP_POOL_2D`, `GGML_OP_POOL_2D_BACK`, `GGML_OP_UPSCALE`,
    `GGML_OP_PAD`, `GGML_OP_PAD_REFLECT_1D`, `GGML_OP_ROLL`, `GGML_OP_ARANGE`,
    `GGML_OP_TIMESTEP_EMBEDDING`, `GGML_OP_ARGSORT`, `GGML_OP_TOP_K`, `GGML_OP_LEAKY_RELU`,
    `GGML_OP_TRI`, `GGML_OP_FILL`, `GGML_OP_FLASH_ATTN_EXT`, `GGML_OP_FLASH_ATTN_BACK`,
    `GGML_OP_SSM_CONV`, `GGML_OP_SSM_SCAN`, `GGML_OP_WIN_PART`, `GGML_OP_WIN_UNPART`,
    `GGML_OP_GET_REL_POS`, `GGML_OP_ADD_REL_POS`, `GGML_OP_RWKV_WKV6`,
    `GGML_OP_GATED_LINEAR_ATTN`, `GGML_OP_RWKV_WKV7`, `GGML_OP_SOLVE_TRI`, `GGML_OP_UNARY`,
    `GGML_OP_MAP_CUSTOM1`, `GGML_OP_MAP_CUSTOM2`, `GGML_OP_MAP_CUSTOM3`, `GGML_OP_CUSTOM`,
    `GGML_OP_CROSS_ENTROPY_LOSS`, `GGML_OP_CROSS_ENTROPY_LOSS_BACK`,
    `GGML_OP_OPT_STEP_ADAMW`, `GGML_OP_OPT_STEP_SGD`, `GGML_OP_GLU`.

## notes
- `GGML_OP_NONE` and `GGML_OP_COUNT` are sentinels and are not emitted as instructions.
- pool instructions use `ggml_op_pool` parameters.

## notes
- instructions are device-specific and may be cached by `kernel::any`.
- instructions are not exposed to graph or generator domains.
