# Problem Statement: Catastrophic Stack Overflow in Memory Subsystem `view()` 

**Background:**
Recent enhancements to `src/emel/memory/kv/sm.hpp`, `src/emel/memory/recurrent/sm.hpp`, and `src/emel/memory/hybrid/sm.hpp` correctly shifted the `view::snapshot` structure out of the mutable operational context into a dynamic `std::unique_ptr<view::snapshot> snapshot_`. 

This was intended to prevent large allocations on the stack, which is critical for meeting embedded and "Aerospace Grade" hard real-time execution limits. 

**The Bug:**
Within the fallback error-handling branch of the `view()` wrapper method in these files, there is a severe violation of stack invariants:

```cpp
  const view::snapshot & view() noexcept {
    // ...
    if (!try_view(*snapshot_, err)) {
      *snapshot_ = view::snapshot{}; // <--- CRITICAL STACK HAZARD
    }
    return *snapshot_;
  }
```

**Why this is Fatal:**
The `view::snapshot` structure contains a $256 \times 4096$ element `uint16_t` matrix (`sequence_kv_blocks`). This struct alone requires **~2.1 Megabytes** of contiguous memory.

When the compiler evaluates the R-value instantiation `view::snapshot{}`, it is forced to allocate that 2.1 MB structure directly on the active thread's call-stack *before* assigning those zeroes into the heap pointer. 

On macOS background threads (often limited to 512 KB), WASM, or aerospace RTOS targets (often limited to 64-128 KB stacks), this single initialization will instantly trigger a fatal Stack Overflow and segfault the process entirely. It is in direct violation of `CPP-RULES` invariants **CPPRT-041** and **CPPRT-042**.

**Required Resolution:**
The engineer must clear the existing heap-allocated memory directly *in-place* without relying on a proxy stack allocation.
1. Add an explicit `void clear() noexcept` function to `struct snapshot` (or utilize `std::memset(snapshot_.get(), 0, sizeof(view::snapshot))` if the struct is trivially verifiable).
2. Refactor all three instances of `*snapshot_ = view::snapshot{};` in `kv`, `recurrent`, and `hybrid` state machines to call this direct, zero-stack clearing mechanism.
