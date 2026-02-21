---
trigger: always_on
globs: src/**/*, tests/**/*
---

these rules define how to use boost.SML (boost::ext SML) to build a high-performance, real-time friendly architecture that behaves like a pure actor model while remaining synchronous run-to-completion (RTC) and using no message queue.

these rules apply to:
- boost.SML state machines (`boost::sml::sm<...>`) and their composition (composite state machines, orthogonal regions).
- synchronous dispatch only (no background workers, no mailboxes, no async buffering).

the rules assume boost.SML v1.1.x semantics as implemented in the public header and utility dispatch table.

primary sources consulted (non-exhaustive)
- docs: https://boost-ext.github.io/sml/ (introduction), https://boost-ext.github.io/sml/tutorial.html (tutorial), https://boost-ext.github.io/sml/user_guide.html (user guide), https://boost-ext.github.io/sml/benchmarks.html (benchmarks), https://boost-ext.github.io/sml/overview.html (overview)
- source code: https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml.hpp , https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml/utility/dispatch_table.hpp
- author talk (overview of design/perf goals): https://www.youtube.com/watch?v=lg3t_iact5_fw

## 2. definitions
- actor: an isolated unit owning (1) exactly one SML `sm` instance and (2) its private context, processed only via event dispatch.
- event: an immutable value (usually a trivial type) passed to `process_event`.
- RTC chain: the complete, synchronous computation triggered by one top-level dispatch call, including SML internal anonymous transitions.
- quiescence: a stable configuration where no further internal (anonymous) transitions are enabled.
- orchestrator: the external driver that calls `process_event` on actors and provides time and ordering.
- no message queue: no SML `process_queue`, no SML `defer_queue`, no user mailbox, and no “post for later” mechanism.

## 3. core invariants
1. RTC invariant: A top-level `actor.dispatch(e)` MUST return only after the machine has reached quiescence for that event (and any internal anonymous transitions).
2. no-queue invariant: actors MUST NOT enable or use SML queuing policies (`sml::process_queue`, `sml::defer_queue`) and MUST NOT implement a mailbox.
3. determinism invariant: given identical initial state, identical event sequence, and identical event payloads, the sequence of executed actions and state changes MUST be identical.
4. single-writer invariant: during any RTC chain, exactly one thread MUST be executing inside any given actor’s `process_event`.
5. allocation invariant: no dynamic allocation (heap) MUST occur during dispatch (guards/actions/entry/exit/anonymous progress).
6. bounded-work invariant: each top-level dispatch MUST have a provable upper bound on executed transitions and on total work.

## 4. event model
1. event types SHOULD be small, trivially copyable, and contain only immutable payload.
2. events MUST NOT contain owning pointers or dynamic containers (e.g., `std::string`, `std::vector`) unless a custom allocator and strict “no allocate during dispatch” enforcement is in place.
3. event dispatch SHOULD be compile-time typed (`sm.process_event(TEvent{...})`). runtime-polymorphic “base event” dispatch SHOULD be avoided.
4. if runtime event IDs are required, the system MAY use `sml::utility::make_dispatch_table` (static jump table indexed by ID) and MUST validate the ID range before indexing. see `make_dispatch_table` implementation using a static function-pointer table indexed by `(id - event_range_begin)`. (source: `dispatch_table.hpp` in SML repo.)
5. each event MUST have a single “owning” actor responsible for interpreting it. cross-actor effects MUST be explicit (see section 8).

## 5. state and context
1. state types SHOULD be “labels”, not storage. actor data MUST live in an explicit context object owned by the actor and passed into SML via dependency injection (constructor args to `sml::sm`).
2. actor context MUST have stable addresses for the lifetime of the state machine (no moves that change addresses) if references are injected.
3. construction-time dependencies order MUST NOT be relied on. the tutorial explicitly notes parameter order is “not specified”. prefer a single context aggregate or named dependency wrappers. (docs: tutorial section “create a state machine”.)
4. context mutation MUST be confined to actions (not guards) unless a guard mutation is proven to be side-effect free in terms of externally observable behavior (recommended: treat guards as pure).
5. state queries for external observers MUST be done using `visit_current_states` or `is(...)` and MUST NOT require locks in the steady state.

## 6. actions and guards
### selection and evaluation order
1. guards MUST be pure predicates of `(event, context)` and MUST return `bool`.
2. for a given (state, event), transitions are attempted in transition-table order, and the first transition whose guard returns true is taken. this follows SML’s `transitions<T, ts...>::execute` recursion: it executes `T` first and only tries `ts...` if `T` does not execute. (source: `boost/sml.hpp`, `transitions<T, ts...>`.)
3. for an external transition with entry/exit enabled, the order MUST be: guard, on-exit, state update, action, on-entry. this follows `transition<...>::execute` which calls `on_exit`, updates current state, executes action, then calls `on_entry`. (source: `boost/sml.hpp`, `transition<state<s1>, state<s2>, event<E>, G, A>::execute`.)

### real-time and determinism constraints
4. guards and actions MUST be bounded time and MUST NOT block (no I/O waits, no mutex waits, no sleeps).
5. guards and actions MUST NOT allocate. if an action MUST allocate for rare paths (e.g., error reporting), it MUST do so outside dispatch and only pass references into dispatch.
6. guards MUST NOT read wall-clock time. time MUST be provided explicitly via events (section 10).
7. actions MUST NOT contain orchestration branching or validation logic. any conditional logic that
   changes control flow (success vs error, retries, mode selection) MUST be expressed as guarded
   transitions or explicit states.
8. actions SHOULD be short. long-running work MUST be split:
   - action initiates work and transitions to a “waiting” state.
   - A later external event represents completion (still no queues).
9. actions SHOULD be `noexcept` in production builds. if exceptions are enabled, the system MUST define a hard policy for exception events and document action-throws semantics (overview page notes different semantics for guard-throws vs action-throws).
10. unexpected-event handling MUST NOT consume internal SML events. do NOT use
    `event<sml::_>` or `unexpected_event<_>` without filtering, as internal
    `boost::sml::back::internal_event` types (e.g., `anonymous`, `on_entry`, `on_exit`,
    `unexpected_event`) can match and create infinite RTC loops.
11. when modeling unexpected events, either:
    - guard wildcard handlers to exclude `boost::sml::back::internal_event`, or
    - handle only explicit external event types with `unexpected_event<T>`.

## 7. reentrancy and nested dispatch
1. an actor MUST NOT call its own `process_event` (directly or indirectly) from inside a guard/action. this prevents unbounded recursion and makes WCET analysis tractable. (motivation: `process_event` is synchronous and can be re-entered; SML users report deep call stacks if they do this.)
2. internal multi-step “microflows” within a single RTC chain MUST be modeled using anonymous transitions (eventless transitions) and/or entry actions, not by self-dispatch.
3. anonymous transition graphs MUST be acyclic or MUST have a statically provable bound on firings per top-level event. SML’s `process_event` loops internal anonymous processing to quiescence (`while (process_internal_events(anonymous{}, ...)) {}`), so cycles can create unbounded work. (source: `boost/sml.hpp`, `sm_impl::process_event` loop.)
4. cross-actor nested dispatch (A calls B synchronously) MAY be used, but MUST obey:
   - no re-entrancy into the same actor instance within a single RTC chain.
   - acyclic call graph per top-level dispatch (enforced by orchestrator stack tracking or design discipline).
5. if multi-thread calls are possible, thread-safety MUST be enforced outside SML using a single external scheduler; using `sml::thread_safe<std::mutex>` inside actors SHOULD be avoided for real-time because lock acquisition is not bounded. (docs: overview “thread safety” shows optional locking policy.)

## 8. cross-actor interaction (no queues)
### allowed interaction patterns
1. synchronous send (recommended): actor A calls `actor_b.dispatch(ev)` directly as a normal function call. this provides actor-like isolation while preserving RTC and no-queue semantics.
2. synchronous request-reply: actor A calls `actor_b.dispatch(req)` and reads results via:
   - return values from `actor_b` non-SML methods, or
   - context fields exposed as read-only snapshots after `dispatch` returns.
3. publish-subscribe (no queue): the orchestrator MAY fan out an input event to multiple actors by deterministic ordering. each actor MUST treat the event as read-only input.
4. synchronous callbacks (allowed): events MAY carry `emel::callback`-style functors for
   immediate notification within the same RTC chain (e.g., request-reply without storing
   results in context). callbacks MUST be invoked before `dispatch` returns, MUST be bounded
   and non-allocating, MUST NOT call back into `process_event`, and MUST NOT be stored in
   actor context or retained for later.

### forbidden interaction patterns
5. actors MUST NOT store “callback handles” that later re-enter `process_event` asynchronously
   (this becomes a mailbox by another name).
6. actors MUST NOT enqueue follow-up work for later (no `sml::process_queue`,
   no `sml::defer_queue`, no user mailbox).

### deterministic ordering rules
6. cross-actor calls MUST have a single total order defined by the orchestrator. the order MUST be deterministic and testable (e.g., fixed actor ID ordering).
7. if an event needs to affect multiple actors, the orchestrator MUST define whether the operation is:
   - sequential (A then B then C) with well-defined intermediate visibility, or
   - transactional (collect decisions, then commit), implemented explicitly as a two-phase protocol with no queues.

## 9. performance rules
1. actors MUST compile with optimizations appropriate for latency (`-o2` or `-o3`) and with exceptions disabled where possible. SML is designed to not use exceptions internally and can be built with `-fno-exceptions`. (docs: overview “exception safety”.)
2. the dispatch mechanism SHOULD be constant-time over states. SML provides dispatch policies including a jump table (`dispatch_table[current_state]`) using a constexpr static function-pointer array. (source: `boost/sml.hpp`, `back::policies::jump_table`.)
3. for maximum predictability, actors SHOULD:
   - avoid RTTI and virtual dispatch in guards/actions,
   - avoid type erasure and `std::function` in hot paths,
   - keep transition tables in a single translation unit to maximize inlining and compile-time resolution.
4. each actor MUST have a hard cap on:
   - maximum number of transitions executed per top-level event,
   - maximum number of anonymous transitions executed per top-level event,
   - maximum number of cross-actor synchronous calls per top-level event.
5. memory footprint SHOULD be fixed and minimal. SML benchmarks report very small “memory usage” for the state machine itself when no queues are used. (docs: benchmarks page.)

## 10. time and scheduling
1. actors MUST be driven by an external scheduler (orchestrator). actors MUST NOT create threads or timers.
2. time MUST enter the system only as explicit event payload (e.g., `tick{now_ns, dt_ns}` or `deadline_reached{...}`).
3. timeout logic MUST be modeled as:
   - store a deadline in context on entry to a waiting state,
   - on each `tick`, guard checks `now >= dead