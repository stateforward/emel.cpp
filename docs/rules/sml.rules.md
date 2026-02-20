# sml.rules.md

## 1. Scope
These rules define how to use Boost.SML (Boost::ext SML) to build a high-performance, real-time friendly architecture that behaves like a pure actor model while remaining synchronous run-to-completion (RTC) and using no message queue.

These rules apply to:
- Boost.SML state machines (`boost::sml::sm<...>`) and their composition (composite state machines, orthogonal regions).
- Synchronous dispatch only (no background workers, no mailboxes, no async buffering).

The rules assume Boost.SML v1.1.x semantics as implemented in the public header and utility dispatch table.

Primary sources consulted (non-exhaustive)
- Docs: https://boost-ext.github.io/sml/ (Introduction), https://boost-ext.github.io/sml/tutorial.html (Tutorial), https://boost-ext.github.io/sml/user_guide.html (User Guide), https://boost-ext.github.io/sml/benchmarks.html (Benchmarks), https://boost-ext.github.io/sml/overview.html (Overview)
- Source code: https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml.hpp , https://github.com/boost-ext/sml/blob/v1.1.13/include/boost/sml/utility/dispatch_table.hpp
- Author talk (overview of design/perf goals): https://www.youtube.com/watch?v=Lg3tIact5Fw

## 2. Definitions
- Actor: an isolated unit owning (1) exactly one SML `sm` instance and (2) its private context, processed only via event dispatch.
- Event: an immutable value (usually a trivial type) passed to `process_event`.
- RTC chain: the complete, synchronous computation triggered by one top-level dispatch call, including SML internal anonymous transitions.
- Quiescence: a stable configuration where no further internal (anonymous) transitions are enabled.
- Orchestrator: the external driver that calls `process_event` on actors and provides time and ordering.
- No message queue: no SML `process_queue`, no SML `defer_queue`, no user mailbox, and no “post for later” mechanism.

## 3. Core invariants
1. RTC invariant: A top-level `actor.dispatch(e)` MUST return only after the machine has reached quiescence for that event (and any internal anonymous transitions).
2. No-queue invariant: Actors MUST NOT enable or use SML queuing policies (`sml::process_queue`, `sml::defer_queue`) and MUST NOT implement a mailbox.
3. Determinism invariant: Given identical initial state, identical event sequence, and identical event payloads, the sequence of executed actions and state changes MUST be identical.
4. Single-writer invariant: During any RTC chain, exactly one thread MUST be executing inside any given actor’s `process_event`.
5. Allocation invariant: No dynamic allocation (heap) MUST occur during dispatch (guards/actions/entry/exit/anonymous progress).
6. Bounded-work invariant: Each top-level dispatch MUST have a provable upper bound on executed transitions and on total work.

## 4. Event model
1. Event types SHOULD be small, trivially copyable, and contain only immutable payload.
2. Events MUST NOT contain owning pointers or dynamic containers (e.g., `std::string`, `std::vector`) unless a custom allocator and strict “no allocate during dispatch” enforcement is in place.
3. Event dispatch SHOULD be compile-time typed (`sm.process_event(TEvent{...})`). Runtime-polymorphic “base event” dispatch SHOULD be avoided.
4. If runtime event IDs are required, the system MAY use `sml::utility::make_dispatch_table` (static jump table indexed by ID) and MUST validate the ID range before indexing. See `make_dispatch_table` implementation using a static function-pointer table indexed by `(id - EventRangeBegin)`. (Source: `dispatch_table.hpp` in SML repo.)
5. Each event MUST have a single “owning” actor responsible for interpreting it. Cross-actor effects MUST be explicit (see Section 8).

## 5. State and context
1. State types SHOULD be “labels”, not storage. Actor data MUST live in an explicit context object owned by the actor and passed into SML via dependency injection (constructor args to `sml::sm`).
2. Actor context MUST have stable addresses for the lifetime of the state machine (no moves that change addresses) if references are injected.
3. Construction-time dependencies order MUST NOT be relied on. The tutorial explicitly notes parameter order is “not specified”. Prefer a single context aggregate or named dependency wrappers. (Docs: Tutorial section “Create a state machine”.)
4. Context mutation MUST be confined to actions (not guards) unless a guard mutation is proven to be side-effect free in terms of externally observable behavior (recommended: treat guards as pure).
5. State queries for external observers MUST be done using `visit_current_states` or `is(...)` and MUST NOT require locks in the steady state.

## 6. Actions and guards
### Selection and evaluation order
1. Guards MUST be pure predicates of `(event, context)` and MUST return `bool`.
2. For a given (state, event), transitions are attempted in transition-table order, and the first transition whose guard returns true is taken. This follows SML’s `transitions<T, Ts...>::execute` recursion: it executes `T` first and only tries `Ts...` if `T` does not execute. (Source: `boost/sml.hpp`, `transitions<T, Ts...>`.)
3. For an external transition with entry/exit enabled, the order MUST be: guard, on-exit, state update, action, on-entry. This follows `transition<...>::execute` which calls `on_exit`, updates current state, executes action, then calls `on_entry`. (Source: `boost/sml.hpp`, `transition<state<S1>, state<S2>, event<E>, G, A>::execute`.)

### Real-time and determinism constraints
4. Guards and actions MUST be bounded time and MUST NOT block (no I/O waits, no mutex waits, no sleeps).
5. Guards and actions MUST NOT allocate. If an action MUST allocate for rare paths (e.g., error reporting), it MUST do so outside dispatch and only pass references into dispatch.
6. Guards MUST NOT read wall-clock time. Time MUST be provided explicitly via events (Section 10).
7. Actions MUST NOT contain orchestration branching or validation logic. Any conditional logic that
   changes control flow (success vs error, retries, mode selection) MUST be expressed as guarded
   transitions or explicit states.
8. Actions SHOULD be short. Long-running work MUST be split:
   - Action initiates work and transitions to a “waiting” state.
   - A later external event represents completion (still no queues).
9. Actions SHOULD be `noexcept` in production builds. If exceptions are enabled, the system MUST define a hard policy for exception events and document action-throws semantics (Overview page notes different semantics for guard-throws vs action-throws).
10. Unexpected-event handling MUST NOT consume internal SML events. Do NOT use
    `event<sml::_>` or `unexpected_event<_>` without filtering, as internal
    `boost::sml::back::internal_event` types (e.g., `anonymous`, `on_entry`, `on_exit`,
    `unexpected_event`) can match and create infinite RTC loops.
11. When modeling unexpected events, either:
    - guard wildcard handlers to exclude `boost::sml::back::internal_event`, or
    - handle only explicit external event types with `unexpected_event<T>`.

## 7. Reentrancy and nested dispatch
1. An actor MUST NOT call its own `process_event` (directly or indirectly) from inside a guard/action. This prevents unbounded recursion and makes WCET analysis tractable. (Motivation: `process_event` is synchronous and can be re-entered; SML users report deep call stacks if they do this.)
2. Internal multi-step “microflows” within a single RTC chain MUST be modeled using anonymous transitions (eventless transitions) and/or entry actions, not by self-dispatch.
3. Anonymous transition graphs MUST be acyclic or MUST have a statically provable bound on firings per top-level event. SML’s `process_event` loops internal anonymous processing to quiescence (`while (process_internal_events(anonymous{}, ...)) {}`), so cycles can create unbounded work. (Source: `boost/sml.hpp`, `sm_impl::process_event` loop.)
4. Cross-actor nested dispatch (A calls B synchronously) MAY be used, but MUST obey:
   - No re-entrancy into the same actor instance within a single RTC chain.
   - Acyclic call graph per top-level dispatch (enforced by orchestrator stack tracking or design discipline).
5. If multi-thread calls are possible, thread-safety MUST be enforced outside SML using a single external scheduler; using `sml::thread_safe<std::mutex>` inside actors SHOULD be avoided for real-time because lock acquisition is not bounded. (Docs: Overview “Thread Safety” shows optional locking policy.)

## 8. Cross-actor interaction (no queues)
### Allowed interaction patterns
1. Synchronous send (recommended): Actor A calls `ActorB.dispatch(ev)` directly as a normal function call. This provides actor-like isolation while preserving RTC and no-queue semantics.
2. Synchronous request-reply: Actor A calls `ActorB.dispatch(req)` and reads results via:
   - return values from `ActorB` non-SML methods, or
   - context fields exposed as read-only snapshots after `dispatch` returns.
3. Publish-subscribe (no queue): The orchestrator MAY fan out an input event to multiple actors by deterministic ordering. Each actor MUST treat the event as read-only input.
4. Synchronous callbacks (allowed): Events MAY carry `emel::callback`-style functors for
   immediate notification within the same RTC chain (e.g., request-reply without storing
   results in context). Callbacks MUST be invoked before `dispatch` returns, MUST be bounded
   and non-allocating, MUST NOT call back into `process_event`, and MUST NOT be stored in
   actor context or retained for later.

### Forbidden interaction patterns
5. Actors MUST NOT store “callback handles” that later re-enter `process_event` asynchronously
   (this becomes a mailbox by another name).
6. Actors MUST NOT enqueue follow-up work for later (no `sml::process_queue`,
   no `sml::defer_queue`, no user mailbox).

### Deterministic ordering rules
6. Cross-actor calls MUST have a single total order defined by the orchestrator. The order MUST be deterministic and testable (e.g., fixed actor ID ordering).
7. If an event needs to affect multiple actors, the orchestrator MUST define whether the operation is:
   - sequential (A then B then C) with well-defined intermediate visibility, or
   - transactional (collect decisions, then commit), implemented explicitly as a two-phase protocol with no queues.

## 9. Performance rules
1. Actors MUST compile with optimizations appropriate for latency (`-O2` or `-O3`) and with exceptions disabled where possible. SML is designed to not use exceptions internally and can be built with `-fno-exceptions`. (Docs: Overview “Exception Safety”.)
2. The dispatch mechanism SHOULD be constant-time over states. SML provides dispatch policies including a jump table (`dispatch_table[current_state]`) using a constexpr static function-pointer array. (Source: `boost/sml.hpp`, `back::policies::jump_table`.)
3. For maximum predictability, actors SHOULD:
   - avoid RTTI and virtual dispatch in guards/actions,
   - avoid type erasure and `std::function` in hot paths,
   - keep transition tables in a single translation unit to maximize inlining and compile-time resolution.
4. Each actor MUST have a hard cap on:
   - maximum number of transitions executed per top-level event,
   - maximum number of anonymous transitions executed per top-level event,
   - maximum number of cross-actor synchronous calls per top-level event.
5. Memory footprint SHOULD be fixed and minimal. SML benchmarks report very small “memory usage” for the state machine itself when no queues are used. (Docs: Benchmarks page.)

## 10. Time and scheduling
1. Actors MUST be driven by an external scheduler (orchestrator). Actors MUST NOT create threads or timers.
2. Time MUST enter the system only as explicit event payload (e.g., `tick{now_ns, dt_ns}` or `deadline_reached{...}`).
3. Timeout logic MUST be modeled as:
   - store a deadline in context on entry to a waiting state,
   - on each `tick`, guard checks `now >= deadline` and transitions.
4. Each dispatch MUST have a time budget. The orchestrator SHOULD measure per-event execution time and enforce a policy:
   - hard fail (assert/terminate) in debug,
   - degrade gracefully (enter fault state) in production.
5. For determinism in simulation/replay, “now” MUST be supplied by the caller, not read from the OS clock.

## 11. Composition rules
### Hierarchy (composite state machines)
1. Composite state machines MAY be used to represent hierarchical actors or subcomponents, but each submachine MUST still obey the same no-queue and RTC rules.
2. If using SML submachines, designers MUST account for SML’s event propagation behavior:
   - submachine gets first chance to handle events,
   - if unhandled, parent transitions may run (fallback). (Source: `boost/sml.hpp`, `transitions_sub<sm<Tsm>, ...>::execute_impl`.)
3. Entry into a composite state MUST initialize its submachine(s) deterministically. SML updates composite state initialization during state updates. (Source: `boost/sml.hpp`, `update_current_state` overload for `state<back::sm<T>>`.)

### Orthogonal regions
4. Orthogonal regions (multiple initial states) MUST be designed so that the same event does not cause side effects in more than one region, unless those side effects commute and ordering does not matter.
5. If a cross-region coordination is required, it MUST be expressed via explicit context variables and deterministic guards, not by relying on region evaluation order.

## 12. Observability and tracing
1. Production tracing MUST be constant-time and bounded-memory. It MUST NOT allocate during dispatch.
2. The preferred integration point is `sml::logger<YourLogger>` which can observe:
   - `log_process_event`, `log_state_change`, `log_action`, `log_guard`. (Docs: User Guide “policies” and logger requirements.)
3. Logger callbacks MUST be deterministic and MUST NOT call back into `process_event`.
4. State inspection for debugging SHOULD use `visit_current_states` and `is(...)` (both are synchronous and can be used without instrumentation).

## 13. Testing and validation
1. Every actor MUST have tests that validate:
   - determinism (same initial state + same event sequence => same trace),
   - bounded internal work (max anonymous transitions per event),
   - no allocations during dispatch (instrument global new/delete in tests),
   - time budget compliance (measure cycles or ns per dispatch).
2. Unit tests SHOULD use `sml::testing::sm` and `set_current_states(...)` to force configurations and test edge cases. (Docs: User Guide “testing::sm”.)
3. Property-based tests SHOULD generate event sequences and assert invariants (never re-enter, never allocate, never exceed transition cap).
4. Integration tests MUST include a single-threaded orchestrator that reproduces the exact dispatch order used in production.

## 14. Reference examples
### Example A: Actor wrapper with synchronous, no-queue dispatch
```cpp
#include <boost/sml.hpp>
namespace sml = boost::sml;

struct Tick { uint64_t now_ns; };
struct Start { uint64_t now_ns; };

struct Ctx {
  uint64_t deadline_ns = 0;
  // References to other actors go here, injected by reference.
};

struct ActorSm {
  auto operator()() const {
    using namespace sml;
    const auto set_deadline = [](Ctx& c, const Start& s) { c.deadline_ns = s.now_ns + 10'000'000; };
    const auto timed_out = [](const Ctx& c, const Tick& t) { return t.now_ns >= c.deadline_ns; };

    return make_transition_table(
      *"idle"_s + event<Start> / set_deadline = "waiting"_s,
       "waiting"_s + event<Tick> [ timed_out ] = "idle"_s
    );
  }
};

struct Actor {
  Ctx ctx;
  sml::sm<ActorSm> sm{ctx};

  template<class Ev>
  bool dispatch(const Ev& ev) { return sm.process_event(ev); }
};
```

### Example B: Bounded internal progress using anonymous transitions (no self-dispatch)
```cpp
#include <boost/sml.hpp>
namespace sml = boost::sml;

struct Step {};
struct Ctx { int i = 0; };

struct Sm {
  auto operator()() const {
    using namespace sml;
    const auto inc = [](Ctx& c) { ++c.i; };
    const auto more = [](const Ctx& c) { return c.i < 3; };

    return make_transition_table(
      *"s0"_s + event<Step> / inc = "s1"_s,
       "s1"_s [ more ] / inc = "s1"_s  // guarded anonymous transition
    );
  }
};
```
Rule check: the anonymous self-loop is bounded by `i < 3`; cycles without bounds are forbidden.
Rule check: the anonymous self-loop is bounded by `i < 3`; cycles without bounds are forbidden.

### Example C: Runtime event dispatch without dynamic allocation
```cpp
#include <boost/sml.hpp>
#include <boost/sml/utility/dispatch_table.hpp>
namespace sml = boost::sml;

struct RuntimeEv { int id; /* payload */ };
struct Ev1 { static constexpr auto id = 1; Ev1(const RuntimeEv&) {} };
struct Ev2 { static constexpr auto id = 2; Ev2(const RuntimeEv&) {} };

struct Sm {
  auto operator()() const {
    using namespace sml;
    return make_transition_table(
      *"idle"_s + event<Ev1> = "idle"_s,
       "idle"_s + event<Ev2> = "idle"_s
    );
  }
};

int main() {
  sml::sm<Sm> sm;

  // Build once, use repeatedly.
  auto dispatch_runtime = sml::utility::make_dispatch_table<RuntimeEv, 1, 2>(sm);

  RuntimeEv e{.id = 1};
  if (e.id < 1 || e.id > 2) return 1;  // MUST validate before indexing.
  dispatch_runtime(e, e.id);
}
```
This uses a static dispatch table indexed by event ID; the ID range must be validated by the caller.
