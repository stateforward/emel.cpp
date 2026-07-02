#!/usr/bin/env bash
# Shared build-parallelism clamp, sourced by every script that builds.
#
# The SML expression-template TUs (generator/kernel/sm tests) cost 2-4GB of
# compiler RSS each, so "one job per core" oversubscribes memory long before
# it saturates CPU (30 jobs x 3GB on a 58GB host swaps the machine). Bound
# jobs by available memory at ~4GB per job as well as by core count.
#
# Override with EMEL_BUILD_JOBS. CMAKE_BUILD_PARALLEL_LEVEL is exported so
# bare `cmake --build` calls (without --parallel) inherit the same bound.

emel_compute_build_jobs() {
  local cores mem_kb mem_jobs jobs
  cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  if [[ -r /proc/meminfo ]]; then
    mem_kb="$(awk '/MemAvailable/ { print $2; exit }' /proc/meminfo)"
  else
    # macOS and other Unix without /proc: derive from total physical memory.
    mem_kb="$(($(sysctl -n hw.memsize 2>/dev/null || echo 17179869184) / 1024))"
  fi
  mem_jobs=$((mem_kb / (4 * 1024 * 1024)))
  jobs=$((cores < mem_jobs ? cores : mem_jobs))
  if ((jobs < 2)); then
    jobs=2
  fi
  printf '%s\n' "$jobs"
}

EMEL_BUILD_JOBS="${EMEL_BUILD_JOBS:-$(emel_compute_build_jobs)}"
export EMEL_BUILD_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="$EMEL_BUILD_JOBS"
