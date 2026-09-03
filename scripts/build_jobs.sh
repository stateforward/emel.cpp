#!/usr/bin/env bash
# Shared memory-budget and build-parallelism calculations. Aggregate quality
# gates install the exported cap as a process-tree envelope; standalone build
# scripts use the same capped budget to derive their secondary job clamp.

emel_memory_error() {
  printf 'error: %s\n' "$*" >&2
}

emel_is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

emel_physical_memory_bytes() {
  if [[ -n "${EMEL_MEMORY_TEST_PHYSICAL_BYTES:-}" ]]; then
    emel_is_uint "$EMEL_MEMORY_TEST_PHYSICAL_BYTES" &&
      ((EMEL_MEMORY_TEST_PHYSICAL_BYTES > 0)) || {
        emel_memory_error "EMEL_MEMORY_TEST_PHYSICAL_BYTES must be a positive integer"
        return 1
      }
    printf '%s\n' "$EMEL_MEMORY_TEST_PHYSICAL_BYTES"
    return
  fi

  if [[ -r /proc/meminfo ]]; then
    local mem_kb
    mem_kb="$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)"
    emel_is_uint "$mem_kb" && ((mem_kb > 0)) || {
      emel_memory_error "could not determine physical memory from /proc/meminfo"
      return 1
    }
    printf '%s\n' "$((mem_kb * 1024))"
    return
  fi

  local mem_bytes
  mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
  emel_is_uint "$mem_bytes" && ((mem_bytes > 0)) || {
    emel_memory_error "could not determine physical memory (expected /proc/meminfo or sysctl hw.memsize)"
    return 1
  }
  printf '%s\n' "$mem_bytes"
}

emel_cgroup_v2_dir() {
  [[ -r /proc/self/cgroup && -r /sys/fs/cgroup/cgroup.controllers ]] || return 1
  local hierarchy controllers membership
  while IFS=: read -r hierarchy controllers membership; do
    if [[ "$hierarchy" == "0" && -z "$controllers" && "$membership" == /* &&
          "$membership" != *'/../'* && "$membership" != */.. ]]; then
      printf '/sys/fs/cgroup%s\n' "$membership"
      return 0
    fi
  done </proc/self/cgroup
  return 1
}

emel_current_cgroup_memory_max() {
  if [[ -n "${EMEL_MEMORY_TEST_CGROUP_MAX:-}" ]]; then
    printf '%s\n' "$EMEL_MEMORY_TEST_CGROUP_MAX"
    return
  fi
  local cgroup_dir
  cgroup_dir="$(emel_cgroup_v2_dir)" || {
    printf 'max\n'
    return
  }
  [[ -r "$cgroup_dir/memory.max" ]] || {
    printf 'max\n'
    return
  }
  IFS= read -r REPLY <"$cgroup_dir/memory.max"
  printf '%s\n' "$REPLY"
}

emel_effective_total_memory_bytes() {
  local physical_bytes cgroup_max effective
  physical_bytes="$(emel_physical_memory_bytes)" || return

  # An active envelope exports the pre-envelope basis so sourcing this file in
  # nested build scripts does not repeatedly halve the already-capped scope.
  if [[ "${EMEL_MEMORY_ENVELOPE_ACTIVE:-0}" == "1" &&
        -n "${EMEL_MEMORY_BASE_TOTAL_BYTES:-}" ]]; then
    emel_is_uint "$EMEL_MEMORY_BASE_TOTAL_BYTES" &&
      ((EMEL_MEMORY_BASE_TOTAL_BYTES > 0 && EMEL_MEMORY_BASE_TOTAL_BYTES <= physical_bytes)) || {
        emel_memory_error "invalid EMEL_MEMORY_BASE_TOTAL_BYTES in active envelope"
        return 1
      }
    printf '%s\n' "$EMEL_MEMORY_BASE_TOTAL_BYTES"
    return
  fi

  effective="$physical_bytes"
  cgroup_max="$(emel_current_cgroup_memory_max)" || return
  if [[ "$cgroup_max" != "max" ]]; then
    emel_is_uint "$cgroup_max" && ((cgroup_max > 0)) || {
      emel_memory_error "cgroup memory.max is neither 'max' nor a positive integer: $cgroup_max"
      return 1
    }
    if ((cgroup_max < effective)); then
      effective="$cgroup_max"
    fi
  fi
  printf '%s\n' "$effective"
}

emel_memory_cap_percent() {
  local percent="${EMEL_MEMORY_CAP_PERCENT:-50}"
  emel_is_uint "$percent" && ((percent >= 1 && percent <= 100)) || {
    emel_memory_error "EMEL_MEMORY_CAP_PERCENT must be an integer from 1 through 100"
    return 1
  }
  if ((percent > 50)) && [[ "${EMEL_DANGEROUS_ALLOW_MEMORY_CAP_ABOVE_50:-0}" != "1" ]]; then
    emel_memory_error "EMEL_MEMORY_CAP_PERCENT above 50 requires EMEL_DANGEROUS_ALLOW_MEMORY_CAP_ABOVE_50=1"
    return 1
  fi
  printf '%s\n' "$percent"
}

emel_compute_build_jobs() {
  local budget_gb="${1:-${EMEL_BUILD_JOB_MEM_GB:-6}}"
  local cores="${EMEL_MEMORY_TEST_CORES:-}"
  local usable_bytes mem_jobs safe_jobs
  emel_is_uint "$budget_gb" && ((budget_gb > 0)) || {
    emel_memory_error "EMEL_BUILD_JOB_MEM_GB must be a positive integer"
    return 1
  }
  if [[ -z "$cores" ]]; then
    cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4\n')"
  fi
  emel_is_uint "$cores" && ((cores > 0)) || {
    emel_memory_error "online processor count must be a positive integer"
    return 1
  }

  usable_bytes=$((EMEL_MEMORY_CAP_BYTES - EMEL_MEMORY_RESERVE_BYTES))
  if ((usable_bytes < 1)); then
    usable_bytes=1
  fi
  mem_jobs=$((usable_bytes / (budget_gb * 1024 * 1024 * 1024)))
  safe_jobs=$((cores < mem_jobs ? cores : mem_jobs))
  if ((safe_jobs < 1)); then
    safe_jobs=1
  fi
  printf '%s\n' "$safe_jobs"
}

emel_initialize_build_memory() {
  local effective_total cap_percent reserve_default requested_jobs
  effective_total="$(emel_effective_total_memory_bytes)" || return
  cap_percent="$(emel_memory_cap_percent)" || return

  EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES="$effective_total"
  EMEL_MEMORY_BASE_TOTAL_BYTES="$effective_total"
  EMEL_MEMORY_CAP_PERCENT_EFFECTIVE="$cap_percent"
  EMEL_MEMORY_CAP_BYTES=$((effective_total / 100 * cap_percent + (effective_total % 100) * cap_percent / 100))
  ((EMEL_MEMORY_CAP_BYTES > 0)) || {
    emel_memory_error "computed memory cap is zero"
    return 1
  }

  reserve_default=$((EMEL_MEMORY_CAP_BYTES / 10))
  if ((reserve_default < 1024 * 1024 * 1024)); then
    reserve_default=$((1024 * 1024 * 1024))
  elif ((reserve_default > 4 * 1024 * 1024 * 1024)); then
    reserve_default=$((4 * 1024 * 1024 * 1024))
  fi
  if ((reserve_default > EMEL_MEMORY_CAP_BYTES / 2)); then
    reserve_default=$((EMEL_MEMORY_CAP_BYTES / 2))
  fi
  EMEL_MEMORY_RESERVE_BYTES="${EMEL_BUILD_RESERVE_BYTES:-$reserve_default}"
  emel_is_uint "$EMEL_MEMORY_RESERVE_BYTES" &&
    ((EMEL_MEMORY_RESERVE_BYTES < EMEL_MEMORY_CAP_BYTES)) || {
      emel_memory_error "EMEL_BUILD_RESERVE_BYTES must be a non-negative integer below the memory cap"
      return 1
    }

  EMEL_SAFE_BUILD_JOBS="$(emel_compute_build_jobs)" || return
  requested_jobs="${EMEL_BUILD_JOBS:-$EMEL_SAFE_BUILD_JOBS}"
  emel_is_uint "$requested_jobs" && ((requested_jobs > 0)) || {
    emel_memory_error "EMEL_BUILD_JOBS must be a positive integer"
    return 1
  }
  if ((requested_jobs > EMEL_SAFE_BUILD_JOBS)) &&
    [[ "${EMEL_DANGEROUS_ALLOW_UNSAFE_BUILD_JOBS:-0}" != "1" ]]; then
    printf 'warning: clamping EMEL_BUILD_JOBS=%s to memory-safe maximum %s; use EMEL_DANGEROUS_ALLOW_UNSAFE_BUILD_JOBS=1 to bypass\n' \
      "$requested_jobs" "$EMEL_SAFE_BUILD_JOBS" >&2
    requested_jobs="$EMEL_SAFE_BUILD_JOBS"
  fi
  EMEL_BUILD_JOBS="$requested_jobs"
  CMAKE_BUILD_PARALLEL_LEVEL="$EMEL_BUILD_JOBS"

  export EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES EMEL_MEMORY_BASE_TOTAL_BYTES
  export EMEL_MEMORY_CAP_PERCENT_EFFECTIVE EMEL_MEMORY_CAP_BYTES
  export EMEL_MEMORY_RESERVE_BYTES EMEL_SAFE_BUILD_JOBS EMEL_BUILD_JOBS
  export CMAKE_BUILD_PARALLEL_LEVEL
}

emel_initialize_build_memory || return 1 2>/dev/null || exit 1

if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == "--memory-cap-check" ]]; then
  printf 'effective_total_bytes=%s\n' "$EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES"
  printf 'cap_percent=%s\n' "$EMEL_MEMORY_CAP_PERCENT_EFFECTIVE"
  printf 'cap_bytes=%s\n' "$EMEL_MEMORY_CAP_BYTES"
  printf 'reserve_bytes=%s\n' "$EMEL_MEMORY_RESERVE_BYTES"
  printf 'safe_build_jobs=%s\n' "$EMEL_SAFE_BUILD_JOBS"
  printf 'build_jobs=%s\n' "$EMEL_BUILD_JOBS"
fi
