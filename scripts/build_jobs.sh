#!/usr/bin/env bash
# Shared hard memory envelope and secondary build-parallelism clamp. Every
# repository script that builds sources this file before invoking a build tool.

unset EMEL_MEMORY_TEST_MODE
if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == "--memory-cap-check" ]]; then
  EMEL_MEMORY_TEST_MODE=1
fi

emel_memory_error() { printf 'error: %s\n' "$*" >&2; }
emel_is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

emel_physical_memory_bytes() {
  if [[ "${EMEL_MEMORY_TEST_MODE:-0}" == 1 && -n "${EMEL_MEMORY_TEST_PHYSICAL_BYTES:-}" ]]; then
    emel_is_uint "$EMEL_MEMORY_TEST_PHYSICAL_BYTES" && ((EMEL_MEMORY_TEST_PHYSICAL_BYTES > 0)) || {
      emel_memory_error "EMEL_MEMORY_TEST_PHYSICAL_BYTES must be a positive integer"; return 1;
    }
    printf '%s\n' "$EMEL_MEMORY_TEST_PHYSICAL_BYTES"; return
  fi
  if [[ -r /proc/meminfo ]]; then
    local kb; kb="$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)"
    emel_is_uint "$kb" && ((kb > 0)) || { emel_memory_error "could not determine physical memory"; return 1; }
    printf '%s\n' "$((kb * 1024))"; return
  fi
  local bytes; bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
  emel_is_uint "$bytes" && ((bytes > 0)) || { emel_memory_error "could not determine physical memory"; return 1; }
  printf '%s\n' "$bytes"
}

emel_cgroup_v2_dir() {
  [[ -r /proc/self/cgroup && -r /sys/fs/cgroup/cgroup.controllers ]] || return 1
  local hierarchy controllers membership
  while IFS=: read -r hierarchy controllers membership; do
    if [[ "$hierarchy" == 0 && -z "$controllers" && "$membership" == /* &&
          "$membership" != *'/../'* && "$membership" != */.. ]]; then
      printf '/sys/fs/cgroup%s\n' "$membership"; return
    fi
  done </proc/self/cgroup
  return 1
}

emel_read_cgroup_limits() {
  local dir="$1" max swap
  [[ -r "$dir/memory.max" && -r "$dir/memory.swap.max" ]] || return 1
  IFS= read -r max <"$dir/memory.max"; IFS= read -r swap <"$dir/memory.swap.max"
  printf '%s %s\n' "$max" "$swap"
}

emel_current_cgroup_limits() {
  if [[ "${EMEL_MEMORY_TEST_MODE:-0}" == 1 &&
        ( -n "${EMEL_MEMORY_TEST_CURRENT_MAX:-}" || -n "${EMEL_MEMORY_TEST_CURRENT_SWAP:-}" ) ]]; then
    printf '%s %s\n' "${EMEL_MEMORY_TEST_CURRENT_MAX:-max}" "${EMEL_MEMORY_TEST_CURRENT_SWAP:-max}"; return
  fi
  local dir; dir="$(emel_cgroup_v2_dir)" || return 1
  emel_read_cgroup_limits "$dir"
}

emel_min_cgroup_memory_max() {
  local start="$1" dir="$1" max swap minimum=max
  if [[ "${EMEL_MEMORY_TEST_MODE:-0}" == 1 && -n "${EMEL_MEMORY_TEST_ANCESTOR_MAXES:-}" ]]; then
    local value
    IFS=, read -ra values <<<"$EMEL_MEMORY_TEST_ANCESTOR_MAXES"
    for value in "${values[@]}"; do
      if [[ "$value" != max ]]; then
        emel_is_uint "$value" && ((value > 0)) || { emel_memory_error "invalid test ancestor memory.max: $value"; return 1; }
        if [[ "$minimum" == max || value -lt minimum ]]; then minimum="$value"; fi
      fi
    done
    printf '%s\n' "$minimum"; return
  fi
  while :; do
    read -r max swap < <(emel_read_cgroup_limits "$dir") || return 1
    if [[ "$max" != max ]]; then
      emel_is_uint "$max" && ((max > 0)) || { emel_memory_error "invalid cgroup memory.max at $dir: $max"; return 1; }
      if [[ "$minimum" == max || max -lt minimum ]]; then minimum="$max"; fi
    fi
    [[ "$dir" == /sys/fs/cgroup ]] && break
    dir="${dir%/*}"; [[ -n "$dir" && "$dir" != "$start/.." ]] || return 1
  done
  printf '%s\n' "$minimum"
}

emel_effective_total_from_limit() {
  local physical="$1" limit="$2"
  if [[ "$limit" == max ]]; then printf '%s\n' "$physical"; return; fi
  if ((limit < physical)); then printf '%s\n' "$limit"; else printf '%s\n' "$physical"; fi
}

emel_effective_total_memory_bytes() {
  local physical dir minimum; physical="$(emel_physical_memory_bytes)" || return
  dir="$(emel_cgroup_v2_dir)" || { printf '%s\n' "$physical"; return; }
  minimum="$(emel_min_cgroup_memory_max "$dir")" || return
  emel_effective_total_from_limit "$physical" "$minimum"
}

emel_cap_bytes_for_total() {
  local total="$1" percent="${EMEL_MEMORY_CAP_PERCENT:-50}"
  emel_is_uint "$percent" && ((percent >= 1 && percent <= 50)) || { emel_memory_error "EMEL_MEMORY_CAP_PERCENT must be 1..50"; return 1; }
  printf '%s\n' "$((total / 100 * percent + (total % 100) * percent / 100))"
}

emel_inside_owned_envelope() {
  if [[ "${EMEL_MEMORY_TEST_MODE:-0}" == 1 && "${EMEL_MEMORY_TEST_OWNED_SCOPE:-0}" == 1 ]]; then return 0; fi
  local dir base; dir="$(emel_cgroup_v2_dir)" || return 1; base="${dir##*/}"
  [[ "$base" == emel-build-*.scope || "$base" == emel-build-* ]]
}

emel_verify_active_linux_envelope() {
  local physical dir parent ancestor_min expected current_max current_swap
  physical="$(emel_physical_memory_bytes)" || return
  dir="$(emel_cgroup_v2_dir)" || { emel_memory_error "cannot locate active cgroup v2"; return 1; }
  parent="${dir%/*}"; [[ "$parent" != "$dir" ]] || { emel_memory_error "owned envelope has no parent cgroup"; return 1; }
  ancestor_min="$(emel_min_cgroup_memory_max "$parent")" || return
  EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES="$(emel_effective_total_from_limit "$physical" "$ancestor_min")" || return
  expected="$(emel_cap_bytes_for_total "$EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES")" || return
  read -r current_max current_swap < <(emel_current_cgroup_limits) || { emel_memory_error "cannot read active cgroup v2 limits"; return 1; }
  emel_is_uint "$current_max" && ((current_max <= expected)) || { emel_memory_error "memory.max=$current_max exceeds recomputed cap $expected"; return 1; }
  [[ "$current_swap" == 0 ]] || { emel_memory_error "memory.swap.max=$current_swap, required 0"; return 1; }
  EMEL_MEMORY_CAP_BYTES="$current_max"
}

emel_systemd_scope_command() {
  printf 'systemd-run --user --scope --quiet --same-dir --unit=emel-build-%s --property=MemoryMax=%s --property=MemorySwapMax=0 %q' "$$" "$EMEL_MEMORY_CAP_BYTES" "$0"
  local arg; for arg in "$@"; do printf ' %q' "$arg"; done; printf '\n'
}

emel_systemd_available() {
  command -v systemd-run >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1 &&
    systemctl --user show-environment >/dev/null 2>&1 &&
    systemd-run --user --scope --quiet --unit="emel-probe-$$" --property="MemoryMax=$EMEL_MEMORY_CAP_BYTES" --property=MemorySwapMax=0 true >/dev/null 2>&1
}

emel_run_delegated_cgroup() {
  local parent scope ready child status; parent="$(emel_cgroup_v2_dir)" || return 125; [[ -w "$parent/cgroup.procs" ]] || return 125
  scope="$parent/emel-build-$$"; mkdir "$scope" 2>/dev/null || return 125; ready="${TMPDIR:-/tmp}/emel-build-$$.ready"; rm -f "$ready"
  if ! printf '%s\n' "$EMEL_MEMORY_CAP_BYTES" >"$scope/memory.max" || ! printf '0\n' >"$scope/memory.swap.max"; then rmdir "$scope" 2>/dev/null || true; return 125; fi
  env -u EMEL_MEMORY_TEST_PHYSICAL_BYTES -u EMEL_MEMORY_TEST_CURRENT_MAX -u EMEL_MEMORY_TEST_CURRENT_SWAP -u EMEL_MEMORY_TEST_PARENT_MAX -u EMEL_MEMORY_TEST_OS -u EMEL_MEMORY_TEST_OWNED_SCOPE EMEL_MEMORY_CGROUP_READY_FILE="$ready" "$0" "$@" & child=$!
  if ! printf '%s\n' "$child" >"$scope/cgroup.procs"; then kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; rmdir "$scope" 2>/dev/null || true; return 125; fi
  : >"$ready"; status=0; wait "$child" || status=$?; rm -f "$ready"; rmdir "$scope" 2>/dev/null || true; return "$status"
}

emel_enter_memory_envelope() {
  local os status
  if [[ "${EMEL_MEMORY_TEST_MODE:-0}" == 1 && -n "${EMEL_MEMORY_TEST_OS:-}" ]]; then os="$EMEL_MEMORY_TEST_OS"; else os="$(uname -s)"; fi
  if [[ "$os" != Linux ]]; then emel_memory_error "macOS has no supported native aggregate descendant memory controller; sampled watchdogs and Linux-artifact container builds are not valid hard envelopes"; return 1; fi
  if [[ -n "${EMEL_MEMORY_CGROUP_READY_FILE:-}" ]]; then while [[ ! -e "$EMEL_MEMORY_CGROUP_READY_FILE" ]]; do sleep 0.01; done; fi
  if emel_inside_owned_envelope; then emel_verify_active_linux_envelope; return; fi
  if [[ "${EMEL_MEMORY_ENVELOPE_DRY_RUN:-0}" == 1 ]]; then emel_systemd_scope_command "$@"; return 2; fi
  if emel_systemd_available; then
    exec env -u EMEL_MEMORY_TEST_PHYSICAL_BYTES -u EMEL_MEMORY_TEST_CURRENT_MAX -u EMEL_MEMORY_TEST_CURRENT_SWAP -u EMEL_MEMORY_TEST_PARENT_MAX -u EMEL_MEMORY_TEST_OS -u EMEL_MEMORY_TEST_OWNED_SCOPE systemd-run --user --scope --quiet --same-dir --unit="emel-build-$$" --property="MemoryMax=$EMEL_MEMORY_CAP_BYTES" --property=MemorySwapMax=0 "$0" "$@"
  fi
  status=0; emel_run_delegated_cgroup "$@" || status=$?; if ((status != 125)); then exit "$status"; fi
  emel_memory_error "could not install the $EMEL_MEMORY_CAP_BYTES-byte Linux cgroup v2 process-tree envelope; enable user systemd or delegated cgroup v2"; return 1
}

emel_compute_build_jobs() {
  local budget="${1:-${EMEL_BUILD_JOB_MEM_GB:-6}}" cores usable mem_jobs jobs
  emel_is_uint "$budget" && ((budget > 0)) || { emel_memory_error "EMEL_BUILD_JOB_MEM_GB must be positive"; return 1; }
  if [[ "${EMEL_MEMORY_TEST_MODE:-0}" == 1 && -n "${EMEL_MEMORY_TEST_CORES:-}" ]]; then cores="$EMEL_MEMORY_TEST_CORES"; else cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4\n')"; fi
  emel_is_uint "$cores" && ((cores > 0)) || return 1
  usable=$((EMEL_MEMORY_CAP_BYTES - EMEL_MEMORY_RESERVE_BYTES)); if ((usable < 1)); then usable=1; fi; mem_jobs=$((usable / (budget * 1073741824))); jobs=$((cores < mem_jobs ? cores : mem_jobs)); if ((jobs < 1)); then jobs=1; fi; printf '%s\n' "$jobs"
}

emel_initialize_build_memory() {
  local reserve requested
  if emel_inside_owned_envelope; then emel_verify_active_linux_envelope || return; else EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES="$(emel_effective_total_memory_bytes)" || return; EMEL_MEMORY_CAP_BYTES="$(emel_cap_bytes_for_total "$EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES")" || return; fi
  reserve=$((EMEL_MEMORY_CAP_BYTES / 10)); if ((reserve < 1073741824)); then reserve=1073741824; elif ((reserve > 4294967296)); then reserve=4294967296; fi; if ((reserve > EMEL_MEMORY_CAP_BYTES / 2)); then reserve=$((EMEL_MEMORY_CAP_BYTES / 2)); fi
  EMEL_MEMORY_RESERVE_BYTES="${EMEL_BUILD_RESERVE_BYTES:-$reserve}"; emel_is_uint "$EMEL_MEMORY_RESERVE_BYTES" && ((EMEL_MEMORY_RESERVE_BYTES < EMEL_MEMORY_CAP_BYTES)) || { emel_memory_error "invalid EMEL_BUILD_RESERVE_BYTES"; return 1; }
  EMEL_SAFE_BUILD_JOBS="$(emel_compute_build_jobs)" || return; requested="${EMEL_BUILD_JOBS:-$EMEL_SAFE_BUILD_JOBS}"; emel_is_uint "$requested" && ((requested > 0)) || { emel_memory_error "EMEL_BUILD_JOBS must be a positive integer"; return 1; }; if ((requested > EMEL_SAFE_BUILD_JOBS)); then printf 'warning: clamping EMEL_BUILD_JOBS=%s to %s\n' "$requested" "$EMEL_SAFE_BUILD_JOBS" >&2; requested="$EMEL_SAFE_BUILD_JOBS"; fi
  EMEL_BUILD_JOBS="$requested"; CMAKE_BUILD_PARALLEL_LEVEL="$requested"; export EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES EMEL_MEMORY_CAP_BYTES EMEL_MEMORY_RESERVE_BYTES EMEL_SAFE_BUILD_JOBS EMEL_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL
}

emel_initialize_build_memory || return 1 2>/dev/null || exit 1
if [[ "${1:-}" == --memory-cap-check ]]; then printf 'effective_total_bytes=%s\ncap_bytes=%s\nreserve_bytes=%s\nsafe_build_jobs=%s\nbuild_jobs=%s\n' "$EMEL_MEMORY_EFFECTIVE_TOTAL_BYTES" "$EMEL_MEMORY_CAP_BYTES" "$EMEL_MEMORY_RESERVE_BYTES" "$EMEL_SAFE_BUILD_JOBS" "$EMEL_BUILD_JOBS"; if ! emel_inside_owned_envelope; then check_status=0; EMEL_MEMORY_ENVELOPE_DRY_RUN=1 emel_enter_memory_envelope "$@" || check_status=$?; if ((check_status != 2)); then exit "$check_status"; fi; fi; exit 0; fi
if [[ "${1:-}" == --memory-cap-run ]]; then shift; (($# > 0)) || exit 1; emel_enter_memory_envelope --memory-cap-run "$@"; exec "$@"; fi
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then emel_enter_memory_envelope "$@"; fi
