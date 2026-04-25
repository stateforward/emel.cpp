#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GSD_DIR="$ROOT_DIR/get-shit-done"
VERSION_FILE="$GSD_DIR/VERSION"
CACHE_DIR="$ROOT_DIR/.cache"
CACHE_FILE="$CACHE_DIR/gsd-latest-release.json"
CACHE_TTL_SECONDS="${GSD_UPDATE_HOOK_CACHE_TTL_SECONDS:-43200}"
LATEST_RELEASE_URL="${GSD_UPDATE_HOOK_RELEASE_URL:-https://api.github.com/repos/gsd-build/get-shit-done/releases/latest}"

log() {
  printf '%s\n' "$*" >&2
}

read_local_version() {
  [[ -f "$VERSION_FILE" ]] || return 1
  tr -d '\r\n' < "$VERSION_FILE" 2>/dev/null
}

now_epoch() {
  date +%s
}

file_mtime_epoch() {
  local path="$1"
  stat -f %m "$path" 2>/dev/null && return 0
  stat -c %Y "$path" 2>/dev/null && return 0
  python3 - "$path" <<'PY' 2>/dev/null
import os
import sys

try:
    print(int(os.path.getmtime(sys.argv[1])))
except Exception:
    print(0)
PY
}

cache_fresh() {
  local now mtime age
  [[ -f "$CACHE_FILE" ]] || return 1
  now="$(now_epoch)"
  mtime="$(file_mtime_epoch "$CACHE_FILE")"
  [[ "$mtime" =~ ^[0-9]+$ ]] || return 1
  age=$(( now - mtime ))
  (( age >= 0 && age < CACHE_TTL_SECONDS ))
}

fetch_release_json() {
  mkdir -p "$CACHE_DIR"
  if cache_fresh; then
    cat "$CACHE_FILE"
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    [[ -f "$CACHE_FILE" ]] && cat "$CACHE_FILE"
    return 0
  fi

  local response=""
  if response="$(curl -fsSL --connect-timeout 2 --max-time 4 "$LATEST_RELEASE_URL" 2>/dev/null)"; then
    printf '%s\n' "$response" > "$CACHE_FILE"
    printf '%s\n' "$response"
    return 0
  fi

  [[ -f "$CACHE_FILE" ]] && cat "$CACHE_FILE"
  return 0
}

json_field() {
  local key="$1"
  python3 - "$key" <<'PY' 2>/dev/null
import json, sys
key = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
value = data.get(key, "")
if isinstance(value, str):
    print(value)
PY
}

normalize_semver() {
  local raw="${1:-}"
  raw="${raw#v}"
  printf '%s\n' "$raw"
}

version_lt() {
  local a="${1:-0.0.0}" b="${2:-0.0.0}"
  local IFS=.
  local a1=0 a2=0 a3=0 b1=0 b2=0 b3=0
  read -r a1 a2 a3 _ <<< "$a"
  read -r b1 b2 b3 _ <<< "$b"
  a1="${a1:-0}"; a2="${a2:-0}"; a3="${a3:-0}"
  b1="${b1:-0}"; b2="${b2:-0}"; b3="${b3:-0}"
  (( 10#$a1 < 10#$b1 )) && return 0
  (( 10#$a1 > 10#$b1 )) && return 1
  (( 10#$a2 < 10#$b2 )) && return 0
  (( 10#$a2 > 10#$b2 )) && return 1
  (( 10#$a3 < 10#$b3 ))
}

main() {
  local local_version release_json latest_tag latest_version release_url

  local_version="$(read_local_version || true)"
  [[ -n "$local_version" ]] || exit 0

  release_json="$(fetch_release_json || true)"
  [[ -n "$release_json" ]] || exit 0

  latest_tag="$(printf '%s' "$release_json" | json_field tag_name)"
  latest_version="$(normalize_semver "$latest_tag")"
  release_url="$(printf '%s' "$release_json" | json_field html_url)"
  [[ -n "$latest_version" ]] || exit 0

  if version_lt "$local_version" "$latest_version"; then
    log "GSD update available: $local_version -> $latest_version"
    if [[ -n "$release_url" ]]; then
      log "Run \$gsd-update to upgrade. Release: $release_url"
    else
      log "Run \$gsd-update to upgrade."
    fi
  fi
}

main "$@" || exit 0
