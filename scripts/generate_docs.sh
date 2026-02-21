#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mode="${1:-}"
if [[ -n "$mode" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 1
fi

CHECK_MODE=false
if [[ "$mode" == "--check" ]]; then
  CHECK_MODE=true
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

README_FILE="$ROOT_DIR/README.md"
BENCH_FILE="$ROOT_DIR/docs/benchmarks.md"
BENCH_COMPARE_SNAPSHOT="$ROOT_DIR/snapshots/bench/benchmarks_compare.txt"

require_tools() {
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
}

run_benchmarks_compare() {
  if [[ ! -x "$ROOT_DIR/scripts/bench.sh" ]]; then
    echo "error: missing executable scripts/bench.sh" >&2
    exit 1
  fi

  if $CHECK_MODE; then
    local current="$TMP_DIR/benchmarks_compare.txt"
    "$ROOT_DIR/scripts/bench.sh" --compare > "$current"
    if [[ ! -f "$BENCH_COMPARE_SNAPSHOT" ]]; then
      echo "error: missing $BENCH_COMPARE_SNAPSHOT" >&2
      exit 1
    fi
    if ! diff -u "$BENCH_COMPARE_SNAPSHOT" "$current"; then
      echo "error: benchmark comparison snapshot out of sync" >&2
      exit 1
    fi
    return
  fi

  "$ROOT_DIR/scripts/bench.sh" --compare-update
}

collect_sm_headers() {
  find "$ROOT_DIR/src/emel" -type f -name 'sm.hpp' \
    ! -path "$ROOT_DIR/src/emel/sm.hpp" | sort
}

docs_toc() {
  echo "- [\`docs/benchmarks.md\`](docs/benchmarks.md)"
  while IFS= read -r h; do
    local rel_emel="${h#"$ROOT_DIR/src/emel/"}"
    local dir="${rel_emel%/sm.hpp}"
    local name="${dir//\//_}"
    echo "- [\`docs/architecture/$name.md\`](docs/architecture/$name.md)"
  done < <(collect_sm_headers)
}

generate_readme() {
  local toc_file="$TMP_DIR/docs_toc.txt"
  docs_toc > "$toc_file"
  sed -e "/__DOCS_TOC__/r $toc_file" -e "/__DOCS_TOC__/d" <<'MD'
# EMEL

Deterministic, production-grade C++ inference engine built around Boost.SML orchestration.

## Status: WIP

This repository is under active development. APIs, state machines, and formats will change.
If you’re evaluating EMEL, expect fast iteration and breaking changes until the core loader,
allocator, and execution pipelines stabilize.

This inference engine is being implemented by AI under human engineering and architecture direction.

## Why EMEL

EMEL exists to make inference behavior explicit and verifiable. Instead of ad-hoc control flow,
orchestration is modeled as Boost.SML state machines with deterministic, testable transitions.
That enables:

1. Clear operational semantics and failure modes.
2. Deterministic, reproducible inference paths.
3. High-performance, C-compatible boundaries without dynamic dispatch in hot paths.
4. Auditable parity work against reference implementations without copying their control flow.

## The name

“EMEL” is pronounced like “ML”. It’s a short, neutral name that doesn’t carry existing
assumptions or baggage. It’s intentionally low-ceremony while we iterate on the core design.

## Build and test

```bash
scripts/quality_gates.sh
```

Individual gates live in `scripts/build_with_zig.sh`, `scripts/test_with_coverage.sh`,
`scripts/lint_snapshot.sh`, and `scripts/bench.sh`.

### Why Zig for builds

Zig’s C/C++ toolchain gives us consistent, fast, cross-platform builds without forcing a full
dependency on any single system compiler or SDK. It keeps the default dev path reproducible,
while still allowing native toolchains when needed.

### Why CMake for tests and coverage

Coverage and CI tooling are already standardized around CMake + CTest + llvm-cov/gcovr in this
repo. Using CMake for test/coverage builds keeps gates deterministic and portable across CI
environments, while Zig remains the default for day-to-day builds.

## Documentation

- [Architecture](docs/architecture/) (generated state-machine docs + Mermaid diagrams)
- [Benchmarks](docs/benchmarks.md) (generated benchmark snapshot table)
- [SML Conventions](docs/third_party/sml.md) (Boost.SML conventions and usage)
- [Parity Audit](docs/gaps.md) (parity audit status)

## Docs index

__DOCS_TOC__

## Regenerating docs

```bash
scripts/generate_docs.sh
```

Use `scripts/generate_docs.sh --check` in CI to validate generated artifacts.
MD
}

generate_sm_docs() {
  local out_md_dir="$ROOT_DIR/docs/architecture"
  local out_mermaid_dir="$ROOT_DIR/docs/architecture/mermaid"
  local build_dir="$ROOT_DIR/build/debug"
  local sm_headers=()

  while IFS= read -r h; do
    sm_headers+=("$h")
  done < <(collect_sm_headers)

  if [[ ${#sm_headers[@]} -eq 0 ]]; then
    echo "error: no sm.hpp files found under $ROOT_DIR/src/emel" >&2
    exit 1
  fi

  require_tools cmake zig

  cmake -S "$ROOT_DIR" --preset debug >/dev/null

  local boost_sml_include="$build_dir/_deps/boost_sml-src/include"
  if [[ ! -d "$boost_sml_include" ]]; then
    echo "error: boost_sml include dir not found: $boost_sml_include" >&2
    exit 1
  fi

  local cpp_file="$TMP_DIR/generate_dynamic_md.cpp"
  local bin_file="$TMP_DIR/generate_dynamic_md"
  local gen_md_dir="$TMP_DIR/generated_md"
  local gen_mermaid_dir="$TMP_DIR/generated_mermaid"
  mkdir -p "$gen_md_dir" "$gen_mermaid_dir"

  {
    echo '#include <boost/sml.hpp>'
    echo '#include <fstream>'
    echo '#include <iostream>'
    echo '#include <sstream>'
    echo '#include <string>'
    echo '#include <cctype>'
    echo

    for h in "${sm_headers[@]}"; do
      local rel="${h#"$ROOT_DIR/src/"}"
      echo "#include \"$rel\""
    done

    sed "s|REPO_URL_PLACEHOLDER|https://github.com/stateforward/emel.cpp|g" <<'CPP'

constexpr const char * k_repo_url = "REPO_URL_PLACEHOLDER";
constexpr const char * k_repo_src_prefix = "src/";

template <class T>
std::string sml_name() noexcept {
  namespace sml = boost::sml;
  return std::string{sml::aux::string<T>{}.c_str()};
}

inline bool is_none_like(const std::string & name) noexcept {
  return name.empty() || name == "none" || name == "zero_wrapper";
}

inline std::string short_name(const std::string & name) noexcept {
  const auto pos = name.rfind("::");
  if (pos == std::string::npos) {
    return name;
  }
  return name.substr(pos + 2);
}

inline std::string sanitize_mermaid(const std::string & name) noexcept {
  if (name == "[*]") {
    return name;
  }
  std::string out;
  out.reserve(name.size());
  for (unsigned char c : name) {
    if (std::isalnum(c) || c == '_') {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "state";
  }
  return out;
}

inline std::string md_link(const std::string & name, const std::string & source_header) noexcept {
  if (name.empty() || name == "-") {
    return "-";
  }
  if (name == "[*]") {
    return "`[*]`";
  }
  const auto shorty = short_name(name);
  std::string link;
  link.reserve(64 + source_header.size() + shorty.size());
  link.append("[`");
  link.append(shorty);
  link.append("`](");
  link.append(k_repo_url);
  link.append("/blob/main/");
  link.append(k_repo_src_prefix);
  link.append(source_header);
  link.append(")");
  return link;
}

template <class T>
void dump_transition(std::ostream & md, std::ostream & mermaid,
                     const std::string & source_header) noexcept {
  namespace sml = boost::sml;

  auto src_state = sml_name<typename T::src_state>();
  auto dst_state = sml_name<typename T::dst_state>();

  if (dst_state == "X" || dst_state == "terminate") {
    dst_state = "[*]";
  }

  auto event = sml_name<typename T::event>();
  auto guard = sml_name<typename T::guard>();
  auto action = sml_name<typename T::action>();

  if (event == "anonymous") {
    event.clear();
  }
  if (guard == "always") {
    guard.clear();
  }
  if (is_none_like(action)) {
    action.clear();
  }

  const auto src_id = sanitize_mermaid(src_state);
  const auto dst_id = sanitize_mermaid(dst_state);
  const auto event_label = sanitize_mermaid(event);
  const auto guard_label = sanitize_mermaid(guard);
  const auto action_label = sanitize_mermaid(action);

  if (T::initial) {
    mermaid << "  [*] --> " << src_id << "\n";
  }

  mermaid << "  " << src_id << " --> " << dst_id;
  if (!event.empty() || !guard.empty() || !action.empty()) {
    mermaid << " :";
    if (!event.empty()) {
      mermaid << " " << event_label;
    }
    if (!guard.empty()) {
      mermaid << " [" << guard_label << "]";
    }
    if (!action.empty()) {
      mermaid << " / " << action_label;
    }
  }
  mermaid << "\n";

  md << "| " << md_link(src_state, source_header) << " | ";
  if (event.empty()) {
    md << "-";
  } else {
    md << md_link(event, source_header);
  }
  md << " | ";
  if (guard.empty()) {
    md << "-";
  } else {
    md << md_link(guard, source_header);
  }
  md << " | ";
  if (action.empty()) {
    md << "-";
  } else {
    md << md_link(action, source_header);
  }
  md << " | " << md_link(dst_state, source_header) << " |\n";
}

template <class TTransitions>
void dump_transitions(const TTransitions &, std::ostream &, std::ostream &,
                      const std::string &) noexcept {}

template <template <class...> class T, class... Ts>
void dump_transitions(const T<Ts...> &, std::ostream & md, std::ostream & mermaid,
                      const std::string & source_header) noexcept {
  int _[]{0, (dump_transition<Ts>(md, mermaid, source_header), 0)...};
  (void) _;
}

template <class TSM>
void dump_machine_doc(
    const std::string & machine_name, const std::string & source_header, std::ostream & md,
    std::ostream & mermaid) noexcept {
  namespace sml = boost::sml;
  using transitions_t = typename sml::sm<TSM>::transitions;

  std::ostringstream table_rows;
  std::ostringstream mermaid_rows;
  dump_transitions(transitions_t{}, table_rows, mermaid_rows, source_header);

  md << "# " << machine_name << "\n\n";
  md << "Source: [`" << source_header << "`](" << k_repo_url << "/blob/main/"
     << k_repo_src_prefix << source_header << ")\n\n";
  md << "## Mermaid\n\n";
  md << "```mermaid\n";
  md << "stateDiagram-v2\n";
  md << mermaid_rows.str();
  md << "```\n\n";

  mermaid << "stateDiagram-v2\n";
  mermaid << mermaid_rows.str();

  md << "## Transitions\n\n";
  md << "| Source | Event | Guard | Action | Target |\n";
  md << "| --- | --- | --- | --- | --- |\n";
  md << table_rows.str();
}

int main(int argc, char ** argv) {
  if (argc != 3) {
    std::cerr << "usage: generate_dynamic_md <out_md_dir> <out_mermaid_dir>\n";
    return 1;
  }
  const std::string out_md_dir = argv[1];
  const std::string out_mermaid_dir = argv[2];
CPP

    for h in "${sm_headers[@]}"; do
      local rel="${h#"$ROOT_DIR/src/"}"
      local rel_emel="${h#"$ROOT_DIR/src/emel/"}"
      local dir="${rel_emel%/sm.hpp}"
      local name="${dir//\//_}"
      local type="emel::${dir//\//::}::sm::model_type"
      echo "  {"
      echo "    std::ofstream md(out_md_dir + \"/$name.md\");"
      echo "    std::ofstream mermaid(out_mermaid_dir + \"/$name.mmd\");"
      echo "    dump_machine_doc<$type>(\"$name\", \"$rel\", md, mermaid);"
      echo "  }"
    done

    cat <<'CPP'
  return 0;
}
CPP
  } > "$cpp_file"

  zig c++ -std=c++20 \
    -I"$ROOT_DIR/src" \
    -I"$ROOT_DIR/include" \
    -I"$boost_sml_include" \
    "$cpp_file" -o "$bin_file"

  "$bin_file" "$gen_md_dir" "$gen_mermaid_dir"

  if $CHECK_MODE; then
    for h in "${sm_headers[@]}"; do
      local rel_emel="${h#"$ROOT_DIR/src/emel/"}"
      local dir="${rel_emel%/sm.hpp}"
      local name="${dir//\//_}"
      if [[ ! -f "$out_md_dir/$name.md" ]]; then
        echo "error: missing $out_md_dir/$name.md" >&2
        exit 1
      fi
      if [[ ! -f "$out_mermaid_dir/$name.mmd" ]]; then
        echo "error: missing $out_mermaid_dir/$name.mmd" >&2
        exit 1
      fi
      if ! diff -u "$out_md_dir/$name.md" "$gen_md_dir/$name.md"; then
        echo "error: markdown out of sync ($name)" >&2
        exit 1
      fi
      if ! diff -u "$out_mermaid_dir/$name.mmd" "$gen_mermaid_dir/$name.mmd"; then
        echo "error: mermaid out of sync ($name)" >&2
        exit 1
      fi
    done
    return
  fi

  mkdir -p "$out_md_dir" "$out_mermaid_dir"
  for h in "${sm_headers[@]}"; do
    local rel_emel="${h#"$ROOT_DIR/src/emel/"}"
    local dir="${rel_emel%/sm.hpp}"
    local name="${dir//\//_}"
    cp "$gen_md_dir/$name.md" "$out_md_dir/$name.md"
    cp "$gen_mermaid_dir/$name.mmd" "$out_mermaid_dir/$name.mmd"
    echo "updated: $out_md_dir/$name.md"
    echo "updated: $out_mermaid_dir/$name.mmd"
  done
}

generate_benchmarks() {
  local snapshot="$BENCH_COMPARE_SNAPSHOT"
  local tmp_file="$TMP_DIR/benchmarks.md"

  if [[ ! -f "$snapshot" ]]; then
    echo "error: missing $snapshot" >&2
    exit 1
  fi

  {
    echo "# Benchmarks"
    echo
    echo "Source: \`snapshots/bench/benchmarks_compare.txt\`"
    echo
    echo "Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very"
    echo "intertwined. These microbenches aim for apples-to-apples comparisons but likely"
    echo "are not. True benchmarks will be end-to-end once the system is complete."
    echo
    echo "| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |"
    echo "| --- | ---: | ---: | ---: |"
    awk 'NF && $1 !~ /^#/ {
      name = $1;
      emel = $3;
      ref = $6;
      ratio = $8;
      sub(/^ratio=/, "", ratio);
      printf("| `%s` | %s | %s | %s |\n", name, emel, ref, ratio);
    }' "$snapshot"
  } > "$tmp_file"

  if $CHECK_MODE; then
    if [[ ! -f "$BENCH_FILE" ]]; then
      echo "error: missing $BENCH_FILE" >&2
      exit 1
    fi
    if ! diff -u "$BENCH_FILE" "$tmp_file"; then
      echo "error: benchmarks doc out of sync" >&2
      exit 1
    fi
    return
  fi

  mkdir -p "$(dirname "$BENCH_FILE")"
  cp "$tmp_file" "$BENCH_FILE"
  echo "updated: $BENCH_FILE"
}

write_readme() {
  local tmp_readme="$TMP_DIR/README.md"
  generate_readme > "$tmp_readme"

  if $CHECK_MODE; then
    if [[ ! -f "$README_FILE" ]]; then
      echo "error: missing $README_FILE" >&2
      exit 1
    fi
    if ! diff -u "$README_FILE" "$tmp_readme"; then
      echo "error: README.md out of sync" >&2
      exit 1
    fi
    return
  fi

  cp "$tmp_readme" "$README_FILE"
  echo "updated: $README_FILE"
}

generate_sm_docs
run_benchmarks_compare
generate_benchmarks
write_readme
