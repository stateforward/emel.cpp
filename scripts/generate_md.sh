#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_MD_DIR="$ROOT_DIR/docs/architecture"
OUT_MERMAID_DIR="$ROOT_DIR/docs/architecture/mermaid"
BUILD_DIR="$ROOT_DIR/build/debug"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$OUT_MD_DIR" "$OUT_MERMAID_DIR"

if ! command -v cmake >/dev/null 2>&1; then
  echo "error: required tool missing: cmake" >&2
  exit 1
fi

if ! command -v zig >/dev/null 2>&1; then
  echo "error: required tool missing: zig" >&2
  exit 1
fi

cmake -S "$ROOT_DIR" --preset debug >/dev/null

BOOST_SML_INCLUDE="$BUILD_DIR/_deps/boost_sml-src/include"
if [[ ! -d "$BOOST_SML_INCLUDE" ]]; then
  echo "error: boost_sml include dir not found: $BOOST_SML_INCLUDE" >&2
  exit 1
fi

sm_headers=()
while IFS= read -r h; do
  sm_headers+=("$h")
done < <(find "$ROOT_DIR/src/emel" -type f -name 'sm.hpp' \
  ! -path "$ROOT_DIR/src/emel/sm.hpp" | sort)

if [[ ${#sm_headers[@]} -eq 0 ]]; then
  echo "error: no sm.hpp files found under $ROOT_DIR/src/emel" >&2
  exit 1
fi

CPP_FILE="$TMP_DIR/generate_dynamic_md.cpp"
BIN_FILE="$TMP_DIR/generate_dynamic_md"
GEN_MD_DIR="$TMP_DIR/generated_md"
GEN_MERMAID_DIR="$TMP_DIR/generated_mermaid"
mkdir -p "$GEN_MD_DIR" "$GEN_MERMAID_DIR"

{
  echo '#include <boost/sml.hpp>'
  echo '#include <fstream>'
  echo '#include <iostream>'
  echo '#include <sstream>'
  echo '#include <string>'
  echo

  for h in "${sm_headers[@]}"; do
    rel="${h#"$ROOT_DIR/src/"}"
    echo "#include \"$rel\""
  done

  cat <<'CPP'

template <class T>
std::string sml_name() noexcept {
  namespace sml = boost::sml;
  return std::string{sml::aux::string<T>{}.c_str()};
}

inline bool is_none_like(const std::string & name) noexcept {
  return name.empty() || name == "none" || name == "zero_wrapper";
}

template <class T>
void dump_transition(std::ostream & md, std::ostream & mermaid) noexcept {
  namespace sml = boost::sml;

  auto src_state = sml_name<typename T::src_state>();
  auto dst_state = sml_name<typename T::dst_state>();

  if (dst_state == "X" || dst_state == "terminate") {
    dst_state = "[*]";
  }

  if (T::initial) {
    mermaid << "  [*] --> " << src_state << "\n";
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

  mermaid << "  " << src_state << " --> " << dst_state;
  if (!event.empty() || !guard.empty() || !action.empty()) {
    mermaid << " :";
    if (!event.empty()) {
      mermaid << " " << event;
    }
    if (!guard.empty()) {
      mermaid << " [" << guard << "]";
    }
    if (!action.empty()) {
      mermaid << " / " << action;
    }
  }
  mermaid << "\n";

  md << "| `" << src_state << "` | ";
  if (event.empty()) {
    md << "-";
  } else {
    md << "`" << event << "`";
  }
  md << " | ";
  if (guard.empty()) {
    md << "-";
  } else {
    md << "`" << guard << "`";
  }
  md << " | ";
  if (action.empty()) {
    md << "-";
  } else {
    md << "`" << action << "`";
  }
  md << " | `" << dst_state << "` |\n";
}

template <class TTransitions>
void dump_transitions(const TTransitions &, std::ostream &, std::ostream &) noexcept {}

template <template <class...> class T, class... Ts>
void dump_transitions(const T<Ts...> &, std::ostream & md, std::ostream & mermaid) noexcept {
  int _[]{0, (dump_transition<Ts>(md, mermaid), 0)...};
  (void)_;
}

template <class TSM>
void dump_machine_doc(
    const std::string & machine_name, const std::string & source_header, std::ostream & md,
    std::ostream & mermaid) noexcept {
  namespace sml = boost::sml;
  using transitions_t = typename sml::sm<TSM>::transitions;

  std::ostringstream table_rows;
  std::ostringstream mermaid_rows;
  dump_transitions(transitions_t{}, table_rows, mermaid_rows);

  md << "# " << machine_name << "\n\n";
  md << "Source: `" << source_header << "`\n\n";
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
    rel="${h#"$ROOT_DIR/src/"}"
    rel_emel="${h#"$ROOT_DIR/src/emel/"}"
    dir="${rel_emel%/sm.hpp}"
    name="${dir//\//_}"
    type="emel::${dir//\//::}::model"
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
} > "$CPP_FILE"

zig c++ -std=c++20 \
  -I"$ROOT_DIR/src" \
  -I"$ROOT_DIR/include" \
  -I"$BOOST_SML_INCLUDE" \
  "$CPP_FILE" -o "$BIN_FILE"

"$BIN_FILE" "$GEN_MD_DIR" "$GEN_MERMAID_DIR"

if [[ "${1:-}" == "--check" ]]; then
  for h in "${sm_headers[@]}"; do
    rel="${h#"$ROOT_DIR/src/emel/"}"
    dir="${rel%/sm.hpp}"
    name="${dir//\//_}"

    if [[ ! -f "$OUT_MD_DIR/$name.md" ]]; then
      echo "error: missing $OUT_MD_DIR/$name.md" >&2
      exit 1
    fi
    if [[ ! -f "$OUT_MERMAID_DIR/$name.mmd" ]]; then
      echo "error: missing $OUT_MERMAID_DIR/$name.mmd" >&2
      exit 1
    fi
    if ! diff -u "$OUT_MD_DIR/$name.md" "$GEN_MD_DIR/$name.md"; then
      echo "error: markdown out of sync ($name)" >&2
      exit 1
    fi
    if ! diff -u "$OUT_MERMAID_DIR/$name.mmd" "$GEN_MERMAID_DIR/$name.mmd"; then
      echo "error: mermaid out of sync ($name)" >&2
      exit 1
    fi
  done
  exit 0
fi

for h in "${sm_headers[@]}"; do
  rel="${h#"$ROOT_DIR/src/emel/"}"
  dir="${rel%/sm.hpp}"
  name="${dir//\//_}"
  cp "$GEN_MD_DIR/$name.md" "$OUT_MD_DIR/$name.md"
  cp "$GEN_MERMAID_DIR/$name.mmd" "$OUT_MERMAID_DIR/$name.mmd"
  echo "updated: $OUT_MD_DIR/$name.md"
  echo "updated: $OUT_MERMAID_DIR/$name.mmd"
done
