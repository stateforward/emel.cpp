#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_MD_DIR="$ROOT_DIR/docs/architecture"
OUT_MERMAID_DIR="$ROOT_DIR/docs/architecture/mermaid"
BUILD_DIR="$ROOT_DIR/build/debug"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
REPO_URL="https://github.com/stateforward/emel.cpp"

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
  echo '#include <cctype>'
  echo

  for h in "${sm_headers[@]}"; do
    rel="${h#"$ROOT_DIR/src/"}"
    echo "#include \"$rel\""
  done

  sed "s|REPO_URL_PLACEHOLDER|$REPO_URL|g" <<'CPP'

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
