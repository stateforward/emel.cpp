#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/docs/architecture/puml"
BUILD_DIR="$ROOT_DIR/build/debug"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$OUT_DIR"

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

CPP_FILE="$TMP_DIR/generate_dynamic_puml.cpp"
BIN_FILE="$TMP_DIR/generate_dynamic_puml"
GEN_DIR="$TMP_DIR/generated"
mkdir -p "$GEN_DIR"

{
  echo '#include <boost/sml.hpp>'
  echo '#include <fstream>'
  echo '#include <iostream>'
  echo '#include <string>'
  echo

  for h in "${sm_headers[@]}"; do
    rel="${h#"$ROOT_DIR/src/"}"
    echo "#include \"$rel\""
  done

  cat <<'CPP'

template <class T>
void dump_transition(std::ostream & out) noexcept {
  namespace sml = boost::sml;
  auto src_state = std::string{sml::aux::string<typename T::src_state>{}.c_str()};
  auto dst_state = std::string{sml::aux::string<typename T::dst_state>{}.c_str()};

  if (dst_state == "X" || dst_state == "terminate") {
    dst_state = "[*]";
  }

  if (T::initial) {
    out << "[*] --> " << src_state << "\n";
  }

  const auto has_event = !sml::aux::is_same<typename T::event, sml::anonymous>::value;
  const auto has_guard = !sml::aux::is_same<typename T::guard, sml::front::always>::value;

  out << src_state << " --> " << dst_state;

  if (has_event || has_guard) {
    out << " :";
  }

  if (has_event) {
    out << " " << sml::aux::string<typename T::event>{}.c_str();
  }

  if (has_guard) {
    out << "\\n [" << sml::aux::string<typename T::guard>{}.c_str() << "]";
  }

  out << "\n";
}

template <class TTransitions>
void dump_transitions(const TTransitions &, std::ostream & out) noexcept {}

template <template <class...> class T, class... Ts>
void dump_transitions(const T<Ts...> &, std::ostream & out) noexcept {
  int _[]{0, (dump_transition<Ts>(out), 0)...};
  (void)_;
}

template <class TSM>
void dump_model(std::ostream & out) noexcept {
  namespace sml = boost::sml;
  out << "@startuml\n\n";
  dump_transitions(typename sml::sm<TSM>::transitions{}, out);
  out << "\n@enduml\n";
}

int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr << "usage: generate_dynamic_puml <out_dir>\n";
    return 1;
  }
  const std::string out_dir = argv[1];
CPP

  for h in "${sm_headers[@]}"; do
    rel="${h#"$ROOT_DIR/src/emel/"}"
    dir="${rel%/sm.hpp}"
    name="${dir//\//_}"
    type="emel::${dir//\//::}::model"
    echo "  { std::ofstream f(out_dir + \"/$name.puml\"); dump_model<$type>(f); }"
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

"$BIN_FILE" "$GEN_DIR"

if [[ "${1:-}" == "--check" ]]; then
  for h in "${sm_headers[@]}"; do
    rel="${h#"$ROOT_DIR/src/emel/"}"
    dir="${rel%/sm.hpp}"
    name="${dir//\//_}"
    if [[ ! -f "$OUT_DIR/$name.puml" ]]; then
      echo "error: missing $OUT_DIR/$name.puml" >&2
      exit 1
    fi
    if ! diff -u "$OUT_DIR/$name.puml" "$GEN_DIR/$name.puml"; then
      echo "error: puml out of sync ($name)" >&2
      exit 1
    fi
  done
  exit 0
fi

for h in "${sm_headers[@]}"; do
  rel="${h#"$ROOT_DIR/src/emel/"}"
  dir="${rel%/sm.hpp}"
  name="${dir//\//_}"
  cp "$GEN_DIR/$name.puml" "$OUT_DIR/$name.puml"
  echo "updated: $OUT_DIR/$name.puml"
done
