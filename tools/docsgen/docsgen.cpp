#include <boost/sml.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <vector>

#include "emel/emel.h"
#include "emel/jinja/parser/sm.hpp"
#include "emel/jinja/renderer/sm.hpp"
#include "emel/jinja/value.hpp"

namespace fs = std::filesystem;

struct doc_paths {
  fs::path root;
  fs::path docs_dir;
  fs::path architecture_dir;
  fs::path mermaid_dir;
  fs::path benchmarks_md;
  fs::path benchmarks_snapshot;
  fs::path benchmarks_template;
  fs::path readme_template;
  fs::path readme_path;
};

struct machine_spec {
  std::string name;
  std::string source_path;
  void (*emit)(const machine_spec & spec, const doc_paths & paths, bool check);
};

struct options {
  fs::path root;
  bool check = false;
};

std::string read_file(const fs::path & path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

bool write_file(const fs::path & path, const std::string & content, bool check) {
  if (check) {
    std::string existing = read_file(path);
    if (existing != content) {
      std::fprintf(stderr, "error: %s is out of date\n", path.string().c_str());
      return false;
    }
    return true;
  }

  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    std::fprintf(stderr, "error: unable to write %s\n", path.string().c_str());
    return false;
  }
  output << content;
  return static_cast<bool>(output);
}

std::string sanitize_mermaid(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  for (std::size_t i = 0; i < name.size(); ++i) {
    const char ch = name[i];
    if (ch == ':' && i + 1 < name.size() && name[i + 1] == ':') {
      out.push_back('_');
      out.push_back('_');
      ++i;
      continue;
    }
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) != 0 || ch == '_') {
      out.push_back(ch);
      continue;
    }
    out.push_back('_');
  }
  return out;
}

std::string shorten_type_name(std::string_view name) {
  std::string out(name);
  const std::size_t pos = out.rfind("::");
  if (pos != std::string::npos) {
    out = out.substr(pos + 2);
  }
  if (out.find("lambda at ") != std::string::npos) {
    std::replace(out.begin(), out.end(), '<', '(');
  }
  return out;
}

std::string md_link(const std::string & label, const std::string & source_path) {
  std::string link = "https://github.com/stateforward/emel.cpp/blob/main/src/";
  link += source_path;
  std::string out = "[`";
  out += label;
  out += "`](";
  out += link;
  out += ")";
  return out;
}

std::string build_docs_toc(const std::vector<machine_spec> & machines) {
  std::string toc;
  toc += "- [`docs/benchmarks.md`](docs/benchmarks.md)\n";
  for (const auto & spec : machines) {
    toc += "- [`docs/architecture/";
    toc += spec.name;
    toc += ".md`](docs/architecture/";
    toc += spec.name;
    toc += ".md)\n";
  }
  return toc;
}

bool parse_options(int argc, char ** argv, options & out) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--check") {
      out.check = true;
      continue;
    }
    if (arg == "--root" && i + 1 < argc) {
      out.root = fs::path(argv[++i]);
      continue;
    }
    std::fprintf(stderr, "usage: docsgen --root <path> [--check]\n");
    return false;
  }

  if (out.root.empty()) {
    std::fprintf(stderr, "error: --root is required\n");
    return false;
  }
  return true;
}

template <class... Ts, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

template <class T>
std::string raw_type_name() {
  return boost::sml::aux::string<T>::c_str();
}

template <class>
struct is_unexpected_event : std::false_type {};

template <class T, class event>
struct is_unexpected_event<boost::sml::back::unexpected_event<T, event>>
    : std::true_type {};

template <class event>
std::string event_type_name() {
  if constexpr (is_unexpected_event<event>::value) {
    using mapped = boost::sml::back::get_event_t<event>;
    return raw_type_name<mapped>();
  }
  return raw_type_name<event>();
}

template <class T>
std::string mermaid_state_name() {
  return sanitize_mermaid(raw_type_name<T>());
}

template <class T>
std::string mermaid_event_name() {
  if constexpr (std::is_same_v<T, boost::sml::back::anonymous>) {
    return {};
  }
  return sanitize_mermaid(event_type_name<T>());
}

template <class T>
std::string table_name() {
  if constexpr (std::is_same_v<T, boost::sml::back::anonymous>) {
    return "-";
  }
  return shorten_type_name(raw_type_name<T>());
}

template <class T>
std::string table_event_name() {
  if constexpr (std::is_same_v<T, boost::sml::back::anonymous>) {
    return "-";
  }
  return shorten_type_name(event_type_name<T>());
}

struct transition_row {
  std::string src_mermaid;
  std::string dst_mermaid;
  std::string event_mermaid;
  std::string guard_mermaid;
  std::string action_mermaid;
  std::string src;
  std::string dst;
  std::string event;
  std::string guard;
  std::string action;
  bool is_anonymous_event = false;
};

template <class model>
void emit_machine(const machine_spec & spec, const doc_paths & paths, bool check) {
  using sm_t = boost::sml::sm<model>;
  using transitions = typename sm_t::transitions;

  std::vector<transition_row> rows;
  std::vector<std::string> initial_states;
  std::unordered_set<std::string> initial_seen;

  for_each_type(transitions{}, [&]<class transition_t>() {
    using src_state = typename transition_t::src_state;
    using dst_state = typename transition_t::dst_state;
    using event = typename transition_t::event;
    using guard = typename transition_t::guard;
    using action = typename transition_t::action;

    if (transition_t::initial) {
      std::string initial_name = mermaid_state_name<src_state>();
      if (initial_seen.insert(initial_name).second) {
        initial_states.push_back(initial_name);
      }
    }

    transition_row row;
    row.src_mermaid = mermaid_state_name<src_state>();
    row.dst_mermaid = mermaid_state_name<dst_state>();
    row.event_mermaid = mermaid_event_name<event>();
    row.guard_mermaid = sanitize_mermaid(raw_type_name<guard>());
    row.action_mermaid = sanitize_mermaid(raw_type_name<action>());

    row.src = table_name<src_state>();
    row.dst = table_name<dst_state>();
    row.event = table_event_name<event>();
    row.guard = shorten_type_name(raw_type_name<guard>());
    row.action = shorten_type_name(raw_type_name<action>());
    row.is_anonymous_event = std::is_same_v<event, boost::sml::back::anonymous>;

    rows.push_back(std::move(row));
  });

  std::string mermaid;
  mermaid += "stateDiagram-v2\n";
  mermaid += "  direction TB\n";
  for (const auto & initial : initial_states) {
    mermaid += "  [*] --> ";
    mermaid += initial;
    mermaid += "\n";
  }
  for (const auto & row : rows) {
    mermaid += "  ";
    mermaid += row.src_mermaid;
    mermaid += " --> ";
    mermaid += row.dst_mermaid;
    mermaid += " :";
    if (!row.event_mermaid.empty()) {
      mermaid += " ";
      mermaid += row.event_mermaid;
    }
    mermaid += " [";
    mermaid += row.guard_mermaid;
    mermaid += "] / ";
    mermaid += row.action_mermaid;
    mermaid += "\n";
  }

  std::string table;
  table += "| Source | Event | Guard | Action | Target |\n";
  table += "| --- | --- | --- | --- | --- |\n";
  for (const auto & row : rows) {
    table += "| ";
    table += md_link(row.src, spec.source_path);
    table += " | ";
    if (row.is_anonymous_event) {
      table += "-";
    } else {
      table += md_link(row.event, spec.source_path);
    }
    table += " | ";
    table += md_link(row.guard, spec.source_path);
    table += " | ";
    table += md_link(row.action, spec.source_path);
    table += " | ";
    table += md_link(row.dst, spec.source_path);
    table += " |\n";
  }

  std::string doc;
  doc += "# ";
  doc += spec.name;
  doc += "\n\n";
  doc += "Source: ";
  doc += md_link(spec.source_path, spec.source_path);
  doc += "\n\n";
  doc += "## Mermaid\n\n";
  doc += "```mermaid\n";
  doc += mermaid;
  doc += "```\n\n";
  doc += "## Transitions\n\n";
  doc += table;

  const fs::path md_path = paths.architecture_dir / (spec.name + ".md");
  const fs::path mmd_path = paths.mermaid_dir / (spec.name + ".mmd");
  if (!write_file(md_path, doc, check)) {
    std::exit(1);
  }
  if (!write_file(mmd_path, mermaid, check)) {
    std::exit(1);
  }
}

template <class model>
void register_machine(std::vector<machine_spec> & out,
                      const char * name,
                      const char * source_path) {
  machine_spec spec;
  spec.name = name;
  spec.source_path = source_path;
  spec.emit = &emit_machine<model>;
  out.push_back(std::move(spec));
}

#include "docsgen_machines.hpp"

struct template_var {
  std::string key;
  std::string value;
};

std::optional<std::string> render_template(const fs::path & template_path,
                                           const std::vector<template_var> & vars) {
  const std::string template_text = read_file(template_path);
  if (template_text.empty()) {
    std::fprintf(stderr, "error: unable to read %s\n",
                 template_path.string().c_str());
    return std::nullopt;
  }

  emel::jinja::program program;
  int32_t parse_err = EMEL_OK;
  emel::jinja::parser::action::context parse_ctx;
  emel::jinja::parser::sm parser{parse_ctx};
  emel::jinja::event::parse parse_ev{
    .template_text = template_text,
    .program_out = &program,
    .error_out = &parse_err,
  };

  parser.process_event(parse_ev);
  if (parse_err != EMEL_OK ||
      !parser.is(boost::sml::state<emel::jinja::parser::done>)) {
    std::fprintf(stderr, "error: jinja parse failed\n");
    return std::nullopt;
  }

  std::vector<emel::jinja::object_entry> entries;
  entries.reserve(vars.size());
  for (const auto & var : vars) {
    emel::jinja::value key;
    key.type = emel::jinja::value_type::string;
    key.string_v.view = var.key;

    emel::jinja::value val;
    val.type = emel::jinja::value_type::string;
    val.string_v.view = var.value;

    emel::jinja::object_entry entry;
    entry.key = key;
    entry.val = val;
    entries.push_back(entry);
  }

  emel::jinja::object_value globals{};
  if (!entries.empty()) {
    globals.entries = entries.data();
    globals.count = entries.size();
    globals.capacity = entries.size();
    globals.has_builtins = false;
  }

  std::size_t estimate = template_text.size() + 8192;
  for (const auto & var : vars) {
    estimate += var.value.size();
  }

  std::vector<char> output_buffer;
  output_buffer.resize(estimate);

  size_t out_len = 0;
  size_t error_pos = 0;
  int32_t render_err = EMEL_OK;
  emel::jinja::renderer::action::context render_ctx;
  emel::jinja::renderer::sm renderer{render_ctx};
  emel::jinja::event::render render_ev{
    .program = &program,
    .globals = entries.empty() ? nullptr : &globals,
    .source_text = template_text,
    .output = output_buffer.data(),
    .output_capacity = output_buffer.size(),
    .output_length = &out_len,
    .error_out = &render_err,
    .error_pos_out = &error_pos,
  };

  renderer.process_event(render_ev);
  if (render_err != EMEL_OK ||
      !renderer.is(boost::sml::state<emel::jinja::renderer::done>)) {
    std::fprintf(stderr, "error: jinja render failed\n");
    return std::nullopt;
  }

  return std::string(output_buffer.data(), out_len);
}

std::optional<std::string> build_benchmarks_table(const doc_paths & paths) {
  const std::string snapshot = read_file(paths.benchmarks_snapshot);
  if (snapshot.empty()) {
    std::fprintf(stderr, "error: unable to read %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }

  std::vector<std::string> rows;
  std::regex line_re(
      R"(^([^ ]+) emel\.cpp ([0-9.]+) ns/op, llama\.cpp ([0-9.]+) ns/op, ratio=([0-9.]+)x$)");

  std::istringstream input(snapshot);
  for (std::string line; std::getline(input, line);) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::smatch match;
    if (!std::regex_match(line, match, line_re)) {
      continue;
    }
    const std::string name = match[1].str();
    const std::string emel_ns = match[2].str();
    const std::string llama_ns = match[3].str();
    const std::string ratio = match[4].str();
    std::string row = "| `" + name + "` | ";
    row += emel_ns;
    row += " | ";
    row += llama_ns;
    row += " | ";
    row += ratio;
    row += "x |";
    rows.push_back(std::move(row));
  }

  std::string table;
  table += "| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |\n";
  table += "| --- | ---: | ---: | ---: |\n";
  for (const auto & row : rows) {
    table += row;
    table += "\n";
  }
  return table;
}

int main(int argc, char ** argv) {
  options opts;
  if (!parse_options(argc, argv, opts)) {
    return 1;
  }

  doc_paths paths;
  paths.root = opts.root;
  paths.docs_dir = paths.root / "docs";
  paths.architecture_dir = paths.docs_dir / "architecture";
  paths.mermaid_dir = paths.architecture_dir / "mermaid";
  paths.benchmarks_md = paths.docs_dir / "benchmarks.md";
  paths.benchmarks_snapshot = paths.root / "snapshots/bench/benchmarks_compare.txt";
  paths.benchmarks_template = paths.docs_dir / "templates/benchmarks.md.j2";
  paths.readme_template = paths.docs_dir / "templates/README.md.j2";
  paths.readme_path = paths.root / "README.md";

  std::vector<machine_spec> machines;
  register_machines(machines);

  for (const auto & spec : machines) {
    spec.emit(spec, paths, opts.check);
  }

  const std::string docs_toc = build_docs_toc(machines);
  const auto benchmarks_table = build_benchmarks_table(paths);
  if (!benchmarks_table.has_value()) {
    return 1;
  }

  const auto benchmarks_doc = render_template(
      paths.benchmarks_template,
      {template_var{"benchmarks_table", *benchmarks_table}});
  if (!benchmarks_doc.has_value()) {
    return 1;
  }
  if (!write_file(paths.benchmarks_md, *benchmarks_doc, opts.check)) {
    return 1;
  }

  const auto readme_doc = render_template(
      paths.readme_template,
      {template_var{"docs_toc", docs_toc}});
  if (!readme_doc.has_value()) {
    return 1;
  }
  if (!write_file(paths.readme_path, *readme_doc, opts.check)) {
    return 1;
  }

  return 0;
}
