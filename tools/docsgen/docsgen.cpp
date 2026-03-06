#include <boost/sml.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <optional>
#include <string>
#include <string_view>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <vector>

#include "emel/emel.h"
#include "emel/docs/detail.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/sm.hpp"

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

bool prune_stale_generated_files(const fs::path & dir,
                                 const std::string_view extension,
                                 const std::unordered_set<std::string> & expected_stems,
                                 bool check) {
  if (!fs::exists(dir)) {
    return true;
  }

  for (const auto & entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != extension) {
      continue;
    }

    const std::string stem = entry.path().stem().string();
    if (expected_stems.contains(stem)) {
      continue;
    }

    if (check) {
      std::fprintf(stderr, "error: stale generated file %s\n", entry.path().string().c_str());
      return false;
    }

    std::error_code ec;
    fs::remove(entry.path(), ec);
    if (ec) {
      std::fprintf(stderr, "error: unable to remove %s\n", entry.path().string().c_str());
      return false;
    }
  }

  return true;
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
std::string mermaid_state_name() {
  return emel::docs::detail::mermaid_label(emel::docs::detail::raw_type_name<T>());
}

template <class T>
std::string table_name() {
  if constexpr (std::is_same_v<T, boost::sml::back::anonymous>) {
    return "-";
  }
  return emel::docs::detail::shorten_type_name(emel::docs::detail::raw_type_name<T>());
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
    row.event_mermaid = emel::docs::detail::mermaid_event_name<event>();
    row.guard_mermaid =
        emel::docs::detail::mermaid_label(emel::docs::detail::raw_type_name<guard>());
    row.action_mermaid =
        emel::docs::detail::mermaid_label(emel::docs::detail::raw_type_name<action>());

    row.src = table_name<src_state>();
    row.dst = table_name<dst_state>();
    row.event = emel::docs::detail::table_event_name<event>();
    row.guard = emel::docs::detail::shorten_type_name(emel::docs::detail::raw_type_name<guard>());
    row.action =
        emel::docs::detail::shorten_type_name(emel::docs::detail::raw_type_name<action>());
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

bool parser_parse_done_sink(const emel::text::jinja::events::parsing_done &) {
  return true;
}

bool parser_parse_error_sink(const emel::text::jinja::events::parsing_error &) {
  return true;
}

const std::string * find_template_var(const std::unordered_map<std::string, std::string> & vars,
                                      const std::string_view key) {
  const auto it = vars.find(std::string(key));
  if (it == vars.end()) {
    return nullptr;
  }
  return &it->second;
}

bool append_rendered_node(std::string & out,
                          const emel::text::jinja::ast_node * const node,
                          const std::unordered_map<std::string, std::string> & vars) {
  if (const auto * text = dynamic_cast<const emel::text::jinja::string_literal *>(node);
      text != nullptr) {
    out += text->value;
    return true;
  }

  if (const auto * id = dynamic_cast<const emel::text::jinja::identifier *>(node);
      id != nullptr) {
    const auto * value = find_template_var(vars, id->name);
    if (value == nullptr) {
      std::fprintf(stderr, "error: template variable not provided: %s\n", id->name.c_str());
      return false;
    }
    out += *value;
    return true;
  }

  if (dynamic_cast<const emel::text::jinja::comment_statement *>(node) != nullptr ||
      dynamic_cast<const emel::text::jinja::noop_statement *>(node) != nullptr) {
    return true;
  }

  std::fprintf(stderr, "error: unsupported template AST node\n");
  return false;
}

std::optional<std::string> render_template(const fs::path & template_path,
                                           const std::vector<template_var> & vars) {
  const std::string template_text = read_file(template_path);
  if (template_text.empty()) {
    std::fprintf(stderr, "error: unable to read %s\n",
                 template_path.string().c_str());
    return std::nullopt;
  }

  emel::text::jinja::program program;
  int32_t parse_err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t parse_error_pos = 0;
  emel::text::jinja::parser::action::context parse_ctx;
  emel::text::jinja::parser::sm parser{parse_ctx};
  emel::text::jinja::event::parse parse_ev{
      template_text,
      program,
      emel::text::jinja::event::parse::done_callback::from<&parser_parse_done_sink>(),
      emel::text::jinja::event::parse::error_callback::from<&parser_parse_error_sink>(),
      parse_err,
      parse_error_pos,
  };

  parser.process_event(parse_ev);
  if (parse_err != static_cast<int32_t>(emel::text::jinja::parser::error::none) ||
      !parser.is(boost::sml::state<emel::text::jinja::parser::done>)) {
    std::fprintf(stderr, "error: jinja parse failed\n");
    return std::nullopt;
  }

  std::unordered_map<std::string, std::string> template_vars;
  template_vars.reserve(vars.size());
  for (const auto & var : vars) {
    template_vars.insert_or_assign(var.key, var.value);
  }

  std::size_t estimate = template_text.size();
  for (const auto & var : vars) {
    estimate += var.value.size();
  }

  std::string rendered;
  rendered.reserve(estimate);
  for (const auto & node : program.body) {
    if (!append_rendered_node(rendered, node.get(), template_vars)) {
      return std::nullopt;
    }
  }

  return rendered;
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

  std::unordered_set<std::string> machine_names;
  machine_names.reserve(machines.size());
  for (const auto & spec : machines) {
    machine_names.insert(spec.name);
  }

  if (!prune_stale_generated_files(paths.architecture_dir, ".md", machine_names, opts.check)) {
    return 1;
  }
  if (!prune_stale_generated_files(paths.mermaid_dir, ".mmd", machine_names, opts.check)) {
    return 1;
  }

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
