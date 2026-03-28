#include <boost/sml.hpp>

#include <charconv>
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
  fs::path generation_pre_arm_flash_optimized_baseline;
  fs::path benchmarks_template;
  fs::path readme_template;
  fs::path readme_path;
};

struct benchmark_row {
  std::string name;
  std::string emel_ns;
  std::string llama_ns;
  std::string ratio;
};

struct benchmark_snapshot {
  std::vector<benchmark_row> rows;
  std::string reference_source;
  std::string reference_ref;
  std::string benchmark_config;
  std::string formatter_contract;
  std::string flash_case;
  std::string flash_dispatch_calls;
  std::string optimized_flash_dispatch_calls;
  std::string shared_flash_dispatch_calls;
  std::string emel_decode_calls;
  std::string emel_logits_calls;
  std::string reference_decode_calls;
  std::string reference_logits_calls;
  std::string runtime_contract_case;
  std::string native_quantized_stage_count;
  std::string approved_dense_f32_stage_count;
  std::string disallowed_fallback_stage_count;
  std::string explicit_no_claim_stage_count;
  std::string quantized_case;
  std::string native_q8_0_dispatch_calls;
  std::string optimized_q2_dispatch_calls;
  std::string shared_q2_dispatch_calls;
  std::string optimized_q3_dispatch_calls;
  std::string shared_q3_dispatch_calls;
  std::string optimized_q6_dispatch_calls;
  std::string shared_q6_dispatch_calls;
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

  // Generated architecture outputs are intentionally flat directories; this prune pass is
  // non-recursive and only removes unexpected files directly under `dir`.
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

std::optional<std::unordered_map<std::string, std::string>>
parse_key_value_file(const fs::path & path) {
  const std::string content = read_file(path);
  if (content.empty()) {
    std::fprintf(stderr, "error: unable to read %s\n", path.string().c_str());
    return std::nullopt;
  }

  std::unordered_map<std::string, std::string> fields;
  std::istringstream input(content);
  for (std::string line; std::getline(input, line);) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || separator == 0u) {
      continue;
    }
    fields.emplace(line.substr(0u, separator), line.substr(separator + 1u));
  }
  return fields;
}

std::optional<std::unordered_map<std::string, std::string>>
parse_inline_key_value_fields(const std::string & line, const char * prefix) {
  const std::string_view prefix_view{prefix};
  if (line.rfind(prefix_view.data(), 0u) != 0) {
    return std::nullopt;
  }

  std::unordered_map<std::string, std::string> fields;
  std::istringstream input(line.substr(prefix_view.size()));
  for (std::string token; input >> token;) {
    const std::size_t separator = token.find('=');
    if (separator == std::string::npos || separator == 0u || separator + 1u >= token.size()) {
      return std::nullopt;
    }
    fields.emplace(token.substr(0u, separator), token.substr(separator + 1u));
  }

  return fields;
}

std::optional<benchmark_snapshot> parse_benchmarks_snapshot(const doc_paths & paths) {
  const std::string snapshot = read_file(paths.benchmarks_snapshot);
  if (snapshot.empty()) {
    std::fprintf(stderr, "error: unable to read %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }

  benchmark_snapshot parsed;
  const std::regex line_re(
      R"(^([^ ]+) emel\.cpp ([0-9.]+) ns/op, llama\.cpp ([0-9.]+) ns/op, ratio=([0-9.]+)x$)");
  constexpr std::string_view k_benchmark_config_prefix = "# benchmark_config: ";
  constexpr std::string_view k_generation_formatter_contract_prefix =
      "# generation_formatter_contract: ";

  std::istringstream input(snapshot);
  for (std::string line; std::getline(input, line);) {
    std::smatch match;
    if (line.rfind(k_benchmark_config_prefix.data(), 0u) == 0u) {
      parsed.benchmark_config = line.substr(k_benchmark_config_prefix.size());
      continue;
    }
    if (line.rfind(k_generation_formatter_contract_prefix.data(), 0u) == 0u) {
      parsed.formatter_contract = line.substr(k_generation_formatter_contract_prefix.size());
      continue;
    }
    if (const auto metadata = parse_inline_key_value_fields(line, "# reference_impl: ");
        metadata.has_value()) {
      const auto source_it = metadata->find("source");
      const auto ref_it = metadata->find("ref");
      if (source_it == metadata->end() || ref_it == metadata->end()) {
        std::fprintf(stderr,
                     "error: invalid # reference_impl metadata in %s\n",
                     paths.benchmarks_snapshot.string().c_str());
        return std::nullopt;
      }
      parsed.reference_source = source_it->second;
      parsed.reference_ref = ref_it->second;
      continue;
    }
    if (const auto metadata = parse_inline_key_value_fields(line, "# generation_flash_evidence: ");
        metadata.has_value()) {
      for (const char * field : {"case",
                                 "flash_dispatch_calls",
                                 "emel_decode_calls",
                                 "emel_logits_calls",
                                 "reference_decode_calls",
                                 "reference_logits_calls"}) {
        if (!metadata->contains(field)) {
          std::fprintf(stderr,
                       "error: invalid # generation_flash_evidence metadata in %s\n",
                       paths.benchmarks_snapshot.string().c_str());
          return std::nullopt;
        }
      }
      parsed.flash_case = metadata->at("case");
      parsed.flash_dispatch_calls = metadata->at("flash_dispatch_calls");
      parsed.optimized_flash_dispatch_calls =
          metadata->contains("optimized_flash_dispatch_calls")
              ? metadata->at("optimized_flash_dispatch_calls")
              : "0";
      parsed.shared_flash_dispatch_calls =
          metadata->contains("shared_flash_dispatch_calls")
              ? metadata->at("shared_flash_dispatch_calls")
              : "0";
      parsed.emel_decode_calls = metadata->at("emel_decode_calls");
      parsed.emel_logits_calls = metadata->at("emel_logits_calls");
      parsed.reference_decode_calls = metadata->at("reference_decode_calls");
      parsed.reference_logits_calls = metadata->at("reference_logits_calls");
      continue;
    }
    if (const auto metadata =
            parse_inline_key_value_fields(line, "# generation_quantized_evidence: ");
        metadata.has_value()) {
      for (const char * field : {"case", "native_q8_0_dispatch_calls"}) {
        if (!metadata->contains(field)) {
          std::fprintf(stderr,
                       "error: invalid # generation_quantized_evidence metadata in %s\n",
                       paths.benchmarks_snapshot.string().c_str());
          return std::nullopt;
        }
      }
      parsed.quantized_case = metadata->at("case");
      parsed.native_q8_0_dispatch_calls = metadata->at("native_q8_0_dispatch_calls");
      parsed.optimized_q2_dispatch_calls =
          metadata->contains("optimized_q2_dispatch_calls")
              ? metadata->at("optimized_q2_dispatch_calls")
              : "0";
      parsed.shared_q2_dispatch_calls =
          metadata->contains("shared_q2_dispatch_calls")
              ? metadata->at("shared_q2_dispatch_calls")
              : "0";
      parsed.optimized_q3_dispatch_calls =
          metadata->contains("optimized_q3_dispatch_calls")
              ? metadata->at("optimized_q3_dispatch_calls")
              : "0";
      parsed.shared_q3_dispatch_calls =
          metadata->contains("shared_q3_dispatch_calls")
              ? metadata->at("shared_q3_dispatch_calls")
              : "0";
      parsed.optimized_q6_dispatch_calls =
          metadata->contains("optimized_q6_dispatch_calls")
              ? metadata->at("optimized_q6_dispatch_calls")
              : "0";
      parsed.shared_q6_dispatch_calls =
          metadata->contains("shared_q6_dispatch_calls")
              ? metadata->at("shared_q6_dispatch_calls")
              : "0";
      continue;
    }
    if (const auto metadata =
            parse_inline_key_value_fields(line, "# generation_runtime_contract: ");
        metadata.has_value()) {
      for (const char * field : {"case",
                                 "native_quantized",
                                 "approved_dense_f32_by_contract",
                                 "disallowed_fallback",
                                 "explicit_no_claim"}) {
        if (!metadata->contains(field)) {
          std::fprintf(stderr,
                       "error: invalid # generation_runtime_contract metadata in %s\n",
                       paths.benchmarks_snapshot.string().c_str());
          return std::nullopt;
        }
      }
      parsed.runtime_contract_case = metadata->at("case");
      parsed.native_quantized_stage_count = metadata->at("native_quantized");
      parsed.approved_dense_f32_stage_count = metadata->at("approved_dense_f32_by_contract");
      parsed.disallowed_fallback_stage_count = metadata->at("disallowed_fallback");
      parsed.explicit_no_claim_stage_count = metadata->at("explicit_no_claim");
      continue;
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (!std::regex_match(line, match, line_re)) {
      continue;
    }
    parsed.rows.push_back(benchmark_row{
        .name = match[1].str(),
        .emel_ns = match[2].str(),
        .llama_ns = match[3].str(),
        .ratio = match[4].str(),
    });
  }

  if (parsed.reference_source.empty() || parsed.reference_ref.empty()) {
    std::fprintf(stderr,
                 "error: missing # reference_impl metadata in %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }
  if (parsed.benchmark_config.empty()) {
    std::fprintf(stderr,
                 "error: missing # benchmark_config metadata in %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }
  if (parsed.formatter_contract.empty()) {
    std::fprintf(stderr,
                 "error: missing # generation_formatter_contract metadata in %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }
  if (parsed.flash_case.empty()) {
    std::fprintf(stderr,
                 "error: missing # generation_flash_evidence metadata in %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }
  if (parsed.quantized_case.empty()) {
    std::fprintf(stderr,
                 "error: missing # generation_quantized_evidence metadata in %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }
  if (parsed.runtime_contract_case.empty()) {
    std::fprintf(stderr,
                 "error: missing # generation_runtime_contract metadata in %s\n",
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }

  return parsed;
}

std::optional<std::string> build_benchmarks_table(const benchmark_snapshot & snapshot) {
  std::string table;
  table += "| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |\n";
  table += "| --- | ---: | ---: | ---: |\n";
  for (const auto & row : snapshot.rows) {
    table += "| `";
    table += row.name;
    table += "` | ";
    table += row.emel_ns;
    table += " | ";
    table += row.llama_ns;
    table += " | ";
    table += row.ratio;
    table += "x |\n";
  }
  return table;
}

const benchmark_row * find_benchmark_row(const benchmark_snapshot & snapshot,
                                         const std::string & name) {
  for (const auto & row : snapshot.rows) {
    if (row.name == name) {
      return &row;
    }
  }
  return nullptr;
}

std::optional<double> parse_double_field(const std::string & raw_value,
                                         const char * field_name,
                                         const fs::path & source_path) {
  double value = 0.0;
  const char * const begin = raw_value.data();
  const char * const end = begin + raw_value.size();
  const auto result = std::from_chars(begin, end, value);
  const bool parsed_ok = result.ec == std::errc{} && result.ptr == end;
  if (!parsed_ok) {
    std::fprintf(stderr,
                 "error: invalid %s=%s in %s\n",
                 field_name,
                 raw_value.c_str(),
                 source_path.string().c_str());
    return std::nullopt;
  }
  return value;
}

std::optional<std::string> build_flash_publication_section(const doc_paths & paths,
                                                           const benchmark_snapshot & snapshot) {
  const benchmark_row * current = find_benchmark_row(snapshot, snapshot.flash_case);
  if (current == nullptr) {
    std::fprintf(stderr,
                 "error: missing benchmark row for %s in %s\n",
                 snapshot.flash_case.c_str(),
                 paths.benchmarks_snapshot.string().c_str());
    return std::nullopt;
  }

  std::string section;
  section += "## Current Generation Evidence\n\n";
  section += "- Source snapshot: `snapshots/bench/benchmarks_compare.txt`\n";
  section += "- `benchmark_config: ";
  section += snapshot.benchmark_config;
  section += "`\n";
  section += "- `reference_impl: source=";
  section += snapshot.reference_source;
  section += " ref=";
  section += snapshot.reference_ref;
  section += "`\n";
  section += "- `generation_formatter_contract: ";
  section += snapshot.formatter_contract;
  section += "`\n";
  section += "- `generation_flash_evidence: case=";
  section += snapshot.flash_case;
  section += " flash_dispatch_calls=";
  section += snapshot.flash_dispatch_calls;
  section += " optimized_flash_dispatch_calls=";
  section += snapshot.optimized_flash_dispatch_calls;
  section += " shared_flash_dispatch_calls=";
  section += snapshot.shared_flash_dispatch_calls;
  section += " emel_decode_calls=";
  section += snapshot.emel_decode_calls;
  section += " emel_logits_calls=";
  section += snapshot.emel_logits_calls;
  section += " reference_decode_calls=";
  section += snapshot.reference_decode_calls;
  section += " reference_logits_calls=";
  section += snapshot.reference_logits_calls;
  section += "`\n";
  section += "- Current compare row: `";
  section += current->name;
  section += " emel.cpp ";
  section += current->emel_ns;
  section += " ns/op, llama.cpp ";
  section += current->llama_ns;
  section += " ns/op, ratio=";
  section += current->ratio;
  section += "x`\n\n";
  section += "## Current Quantized Evidence\n\n";
  section += "- Source snapshot: `snapshots/bench/benchmarks_compare.txt`\n";
  section += "- `generation_runtime_contract: case=";
  section += snapshot.runtime_contract_case;
  section += " native_quantized=";
  section += snapshot.native_quantized_stage_count;
  section += " approved_dense_f32_by_contract=";
  section += snapshot.approved_dense_f32_stage_count;
  section += " disallowed_fallback=";
  section += snapshot.disallowed_fallback_stage_count;
  section += " explicit_no_claim=";
  section += snapshot.explicit_no_claim_stage_count;
  section += "`\n";
  section += "- `generation_quantized_evidence: case=";
  section += snapshot.quantized_case;
  section += " native_q8_0_dispatch_calls=";
  section += snapshot.native_q8_0_dispatch_calls;
  section += " optimized_q2_dispatch_calls=";
  section += snapshot.optimized_q2_dispatch_calls;
  section += " shared_q2_dispatch_calls=";
  section += snapshot.shared_q2_dispatch_calls;
  section += " optimized_q3_dispatch_calls=";
  section += snapshot.optimized_q3_dispatch_calls;
  section += " shared_q3_dispatch_calls=";
  section += snapshot.shared_q3_dispatch_calls;
  section += " optimized_q6_dispatch_calls=";
  section += snapshot.optimized_q6_dispatch_calls;
  section += " shared_q6_dispatch_calls=";
  section += snapshot.shared_q6_dispatch_calls;
  section += "`\n\n";
  section += "- Contract summary: the maintained canonical Qwen3 workload stayed on the approved "
             "runtime contract with native q8_0 projection and output dispatch, explicit "
             "dense-f32-by-contract token embedding and per-head Q/K RMS norm vectors, and no "
             "disallowed fallback or explicit no-claim branch on the supported path.\n\n";

  if (!fs::exists(paths.generation_pre_arm_flash_optimized_baseline)) {
    return section;
  }

  const auto baseline =
      parse_key_value_file(paths.generation_pre_arm_flash_optimized_baseline);
  if (!baseline.has_value()) {
    return std::nullopt;
  }

  for (const char * field : {"source_commit",
                             "baseline_ref",
                             "case",
                             "baseline_emel_ns",
                             "baseline_reference_ns",
                             "baseline_ratio"}) {
    if (!baseline->contains(field)) {
      std::fprintf(stderr,
                   "error: missing %s in %s\n",
                   field,
                   paths.generation_pre_arm_flash_optimized_baseline.string().c_str());
      return std::nullopt;
    }
  }

  section += "## Preserved ARM Flash Baseline\n\n";
  section += "- Preserved baseline artifact: "
             "`snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`\n";
  section += "- `source_commit=";
  section += baseline->at("source_commit");
  section += "`\n";
  section += "- `baseline_ref=";
  section += baseline->at("baseline_ref");
  section += "`\n";
  section += "- `case=";
  section += baseline->at("case");
  section += "`\n";
  section += "- `baseline_emel_ns=";
  section += baseline->at("baseline_emel_ns");
  section += "`\n";
  section += "- `baseline_reference_ns=";
  section += baseline->at("baseline_reference_ns");
  section += "`\n";
  section += "- `baseline_ratio=";
  section += baseline->at("baseline_ratio");
  section += "`\n";

  const std::string & baseline_case = baseline->at("case");
  const benchmark_row * current_baseline_case = find_benchmark_row(snapshot, baseline_case);
  if (current_baseline_case == nullptr) {
    section += "- Note: this preserved ARM flash baseline remains tied to the archived Llama "
               "canonical slice and is not directly compared against the current canonical "
               "Qwen3 publication because the benchmark case identity changed explicitly.\n";
    return section;
  }

  const auto baseline_emel =
      parse_double_field(baseline->at("baseline_emel_ns"),
                         "baseline_emel_ns",
                         paths.generation_pre_arm_flash_optimized_baseline);
  if (!baseline_emel.has_value()) {
    return std::nullopt;
  }

  const auto current_emel =
      parse_double_field(current_baseline_case->emel_ns,
                         "current_emel_ns",
                         paths.benchmarks_snapshot);
  if (!current_emel.has_value()) {
    return std::nullopt;
  }

  const double speedup = *baseline_emel / *current_emel;
  const double latency_drop_pct = ((*baseline_emel - *current_emel) / *baseline_emel) * 100.0;

  char speedup_buf[32];
  char latency_buf[32];
  std::snprintf(speedup_buf, sizeof(speedup_buf), "%.3fx", speedup);
  std::snprintf(latency_buf, sizeof(latency_buf), "%.1f", latency_drop_pct);
  section += "- `current_emel_ns=";
  section += current_baseline_case->emel_ns;
  section += "`\n";
  section += "- `current_reference_ns=";
  section += current_baseline_case->llama_ns;
  section += "`\n";
  section += "- `current_ratio=";
  section += current_baseline_case->ratio;
  section += "x`\n";
  section += "- `speedup=";
  section += speedup_buf;
  section += "`\n";
  section += "- `latency_drop_pct=";
  section += latency_buf;
  section += "`\n";

  return section;
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
  paths.generation_pre_arm_flash_optimized_baseline =
      paths.root / "snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt";
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
  const auto benchmarks_snapshot = parse_benchmarks_snapshot(paths);
  if (!benchmarks_snapshot.has_value()) {
    return 1;
  }
  const auto benchmarks_table = build_benchmarks_table(*benchmarks_snapshot);
  if (!benchmarks_table.has_value()) {
    return 1;
  }
  const auto flash_publication_section =
      build_flash_publication_section(paths, *benchmarks_snapshot);
  if (!flash_publication_section.has_value()) {
    return 1;
  }

  const auto benchmarks_doc = render_template(
      paths.benchmarks_template,
      {template_var{"flash_publication_section", *flash_publication_section},
       template_var{"benchmarks_table", *benchmarks_table}});
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
