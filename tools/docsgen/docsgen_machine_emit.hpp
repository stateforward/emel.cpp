#pragma once

#include <stateforward/sml.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "emel/docs/detail.hpp"
#include "emel/emel.h"

struct doc_paths {
  std::filesystem::path root;
  std::filesystem::path docs_dir;
  std::filesystem::path architecture_dir;
  std::filesystem::path mermaid_dir;
  std::filesystem::path benchmarks_md;
  std::filesystem::path benchmarks_snapshot;
  std::filesystem::path embedded_size_snapshot;
  std::filesystem::path benchmarks_template;
  std::filesystem::path readme_template;
  std::filesystem::path readme_path;
};

struct machine_spec {
  std::string name;
  std::string source_path;
  void (*emit)(const machine_spec &spec, const doc_paths &paths, bool check);
};

inline std::string machine_ownership_note(const std::string &name) {
  if (name == "io_loader") {
    return "`emel/io` owns loading strategy-boundary events and failure "
           "routing. It does "
           "not own tensor residency, and concrete mmap/read/copy/async "
           "strategies remain "
           "unsupported until strategy actors land.";
  }
  if (name == "model_tensor") {
    return "`model/tensor` owns tensor bind, load, evict, and residency "
           "lifecycle "
           "semantics. It may emit I/O strategy effect requests, but residency "
           "remains "
           "tensor-owned.";
  }
  if (name == "model_loader") {
    return "`model/loader` orchestrates parser callbacks, the tensor actor, "
           "and the I/O "
           "actor. It must not implement low-level file APIs or tensor "
           "residency "
           "lifecycle.";
  }
  return {};
}

std::string md_link(const std::string &label, const std::string &source_path);
bool write_file(const std::filesystem::path &path, const std::string &content,
                bool check);

template <class... Ts, class fn>
constexpr void for_each_type(stateforward::sml::aux::type_list<Ts...>,
                             fn &&visitor) {
  (visitor.template operator()<Ts>(), ...);
}

template <class T> std::string mermaid_state_name() {
  return emel::docs::detail::mermaid_label(
      emel::docs::detail::raw_type_name<T>());
}

template <class T> std::string table_name() {
  if constexpr (std::is_same_v<T, stateforward::sml::back::anonymous>) {
    return "-";
  }
  return emel::docs::detail::shorten_type_name(
      emel::docs::detail::raw_type_name<T>());
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
void emit_machine(const machine_spec &spec, const doc_paths &paths,
                  bool check) {
  using sm_t = stateforward::sml::sm<model>;
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
    row.guard_mermaid = emel::docs::detail::mermaid_label(
        emel::docs::detail::raw_type_name<guard>());
    row.action_mermaid = emel::docs::detail::mermaid_label(
        emel::docs::detail::raw_type_name<action>());

    row.src = table_name<src_state>();
    row.dst = table_name<dst_state>();
    row.event = emel::docs::detail::table_event_name<event>();
    row.guard = emel::docs::detail::shorten_type_name(
        emel::docs::detail::raw_type_name<guard>());
    row.action = emel::docs::detail::shorten_type_name(
        emel::docs::detail::raw_type_name<action>());
    row.is_anonymous_event =
        std::is_same_v<event, stateforward::sml::back::anonymous>;

    rows.push_back(std::move(row));
  });

  std::string mermaid;
  mermaid += "stateDiagram-v2\n";
  mermaid += "  direction TB\n";
  for (const auto &initial : initial_states) {
    mermaid += "  [*] --> ";
    mermaid += initial;
    mermaid += "\n";
  }
  for (const auto &row : rows) {
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
  for (const auto &row : rows) {
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
  const std::string ownership_note = machine_ownership_note(spec.name);
  if (!ownership_note.empty()) {
    doc += "## Ownership\n\n";
    doc += ownership_note;
    doc += "\n\n";
  }
  doc += "## Mermaid\n\n";
  doc += "```mermaid\n";
  doc += mermaid;
  doc += "```\n\n";
  doc += "## Transitions\n\n";
  doc += table;

  const std::filesystem::path md_path =
      paths.architecture_dir / (spec.name + ".md");
  const std::filesystem::path mmd_path =
      paths.mermaid_dir / (spec.name + ".mmd");
  if (!write_file(md_path, doc, check)) {
    std::exit(1);
  }
  if (!write_file(mmd_path, mermaid, check)) {
    std::exit(1);
  }
}

template <class model>
void register_machine(std::vector<machine_spec> &out, const char *name,
                      const char *source_path) {
  machine_spec spec;
  spec.name = name;
  spec.source_path = source_path;
  spec.emit = &emit_machine<model>;
  out.push_back(std::move(spec));
}
