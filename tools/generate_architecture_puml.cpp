#include <boost/sml.hpp>

#include <fstream>
#include <iostream>
#include <string>

#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace sml = boost::sml;

template <class T>
void dump_transition(std::ostream & out) noexcept {
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
void dump_transitions(const TTransitions &, std::ostream &) noexcept {}

template <template <class...> class T, class... Ts>
void dump_transitions(const T<Ts...> &, std::ostream & out) noexcept {
  int _[]{0, (dump_transition<Ts>(out), 0)...};
  (void)_;
}

template <class TSM>
void dump_model(std::ostream & out) noexcept {
  out << "@startuml\n\n";
  dump_transitions(typename sml::sm<TSM>::transitions{}, out);
  out << "\n@enduml\n";
}

int main(int argc, char ** argv) {
  if (argc != 4) {
    std::cerr << "usage: generate_architecture_puml <loader.puml> <parser.puml> <weight_loader.puml>\n";
    return 1;
  }

  {
    std::ofstream f(argv[1]);
    dump_model<emel::model::loader::model>(f);
  }
  {
    std::ofstream f(argv[2]);
    dump_model<emel::model::parser::model>(f);
  }
  {
    std::ofstream f(argv[3]);
    dump_model<emel::model::weight_loader::model>(f);
  }

  return 0;
}
