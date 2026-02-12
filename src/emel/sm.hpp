#pragma once

#include <boost/sml.hpp>
#include <utility>

namespace emel {

template <class Model, class... Policies>
class sm {
 public:
  using model_type = Model;
  using state_machine_type = boost::sml::sm<Model, Policies...>;

  sm() = default;
  ~sm() = default;

  sm(const sm &) = default;
  sm(sm &&) = default;
  sm & operator=(const sm &) = default;
  sm & operator=(sm &&) = default;

  template <class... Args>
  explicit sm(Args &&... args) : state_machine_(std::forward<Args>(args)...) {}

  template <class Event>
  bool process_event(const Event & ev) {
    return state_machine_.process_event(ev);
  }

  template <class State>
  bool is(State state = {}) const {
    return state_machine_.is(state);
  }

  template <class Visitor>
  void visit_current_states(Visitor && visitor) {
    state_machine_.visit_current_states(std::forward<Visitor>(visitor));
  }

 protected:
  state_machine_type & raw_sm() { return state_machine_; }
  const state_machine_type & raw_sm() const { return state_machine_; }

 private:
  state_machine_type state_machine_;
};

}  // namespace emel
