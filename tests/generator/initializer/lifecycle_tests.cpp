#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/docs/detail.hpp"
#include "emel/generator/initializer/sm.hpp"

namespace {

template <class... Ts, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

}  // namespace

TEST_CASE("generator_initializer_sm_models_explicit_initialize_pipeline_states") {
  using machine_t = boost::sml::sm<emel::generator::initializer::model>;
  using states = typename machine_t::states;

  CHECK(emel::detail::type_list_contains<
        emel::generator::initializer::preparing_backend_decision,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::generator::initializer::binding_conditioner,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::generator::initializer::binding_conditioner_decision,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::generator::initializer::reserving_graph_decision,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::generator::initializer::configuring_sampler_decision,
        states>::value);
}

TEST_CASE("generator_initializer_sm_uses_explicit_internal_run_completion") {
  using machine_t = boost::sml::sm<emel::generator::initializer::model>;
  using transitions = typename machine_t::transitions;

  bool has_run_completion = false;

  for_each_type(transitions{}, [&]<class transition_t>() {
    using event = typename transition_t::event;
    const std::string event_name = emel::docs::detail::table_event_name<event>();
    if (event_name == "completion<run>") {
      has_run_completion = true;
    }
  });

  CHECK(has_run_completion);
}

TEST_CASE("generator_initializer_sm_routes_backend_ready_run_completion_directly") {
  using machine_t = boost::sml::sm<emel::generator::initializer::model>;
  using transitions = typename machine_t::transitions;

  bool has_backend_ready_transition = false;

  for_each_type(transitions{}, [&]<class transition_t>() {
    using src_state = typename transition_t::src_state;
    using dst_state = typename transition_t::dst_state;
    using event = typename transition_t::event;

    const std::string src_name = emel::docs::detail::shorten_type_name(
        emel::docs::detail::raw_type_name<src_state>());
    const std::string dst_name = emel::docs::detail::shorten_type_name(
        emel::docs::detail::raw_type_name<dst_state>());
    const std::string event_name = emel::docs::detail::table_event_name<event>();

    if (src_name == "preparing_backend" &&
        dst_name == "binding_conditioner" &&
        event_name == "completion<run>") {
      has_backend_ready_transition = true;
    }
  });

  CHECK(has_backend_ready_transition);
}
