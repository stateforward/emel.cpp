#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/docs/detail.hpp"
#include "emel/generator/prefill/sm.hpp"

namespace {

template <class... Ts, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

}  // namespace

TEST_CASE("generator_prefill_sm_models_explicit_contract_and_compute_states") {
  using machine_t = boost::sml::sm<emel::generator::prefill::model>;
  using states = typename machine_t::states;

  CHECK(emel::detail::type_list_contains<
        emel::generator::prefill::contract_flash_decision,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::generator::prefill::contract_nonflash_decision,
        states>::value);
  CHECK(emel::detail::type_list_contains<
        emel::generator::prefill::compute_result_decision,
        states>::value);
}

TEST_CASE("generator_prefill_sm_uses_explicit_internal_run_completion") {
  using machine_t = boost::sml::sm<emel::generator::prefill::model>;
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
