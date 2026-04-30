#include <doctest/doctest.h>
#include <type_traits>

#include "emel/generator/actions.hpp"
#include "emel/generator/context.hpp"
#include "emel/generator/detail.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/guards.hpp"
#include "emel/generator/initializer/actions.hpp"
#include "emel/generator/initializer/context.hpp"
#include "emel/generator/initializer/detail.hpp"
#include "emel/generator/initializer/guards.hpp"
#include "emel/generator/initializer/sm.hpp"
#include "emel/generator/prefill/actions.hpp"
#include "emel/generator/prefill/context.hpp"
#include "emel/generator/prefill/detail.hpp"
#include "emel/generator/prefill/guards.hpp"
#include "emel/generator/prefill/sm.hpp"
#include "emel/generator/sm.hpp"
#include "emel/text/generator/sm.hpp"

namespace {

static_assert(std::is_same_v<emel::generator::sm, emel::text::generator::sm>);
static_assert(std::is_same_v<emel::generator::event::initialize,
                             emel::text::generator::event::initialize>);
static_assert(std::is_same_v<emel::generator::event::generate,
                             emel::text::generator::event::generate>);
static_assert(std::is_same_v<emel::generator::events::initialize_done,
                             emel::text::generator::events::initialize_done>);
static_assert(std::is_same_v<emel::generator::initializer::sm,
                             emel::text::generator::initializer::sm>);
static_assert(std::is_same_v<emel::generator::prefill::sm,
                             emel::text::generator::prefill::sm>);

}  // namespace

TEST_CASE("legacy generator headers forward to text generator surface") {
  CHECK((std::is_same_v<emel::generator::sm, emel::text::generator::sm>));
}
