#pragma once

namespace emel::telemetry::provider::action {
struct context;
}  // namespace emel::telemetry::provider::action

namespace emel::telemetry::provider::guard {

bool is_configured(const action::context & ctx);

}  // namespace emel::telemetry::provider::guard

