#pragma once

#include "bench_cases.hpp"

#include <cstddef>
#include <span>
#include <string_view>

namespace emel::bench {

std::span<const test_case> default_runner_cases() noexcept;
std::span<const test_case> kernel_runner_cases() noexcept;

std::size_t registered_runner_count() noexcept;
std::string_view registered_runner_suite_at(std::size_t index) noexcept;
const test_case * find_registered_runner(std::string_view suite) noexcept;

}  // namespace emel::bench
