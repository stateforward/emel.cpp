#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "../generation_fixture_registry.hpp"

namespace emel::paritychecker::assets {

using maintained_generation_fixture =
    emel::tools::generation_fixture_registry::maintained_fixture;

std::filesystem::path repo_root_path();

std::filesystem::path generation_baseline_directory_path();

bool file_exists(const std::string &path);

std::filesystem::path normalize_path(const std::filesystem::path &path);

std::filesystem::path
expected_generation_fixture_path(const maintained_generation_fixture &fixture);

const maintained_generation_fixture *
find_generation_fixture(const std::string &model_path);

std::string maintained_generation_fixture_list();

} // namespace emel::paritychecker::assets
