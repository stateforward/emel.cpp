#include <doctest/doctest.h>

#include <cstdint>
#include <utility>

#include "emel/callback.hpp"

namespace {

volatile int64_t g_sink = 0;

struct recorder {
  int32_t total = 0;
};

bool add_owner_value(void * owner, const int32_t value) noexcept {
  auto * rec = static_cast<recorder *>(owner);
  rec->total += value;
  return true;
}

bool add_ref_value(recorder & rec, const int32_t value) noexcept {
  rec.total += value;
  return true;
}

bool add_thunk(void *, const int32_t value) noexcept {
  g_sink += value;
  return true;
}

}  // namespace

TEST_CASE("callback binds object reference to thunk") {
  recorder rec{};
  emel::callback<bool(const int32_t)> cb{rec, add_owner_value};
  CHECK(cb(4));
  CHECK(rec.total == 4);
}

TEST_CASE("callback thunk constructor remains valid after copy and move") {
  g_sink = 0;
  emel::callback<bool(const int32_t)> cb{add_thunk};
  int32_t value = 1;

  CHECK(cb(value));
  CHECK(g_sink == 1);

  const auto copied = cb;
  value = 2;
  CHECK(copied(value));
  CHECK(g_sink == 3);

  auto moved = std::move(cb);
  value = 3;
  CHECK(moved(value));
  CHECK(g_sink == 6);
}

TEST_CASE("callback from object reference and free function binds correctly") {
  recorder rec{};
  const auto cb = emel::callback<bool(const int32_t)>::from<recorder, &add_ref_value>(rec);
  CHECK(cb(7));
  CHECK(rec.total == 7);
}

TEST_CASE("callback bool operator reports empty callbacks") {
  emel::callback<bool(int32_t)> empty_bool{};
  CHECK_FALSE(static_cast<bool>(empty_bool));

  emel::callback<void(int32_t &)> empty_void{};
  CHECK_FALSE(static_cast<bool>(empty_void));
}
