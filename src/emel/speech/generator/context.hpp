#pragma once

#include "emel/batch/planner/sm.hpp"
#include "emel/graph/sm.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/memory/hybrid/sm.hpp"

namespace emel::speech::generator {

struct dependencies {
  emel::batch::planner::sm &planner;
  emel::memory::hybrid::sm &memory;
  emel::graph::sm &graph;
  emel::logits::sampler::sm &sampler;
  emel::kernel::sm &kernel;
};

namespace action {

struct context {
  explicit context(const dependencies &deps) noexcept : collaborators(deps) {}

  const dependencies collaborators;
};

} // namespace action

} // namespace emel::speech::generator
