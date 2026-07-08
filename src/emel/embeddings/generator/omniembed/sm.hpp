#pragma once
// benchmark: designed

#include "emel/embeddings/generator/omniembed/route.hpp"
#include "emel/embeddings/generator/sm.hpp"

namespace emel::embeddings::generator::omniembed {

struct model : emel::embeddings::generator::model<route> {};

using sm = emel::embeddings::generator::basic_sm<route>;

}  // namespace emel::embeddings::generator::omniembed
