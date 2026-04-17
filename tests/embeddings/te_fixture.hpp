#pragma once

#include "doctest/doctest.h"

#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"

#include "te_fixture_data.hpp"

namespace emel::tests::embeddings::te_fixture {

struct inspectable_embedding_generator : emel::embeddings::generator::sm {
  using emel::embeddings::generator::sm::sm;

  emel::embeddings::generator::action::context & context_ref() noexcept {
    return this->context_;
  }
};

inline void initialize_embedding_generator(emel::embeddings::generator::sm & embedding_generator,
                                           emel::error::type & initialize_error,
                                           emel::text::tokenizer::sm & tokenizer) {
  emel::embeddings::generator::event::initialize initialize{
    &tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
  };
  initialize.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
  initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
  initialize.error_out = &initialize_error;

  REQUIRE(embedding_generator.process_event(initialize));
  CHECK(initialize_error == emel::error::cast(emel::embeddings::generator::error::none));
}

}  // namespace emel::tests::embeddings::te_fixture
