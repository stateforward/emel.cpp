#include "bench_cases.hpp"

#include "logits/bench_common.hpp"

namespace emel::bench {

void append_emel_logits_validator_cases(std::vector<result> & results, const config & cfg) {
  volatile std::int64_t sink = 0;

  for (const int32_t vocab_size : k_vocab_sizes) {
    logits_case_data validator_data{vocab_size};
    int32_t candidate_count_out = 0;
    emel::error::type validator_error_out = emel::error::cast(emel::logits::validator::error::none);
    emel::logits::validator::event::build build_event{
      validator_data.logits[0],
      vocab_size,
      validator_data.candidate_ids[0],
      validator_data.candidate_scores[0],
      vocab_size,
      candidate_count_out,
      validator_error_out};

    emel::logits::validator::sm validator_machine{};
    const std::string validator_sml_case = make_case_name("validator", "sml", vocab_size);
    auto validator_sml_fn = [&]() {
      (void)validator_machine.process_event(build_event);
      sink ^= static_cast<std::int64_t>(candidate_count_out);
    };
    results.push_back(emel::bench::measure_case(validator_sml_case.c_str(), cfg, validator_sml_fn));

    const std::string validator_raw_case = make_case_name("validator", "raw", vocab_size);
    auto validator_raw_fn = [&]() {
      (void)run_validator_raw(build_event);
      sink ^= static_cast<std::int64_t>(candidate_count_out);
    };
    results.push_back(emel::bench::measure_case(validator_raw_case.c_str(), cfg, validator_raw_fn));
  }

  (void)sink;
}

void append_reference_logits_validator_cases(std::vector<result> & results, const config & cfg) {
  append_emel_logits_validator_cases(results, cfg);
}

}  // namespace emel::bench
