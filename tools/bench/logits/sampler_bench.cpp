#include "bench_cases.hpp"

#include "logits/bench_common.hpp"

namespace emel::bench {

void append_emel_logits_sampler_cases(std::vector<result> & results, const config & cfg) {
  volatile std::int64_t sink = 0;

  for (const int32_t vocab_size : k_vocab_sizes) {
    logits_case_data sampler_data{vocab_size};
    int32_t selected_token_out = -1;
    emel::error::type sampler_error_out = emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::event::sample_logits sample_event{
      sampler_data.logits[0],
      vocab_size,
      sampler_data.candidate_ids[0],
      sampler_data.candidate_scores[0],
      vocab_size,
      selected_token_out,
      sampler_error_out};

    emel::logits::sampler::fn sampler_chain[] = {
      emel::logits::sampler::fn::from<top_k_sampler>(),
      emel::logits::sampler::fn::from<argmax_sampler>(),
    };
    constexpr int32_t sampler_count = static_cast<int32_t>(std::size(sampler_chain));

    emel::error::type sampler_config_error =
        emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::sm sampler_machine{};
    emel::logits::sampler::event::configure sampler_config{
      sampler_chain[0],
      sampler_count,
      sampler_config_error,
    };
    (void)sampler_machine.process_event(sampler_config);
    const std::string sampler_sml_case = make_case_name("sampler", "sml", vocab_size);
    auto sampler_sml_fn = [&]() {
      (void)sampler_machine.process_event(sample_event);
      sink ^= static_cast<std::int64_t>(selected_token_out);
    };
    results.push_back(emel::bench::measure_case(sampler_sml_case.c_str(), cfg, sampler_sml_fn));

    const std::string sampler_raw_case = make_case_name("sampler", "raw", vocab_size);
    auto sampler_raw_fn = [&]() {
      (void)run_sampler_raw(sample_event, sampler_chain, sampler_count);
      sink ^= static_cast<std::int64_t>(selected_token_out);
    };
    results.push_back(emel::bench::measure_case(sampler_raw_case.c_str(), cfg, sampler_raw_fn));
  }

  (void)sink;
}

void append_reference_logits_sampler_cases(std::vector<result> & results, const config & cfg) {
  append_emel_logits_sampler_cases(results, cfg);
}

}  // namespace emel::bench
