#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

status=0

check_no_matches() {
  local label="$1"
  local pattern="$2"
  shift 2

  if rg -n "$pattern" "$@" >/tmp/emel_domain_boundary_matches.$$ 2>/dev/null; then
    echo "error: domain boundary leak: ${label}" >&2
    cat /tmp/emel_domain_boundary_matches.$$ >&2
    status=1
  fi
  rm -f /tmp/emel_domain_boundary_matches.$$
}

check_no_matches_except() {
  local label="$1"
  local pattern="$2"
  local allow_pattern="$3"
  shift 3

  if rg -n "$pattern" "$@" >/tmp/emel_domain_boundary_matches.$$ 2>/dev/null; then
    if grep -Ev "$allow_pattern" /tmp/emel_domain_boundary_matches.$$ \
        >/tmp/emel_domain_boundary_filtered.$$; then
      echo "error: domain boundary leak: ${label}" >&2
      cat /tmp/emel_domain_boundary_filtered.$$ >&2
      status=1
    fi
    rm -f /tmp/emel_domain_boundary_filtered.$$
  fi
  rm -f /tmp/emel_domain_boundary_matches.$$
}

check_absent_path() {
  local label="$1"
  shift

  local matched=0
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      echo "error: domain boundary leak: ${label}: ${path}" >&2
      matched=1
    fi
  done

  if [[ "$matched" -ne 0 ]]; then
    status=1
  fi
}

cd "$ROOT_DIR"

concrete_io_api_pattern='(^|[^[:alnum:]_])(::)?(mmap|munmap|pread|read|fread|fopen|fclose|open|openat|CreateFile|CreateFileMapping|MapViewOfFile|ReadFile)[[:space:]]*\(|std::(ifstream|fstream|filebuf|fread|fopen|freopen)'

check_no_matches "forbidden model-family runtime roots" \
  'emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper|model/whisper/(runtime|inference|encoder|decoder)|model::whisper::(runtime|inference|encoder|decoder)|speech/asr/whisper|speech::asr::whisper|speech/whisper|speech::whisper|transcriber/detail/whisper|transcriber::detail::whisper' \
  src tests tools CMakeLists.txt

check_no_matches "qwen3 model code depends on sibling model-family detail" \
  'emel/model/llama/(detail|any)\.hpp|emel/model/(gemma4|lfm2)/detail\.hpp|emel::model::llama::|emel::model::(gemma4|lfm2)::detail::' \
  src/emel/model/qwen3

check_no_matches "gemma4 model code depends on sibling model-family detail" \
  'emel/model/llama/(detail|any)\.hpp|emel/model/(qwen3|lfm2)/detail\.hpp|emel::model::llama::|emel::model::(qwen3|lfm2)::detail::' \
  src/emel/model/gemma4

check_no_matches "lfm2 model code depends on sibling model-family detail" \
  'emel/model/llama/(detail|any)\.hpp|emel/model/(qwen3|gemma4)/detail\.hpp|emel::model::llama::|emel::model::(qwen3|gemma4)::detail::' \
  src/emel/model/lfm2

check_no_matches_except "legacy text generator root" \
  'emel/generator|emel::generator|namespace emel::generator|src/emel/generator|tests/generator' \
  '^(src/emel/generator/|tests/text/generator/legacy_compatibility_tests\.cpp:[0-9]+:|scripts/(quality_gates|test_with_coverage)\.sh:[0-9]+:.*src/emel/generator)' \
  src tests tools scripts/quality_gates.sh scripts/test_with_coverage.sh CMakeLists.txt

check_no_matches "text generator actor internals in maintained generation parity/benchmark lanes" \
  'emel/text/generator/(detail|actions|guards)\.hpp|emel::text::generator::(detail|action|guard)::|emel::text::generator::prefill::guard::|generation_internal_diagnostics' \
  tools/generation_route_policy.hpp tools/bench/generation_bench.cpp tools/paritychecker/parity_engines.cpp tools/paritychecker/parity_runner.cpp tools/paritychecker/parity_runner.hpp

check_no_matches "model-family detail contracts leaked into generic text generator headers" \
  'emel/model/(generation|llama|qwen3|gemma4|lfm2)/detail|model::(generation|llama|qwen3|gemma4|lfm2)::detail' \
  src/emel/text/generator

check_no_matches "model-family names leaked into neutral generation contract" \
  'llama|qwen3|gemma4|lfm2|whisper|moshi' \
  src/emel/model/generation

check_no_matches "OmniEmbed model-family detail leaked into generic embeddings generator surfaces" \
  'omniembed|OmniEmbed|model/omniembed|model::omniembed' \
  src/emel/embeddings/generator/*.hpp \
  src/emel/embeddings/generator/*.cpp

check_no_matches "model-family route names leaked into generic embeddings generator orchestration surfaces" \
  'bert|Bert|BERT|mobilenet|MobileNet|MOBILENET|efficientat|EfficientAT|EFFICIENTAT|edge_residual|universal_inverted|audio_inverted|has_dw|has_expand|use_hardswish|k_max_blocks|matrix_view|conv2d_view|batch_norm' \
  src/emel/embeddings/generator/context.hpp \
  src/emel/embeddings/generator/detail.hpp \
  src/emel/embeddings/generator/actions.hpp \
  src/emel/embeddings/generator/guards.hpp \
  src/emel/embeddings/generator/sm.hpp \
  src/emel/embeddings/generator/*.cpp

check_no_matches "IO loader concrete system I/O before strategy implementation" \
  "$concrete_io_api_pattern" \
  src/emel/io/loader

check_no_matches "model loader low-level IO strategy implementation" \
  "$concrete_io_api_pattern" \
  src/emel/model/loader

check_no_matches "model tensor concrete IO strategy implementation" \
  "$concrete_io_api_pattern" \
  src/emel/model/tensor

check_no_matches "shadow model tensor residency ownership outside model/tensor" \
  'model::tensor::event::lifecycle::|lifecycle_state|event::tensor_state' \
  src/emel/model/loader src/emel/io

# v1.24 mmap component scope: only the io/mmap strategy can land here. Staged
# read/external buffer/async/device/copy strategy markers must not appear in
# the mmap actor; those belong in io/loader (strategy router) or future v2
# milestones. The doctest source-string check inside the test is informative,
# but VAL-02 demands the gate fail closed at the script level.
check_no_matches "out-of-scope strategy markers leaked into io/mmap actor" \
  'strategy_staged_read|strategy_external_buffer|strategy_async|strategy_device|strategy_copy' \
  src/emel/io/mmap

# v2 strategy implementations are deferred. Until those milestones land, no
# src/ code should declare async, device, or copy strategy guards/states.
# (Staged-read and external-buffer routing legitimately exists today only in
# src/emel/io/loader, so they are excluded here.)
check_no_matches "deferred v2 strategy implementations leaked into src/" \
  'strategy_async|strategy_device|strategy_copy' \
  src

# VAL-02: tensor residency lifecycle ownership stays in model/tensor.
# Loader, mmap, and io must never write or branch on the lifecycle::*
# residency enumerators (mmap_resident, resident, evicted, none). Tools and
# tests legitimately read residency through the public capture_tensor_state
# event and may inspect lifecycle values, so they are not scanned here.
check_no_matches "tensor residency lifecycle enumerators escaped model/tensor" \
  'lifecycle::mmap_resident|lifecycle::resident|lifecycle::evicted' \
  src/emel/model/loader src/emel/io

check_no_matches "maintained benchmark/parity lanes reaching IO or tensor actor internals" \
  'emel/(io/loader|model/tensor|model/loader)/(actions|detail|guards)\.hpp|emel::io::loader::(action|detail|guard)::|emel::model::tensor::(action|detail|guard)::|emel::model::loader::(action|detail|guard)::' \
  tools/bench tools/paritychecker tools/embedded_size

sortformer_bench_actor_pattern=
sortformer_bench_actor_pattern+='emel/diarization/'
sortformer_bench_actor_pattern+='(request|sortformer/'
sortformer_bench_actor_pattern+='(request|encoder(/feature_extractor)?|executor|modules|output|pipeline))'
sortformer_bench_actor_pattern+='/(actions|detail|guards)\.hpp'
sortformer_bench_actor_pattern+='|emel/model/sortformer/detail\.hpp'
sortformer_bench_actor_pattern+='|emel::diarization::sortformer::request::'
sortformer_bench_actor_pattern+='(action|detail|guard)::'
sortformer_bench_actor_pattern+='|emel::diarization::sortformer::'
sortformer_bench_actor_pattern+='(encoder::(action|guard|detail)'
sortformer_bench_actor_pattern+='|encoder::feature_extractor::detail'
sortformer_bench_actor_pattern+='|executor::(action|guard|detail)'
sortformer_bench_actor_pattern+='|modules::detail'
sortformer_bench_actor_pattern+='|output::(action|guard|detail)'
sortformer_bench_actor_pattern+='|pipeline::(action|guard|detail))::'
sortformer_bench_actor_pattern+='|emel::model::sortformer::detail::'
check_no_matches "Sortformer diarization benchmark bypassing actor surfaces" \
  "$sortformer_bench_actor_pattern" \
  tools/bench/diarization/sortformer_bench.cpp \
  tools/bench/diarization/sortformer_fixture.hpp

check_no_matches "Sortformer public facade headers exposing detail internals" \
  'emel/.*/detail\.hpp|::detail::' \
  src/emel/model/sortformer/any.hpp \
  src/emel/diarization/sortformer/request/events.hpp \
  src/emel/diarization/sortformer/executor/events.hpp \
  src/emel/diarization/sortformer/output/any.hpp \
  src/emel/diarization/sortformer/pipeline/any.hpp

check_absent_path "generic diarization request route hardcoding Sortformer" \
  src/emel/diarization/request \
  tests/diarization/request

check_no_matches "forbidden moshi model-family runtime roots" \
  'emel/moshi|namespace emel::moshi|kernel/moshi|kernel::moshi|kernel/mimi|kernel::mimi|model/moshi/(runtime|inference|encoder|decoder|codec)|model::moshi::(runtime|inference|encoder|decoder|codec)|speech/asr/moshi|speech::asr::moshi|speech/moshi/|speech::moshi' \
  src tests tools CMakeLists.txt

check_no_matches "Moshi/Mimi leaked into generic speech transcriber" \
  'moshi|model/moshi|model::moshi|tokenizer::moshi|codec::mimi|speech/codec/mimi' \
  src/emel/speech/transcriber tests/speech/transcriber

check_absent_path "retired generic diarization request owner path" \
  src/emel/diarization/request \
  tests/diarization/request \
  docs/architecture/diarization_request.md \
  docs/architecture/mermaid/diarization_request.mmd \
  .planning/architecture/diarization_request.md \
  .planning/architecture/mermaid/diarization_request.mmd

check_no_matches "Whisper leaked into generic speech transcriber" \
  'whisper|model/whisper|speech/tokenizer/whisper|speech/encoder/whisper|speech/decoder/whisper|model::whisper|tokenizer::whisper|encoder::whisper|decoder::whisper' \
  src/emel/speech/transcriber

# Tests may select component variants through the component-level kind enums
# (encoder_kind::whisper), but must not include variant headers or reach into
# variant namespaces.
check_no_matches "Whisper contracts leaked into speech transcriber tests" \
  'emel/speech/(encoder|decoder|tokenizer)/whisper|emel/model/whisper|(model|tokenizer|encoder|decoder)::whisper::' \
  tests/speech/transcriber

check_no_matches "Variant detail leaked into speech component facades" \
  'emel/speech/(encoder|decoder|tokenizer)/whisper/detail\.hpp|(encoder|decoder|tokenizer)::whisper::detail' \
  src/emel/speech/encoder/any.hpp src/emel/speech/encoder/events.hpp \
  src/emel/speech/decoder/any.hpp src/emel/speech/decoder/events.hpp \
  src/emel/speech/tokenizer/any.hpp src/emel/speech/tokenizer/events.hpp

check_no_matches "Whisper model binding leaked into speech encoder/decoder runtime" \
  'emel/model/whisper|model::whisper' \
  src/emel/speech/encoder src/emel/speech/decoder

check_no_matches "Moshi model detail leaked into Mimi codec" \
  'emel/model/moshi|model::moshi' \
  src/emel/speech/codec/mimi

check_absent_path "retired model weight-loader owner path" \
  src/emel/model/weight_loader \
  tests/model/weight_loader \
  docs/architecture/model_weight_loader.md \
  docs/architecture/mermaid/model_weight_loader.mmd \
  .planning/architecture/model_weight_loader.md \
  .planning/architecture/mermaid/model_weight_loader.mmd

check_no_matches "retired model weight-loader owner references" \
  'model/weight_loader|model_weight_loader|namespace emel::model::weight_loader|WeightLoader|load_weights|bind_weights|src/emel/model/weight_loader|tests/model/weight_loader' \
  src tests tools docs README.md CMakeLists.txt snapshots/lint .planning/codebase .planning/architecture

check_no_matches "retired model weight-loader public docs prose" \
  'weight loader|weight-loader|weight-loading|loader callback parity|async[[:space:]]+upload|loader/parser/weight loader|loader/parser/weight-loader' \
  docs/roadmap.md

exit "$status"
