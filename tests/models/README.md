# Test Models

## model-tiny-q80.gguf
- Source: `https://huggingface.co/oxide-lab/whisper-tiny-GGUF`
- File: `model-tiny-q80.gguf`
- Stable maintained path: `tests/models/model-tiny-q80.gguf`
- License: MIT
- Upstream model: `https://huggingface.co/openai/whisper-tiny`
- Repository commit: `94468a6c81edab8c594d9b1d06ea1dfb64292327`
- Size: `40700160` bytes (`39M`)
- SHA256: `52deb0fdcbb9c36b4d570e35f5a65a5ad4275ccdb85e7a06e81a8b05b3743c9d`
- Xet hash: `9b02c103aabca43b343c667068c2b81fa8d5090597f42adbfdc7e68b2b4d3aa9`
- Download URL:
  `https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/94468a6c81edab8c594d9b1d06ea1dfb64292327/model-tiny-q80.gguf`
- Executable metadata truth: this root artifact is a real GGUF file with `GGUF` magic,
  `general.architecture=whisper`, `whisper.n_mels=80`, `whisper.n_vocab=51865`, `168`
  tensors, and Candle-style tensor names with the `model.` prefix.
- Maintained contract: Whisper tiny encoder-decoder shape only: mono `16000` Hz speech
  frontend, `80` mel bins, encoder context `1500`, decoder context `448`, embedding width
  `384`, feed-forward width `1536`, `6` attention heads, `4` encoder blocks, and `4`
  decoder blocks.
- Variant-family scope note: the maintained v1.16 Whisper tiny GGUF variant family is
  narrowed to the three upstream EMEL-loadable Candle-style GGUFs at the top level of the
  pinned repo commit: `q8_0` (`model-tiny-q80.gguf`), `q4_0` (`whisper-tiny-q4_0.gguf`),
  and `q4_1` (`whisper-tiny-q4_1.gguf`). The broader quant family (`q5_0`, `q5_1`, `q2_k`,
  `q3_k`, `q4_k`, `q5_k`, `q6_k`, `q8_k`) is deferred to a future approved EMEL-owned
  conversion phase. This section does not claim every variant is already runnable.
- Loader/runtime boundary note: the evidence in this section proves fixture provenance and
  loader/model-contract validation only. It is not proof that EMEL Whisper ASR runtime or
  `whisper.cpp` parity is complete.
- Reference-lane note: the sibling `whisper.cpp/` artifacts in the same Hugging Face repo are
  reserved for the whisper.cpp reference lane. Direct inspection of
  `whisper.cpp/whisper-tiny-q4_k.gguf` at the pinned commit found `lmgg` magic rather than
  `GGUF`, so those files must not feed EMEL-owned GGUF runtime objects.
- External tokenizer-asset note: the Candle-style Whisper tiny GGUFs in this repo do not
  embed tokenizer metadata (`tokenizer.model`, `tokenizer.ggml.model`, `tokenizer.tokens`,
  etc.). The maintained Whisper tokenizer source is the upstream `tokenizer-tiny.json`
  sibling at
  `https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/94468a6c81edab8c594d9b1d06ea1dfb64292327/tokenizer-tiny.json`
  (size `2480452` bytes, SHA256
  `dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759`). The reopened v1.16
  speech tokenizer work stages this external asset under `tests/models/tokenizer-tiny.json`;
  runtime wiring must use that tokenizer contract instead of silently falling back to absent GGUF
  tokenizer keys or hardcoded transcript token strings.

## whisper-tiny-q4_0.gguf
- Source: `https://huggingface.co/oxide-lab/whisper-tiny-GGUF`
- File: `whisper-tiny-q4_0.gguf`
- Stable maintained path: `tests/models/whisper-tiny-q4_0.gguf`
- License: MIT
- Upstream model: `https://huggingface.co/openai/whisper-tiny`
- Repository commit: `94468a6c81edab8c594d9b1d06ea1dfb64292327`
- Size: `22087104` bytes (`22M`)
- SHA256: `b2be6457e86d2c917d0c0eecef8e041ed03c60f64fc5744e6720adfb5141c21b`
- Download URL:
  `https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/94468a6c81edab8c594d9b1d06ea1dfb64292327/whisper-tiny-q4_0.gguf`
- Executable metadata truth: top-level Candle-style GGUF sibling of `model-tiny-q80.gguf`
  with `GGUF` magic, `general.architecture=whisper`, `whisper.n_mels=80`,
  `whisper.n_vocab=51865`, and the same Whisper tiny encoder-decoder tensor manifest under
  the `model.` prefix as the q80 root fixture; differs only in quantization scheme.
- Maintained contract: same Whisper tiny encoder-decoder shape as the q80 root (mono
  `16000` Hz, `80` mel bins, encoder context `1500`, decoder context `448`, embedding
  width `384`, feed-forward width `1536`, `6` attention heads, `4` encoder blocks, `4`
  decoder blocks).
- Loader/runtime boundary note: the evidence in this section proves fixture provenance and
  loader/model-contract validation only for the `q4_0` variant. It is not proof that EMEL
  Whisper ASR runtime or `whisper.cpp` parity is complete.

## whisper-tiny-q4_1.gguf
- Source: `https://huggingface.co/oxide-lab/whisper-tiny-GGUF`
- File: `whisper-tiny-q4_1.gguf`
- Stable maintained path: `tests/models/whisper-tiny-q4_1.gguf`
- License: MIT
- Upstream model: `https://huggingface.co/openai/whisper-tiny`
- Repository commit: `94468a6c81edab8c594d9b1d06ea1dfb64292327`
- Size: `24414464` bytes (`24M`)
- SHA256: `7d40a062a67abeb53784edd326610035089164c9c261cbcfa628e017a07e7a3a`
- Download URL:
  `https://huggingface.co/oxide-lab/whisper-tiny-GGUF/resolve/94468a6c81edab8c594d9b1d06ea1dfb64292327/whisper-tiny-q4_1.gguf`
- Executable metadata truth: top-level Candle-style GGUF sibling of `model-tiny-q80.gguf`
  with `GGUF` magic, `general.architecture=whisper`, `whisper.n_mels=80`,
  `whisper.n_vocab=51865`, and the same Whisper tiny encoder-decoder tensor manifest under
  the `model.` prefix as the q80 root fixture; differs only in quantization scheme.
- Maintained contract: same Whisper tiny encoder-decoder shape as the q80 root (mono
  `16000` Hz, `80` mel bins, encoder context `1500`, decoder context `448`, embedding
  width `384`, feed-forward width `1536`, `6` attention heads, `4` encoder blocks, `4`
  decoder blocks).
- Loader/runtime boundary note: the evidence in this section proves fixture provenance and
  loader/model-contract validation only for the `q4_1` variant. It is not proof that EMEL
  Whisper ASR runtime or `whisper.cpp` parity is complete.

## diar_streaming_sortformer_4spk-v2.1.gguf
- Source: `https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`
- File: `diar_streaming_sortformer_4spk-v2.1.gguf`
- Stable maintained path: `tests/models/diar_streaming_sortformer_4spk-v2.1.gguf`
- License: NVIDIA Open Model License
- Upstream model: `https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1`
- Repository commit: `e2970f92934105c6b385a047dc098aaaa593621b`
- Size: `471107712` bytes (`449M`)
- Linked ETag: `1b85d7bf641350d0d355e7494c4b7d92a1ff2fb2d886cd6dcc43f358a6266ff0`
- Xet hash: `c2967a4fb433032ad610e17aee4a2137a770c489fcc0d42397b11ad527aa1628`
- Download URL: `https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1.gguf`
- Original NeMo source: `https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1`
  at repo commit `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`, file
  `diar_streaming_sortformer_4spk-v2.1.nemo`, size `471367680` bytes, linked ETag
  `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`.
- Optional ONNX reference source for diarization parity: repository
  `https://huggingface.co/ooobo/diar_streaming_sortformer_4spk-v2.1-onnx`, file
  `diar_streaming_sortformer_4spk-v2.1.onnx`, expected local path
  `build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx`, size `495690533` bytes, SHA256
  `5df5e883c8dae4e0ecba77739f3db38997c2ae57153de2583d625afb6abb2be0`.
- Executable metadata truth: treat current Hugging Face GGUF/model metadata as authoritative for
  this maintained slice, including `gguf.architecture=sortformer`, source format `nemo`, compact
  tensor-name scheme `compact_v1`, and maintained `f32` outtype.
- Maintained stream contract: mono `float32` PCM at `16000` Hz, four speakers, `80` ms output
  frames, `chunk_len=188`, `chunk_right_context=1`, `fifo_len=0`,
  `spkcache_update_period=188`, and `spkcache_len=188`.
- Conversion source:
  `https://github.com/openresearchtools/engine/blob/main/build/sortformer/convert_nemo_sortformer_to_gguf.py`
- Conversion verification: self-converting the upstream `.nemo` with the conversion source above
  produced a `471107712` byte GGUF with `1007` tensors and `155` metadata keys. Normalized compare
  against the maintained OpenResearchTools GGUF found identical tensor manifests and identical
  payload hashes for all `1007` tensors; the only metadata value difference was display name
  (`general.name`).
- Provenance note: this is an unofficial community GGUF conversion derived from NVIDIA's
  `diar_streaming_sortformer_4spk-v2.1`; EMEL support is limited to the explicit maintained
  contract above until later phases add native execution proof.

## TE-75M-q8_0.gguf
- Source: `https://huggingface.co/augmem/TE-75M-GGUF`
- File: `TE-75M-q8_0.gguf`
- Stable maintained path: `tests/models/TE-75M-q8_0.gguf`
- License: Apache-2.0
- Size: `119710336` bytes (`114M`)
- SHA256: `955b5c847cc95c94ff14a27667d9aca039983448fd8cefe4f2804d3bfae621ae`
- Download URL: `https://huggingface.co/augmem/TE-75M-GGUF/resolve/main/TE-75M-q8_0.gguf`
- Executable metadata truth: treat current Hugging Face GGUF/model metadata as authoritative for
  this maintained slice, including `gguf.architecture=omniembed`, shared embedding width `1280`,
  supported Matryoshka truncation `1280/768/512/256/128`, and the current upstream component
  families `LEAF-IR`, `MobileNetV4-Medium`, and `EfficientAT mn20_as`.
- Maintained tokenizer asset: `tests/models/mdbr-leaf-ir-vocab.txt`
  Source: `https://huggingface.co/MongoDB/mdbr-leaf-ir`
  Download URL: `https://huggingface.co/MongoDB/mdbr-leaf-ir/resolve/main/vocab.txt`
  Size: `231508` bytes
  SHA256: `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
  Runtime use: supplies the MiniLM/BERT WordPiece vocabulary for the maintained TE text-lane
  proof because the GGUF slice does not embed tokenizer vocab metadata.

## TE-75M-q5_0.gguf
- Source: `https://huggingface.co/augmem/TE-75M-GGUF`
- File: `TE-75M-q5_0.gguf`
- Stable maintained path: `tests/models/TE-75M-q5_0.gguf`
- License: Apache-2.0
- Size: `111149248` bytes (`106M`)
- SHA256: `c63eb0db4fe4364e05d732063b45adb1dddfa206ba53886ed6d9b1b6fe1f9b73`
- Download URL: `https://huggingface.co/augmem/TE-75M-GGUF/resolve/main/TE-75M-q5_0.gguf`
- Executable metadata truth: treat current Hugging Face GGUF/model metadata as authoritative for
  this approved maintained sibling slice, including `gguf.architecture=omniembed`, shared
  embedding width `1280`, supported Matryoshka truncation `1280/768/512/256/128`, and the current
  upstream component families `LEAF-IR`, `MobileNetV4-Medium`, and `EfficientAT mn20_as`.
- Maintained proof note: this repo keeps `q5_0` on a separate maintained proof and benchmark lane
  from the default `q8_0` slice; promoting `q5_0` does not imply support for arbitrary TE quant
  siblings.
- Approved maintained TE fixtures in `v1.11`: `TE-75M-q8_0.gguf`, `TE-75M-q5_0.gguf`.
- Other upstream TE sibling artifacts remain unapproved unless they gain explicit maintained
  proof, benchmark, and requirements coverage in a later phase.

## gemma-4-e2b-it-Q8_0.gguf
- Source: `https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF`
- File: `gemma-4-e2b-it-Q8_0.gguf`
- Stable maintained path: `tests/models/gemma-4-e2b-it-Q8_0.gguf`
- License: Apache-2.0
- Size: `4967490208` bytes (`4.6G`)
- SHA256: `12d878964d21f1779dea15abeee048855151b27089fe98b32c628f85740933f3`
- Download URL: `https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q8_0.gguf`
- Executable metadata truth: treat official GGUF/config metadata as authoritative for this
  maintained slice, including `general.architecture=gemma4`, context length `131072`, and the
  upstream text-layer schedule.
- Official sibling files: `gemma-4-e2b-it-f16.gguf` and `mmproj-gemma-4-e2b-it-f16.gguf`.
  Their presence is source truth for the upstream release, not proof of maintained multimodal
  support in this repo.

## Qwen3-0.6B-Q8_0.gguf
- Source: `https://huggingface.co/Qwen/Qwen3-0.6B-GGUF`
- File: `Qwen3-0.6B-Q8_0.gguf`
- Stable maintained path: `tests/models/Qwen3-0.6B-Q8_0.gguf`
- License: Apache-2.0
- Size: `639446688` bytes (`610M`)
- SHA256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Download URL: `https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf`

## LFM2.5-1.2B-Thinking-Q4_K_M.gguf
- Source: `https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF`
- File: `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`
- Stable maintained path: `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf`
- License: Apache-2.0
- Size: `730895360` bytes (`697M`)
- SHA256: `7223a2202405b02e8e1e6c5baa543c43dc98c1d9741a5c2a0ee1583212e1231b`
- Download URL: `https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_K_M.gguf`
- Executable metadata truth: treat GGUF/config metadata as authoritative for this maintained slice,
  including `general.architecture=lfm2` and context length `128000`, even though some prose still
  mentions `32,768`.

## Llama-68M-Chat-v1-Q2_K.gguf
- Source: `https://huggingface.co/tensorblock/Llama-68M-Chat-v1-GGUF`
- File: `Llama-68M-Chat-v1-Q2_K.gguf`
- License: Apache-2.0
- Size: 34 MB
- SHA256: `8ed06dc5bd84bce3154a2b7e751c45a56562691933ee25b5823393f909329a67`
- Download URL: `https://huggingface.co/tensorblock/Llama-68M-Chat-v1-GGUF/resolve/main/Llama-68M-Chat-v1-Q2_K.gguf`

## distilgpt2.Q2_K.gguf
- Source: `https://huggingface.co/RichardErkhov/distilbert_-_distilgpt2-gguf`
- File: `distilgpt2.Q2_K.gguf`
- License: Apache-2.0
- Size: 63.6 MB
- SHA256: `b046ac09ba24a848e2140676fba58c1dcf2f19617e45b03524043eabdb556a31`
- Download URL: `https://huggingface.co/RichardErkhov/distilbert_-_distilgpt2-gguf/resolve/main/distilgpt2.Q2_K.gguf`

## bert-base-uncased-q4_k_m.gguf
- Source: `https://huggingface.co/Talek02/bert-base-uncased-Q4_K_M-GGUF`
- File: `bert-base-uncased-q4_k_m.gguf`
- License: Apache-2.0
- Size: 74.3 MB
- SHA256: `48c02c00843964c2e1675e6d6aebfbdb03d4ca330d65a6b9695eee6f160109b0`
- Download URL: `https://huggingface.co/Talek02/bert-base-uncased-Q4_K_M-GGUF/resolve/main/bert-base-uncased-q4_k_m.gguf`

## flan-t5-small.Q2_K.gguf
- Source: `https://huggingface.co/Felladrin/gguf-flan-t5-small`
- File: `flan-t5-small.Q2_K.gguf`
- License: Apache-2.0
- Size: 83.8 MB
- SHA256: `a67f632d17d2bdb819071c9b4d51e26a191f75c7f725861ee22e23e1d903dc57`
- Download URL: `https://huggingface.co/Felladrin/gguf-flan-t5-small/resolve/main/flan-t5-small.Q2_K.gguf`

## rwkv7-0.1B-g1-F16.gguf
- Source: `https://huggingface.co/zhiyuan8/RWKV-v7-0.1B-G1-GGUF`
- File: `rwkv7-0.1B-g1-F16.gguf`
- License: Apache-2.0
- Size: 386 MB
- SHA256: `fea5c54f3fd2370ac90ae58f2ecd6cbe57c31df023598aed4c95b0966170f9c8`
- Download URL: `https://huggingface.co/zhiyuan8/RWKV-v7-0.1B-G1-GGUF/resolve/main/rwkv7-0.1B-g1-F16.gguf`

## Reference baseline fixtures

These assets back the maintained `tools/bench/embedding_reference_bench.cpp` lane for the approved
same-host ARM comparison matrix in Phase `59.1.1`.

## snowflake-arctic-embed-s-Q8_0.gguf
- Source: `https://huggingface.co/yixuan-chia/snowflake-arctic-embed-s-GGUF`
- File: `snowflake-arctic-embed-s-Q8_0.gguf`
- Stable maintained path: `tests/models/reference/snowflake-arctic-embed-s-Q8_0.gguf`
- Size: `36685120` bytes
- SHA256: `4db660cc9122d6153be9306fb8707bba6b77035cbeb2ab078d9a58c33dfea6be`
- Download URL: `https://huggingface.co/yixuan-chia/snowflake-arctic-embed-s-GGUF/resolve/main/snowflake-arctic-embed-s-Q8_0.gguf`
- Provenance note: this is a third-party GGUF mirror used for the `llama.cpp` reference lane because
  no official `ggml-org` Arctic S GGUF repo was found during Phase `59.1.1`.

## embeddinggemma-300M-Q8_0.gguf
- Source: `https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF`
- File: `embeddinggemma-300M-Q8_0.gguf`
- Stable maintained path: `tests/models/reference/embeddinggemma-300M-Q8_0.gguf`
- Size: `328576992` bytes
- SHA256: `f470220f84b6235197541352d22f10bf00098a8242c18eaacea9c8a4add557bc`
- Download URL: `https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf`

## LFM2-VL-450M-Q8_0.gguf
- Source: `https://huggingface.co/ggml-org/LFM2-VL-450M-GGUF`
- File: `LFM2-VL-450M-Q8_0.gguf`
- Stable maintained path: `tests/models/reference/LFM2-VL-450M-Q8_0.gguf`
- Size: `379215264` bytes
- SHA256: `e97704a0cf0a1d00ca604b4c672c82f4234318dba9d43a7f7a4c0d2df6747844`
- Download URL: `https://huggingface.co/ggml-org/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q8_0.gguf`

## mmproj-LFM2-VL-450M-Q8_0.gguf
- Source: `https://huggingface.co/ggml-org/LFM2-VL-450M-GGUF`
- File: `mmproj-LFM2-VL-450M-Q8_0.gguf`
- Stable maintained path: `tests/models/reference/mmproj-LFM2-VL-450M-Q8_0.gguf`
- Size: `103890016` bytes
- SHA256: `a6a813fae4ba3f90852a990fdd4786a017dba0e80e599a366853ba539ee2b5ef`
- Download URL: `https://huggingface.co/ggml-org/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-Q8_0.gguf`

## Llama-3.2-1B-Instruct-Q8_0.gguf
- Source: `https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF`
- File: `Llama-3.2-1B-Instruct-Q8_0.gguf`
- Stable maintained path: `tests/models/reference/Llama-3.2-1B-Instruct-Q8_0.gguf`
- Size: `1321083008` bytes
- SHA256: `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3`
- Download URL: `https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf`

## mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf
- Source: `https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF`
- File: `mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf`
- Stable maintained path: `tests/models/reference/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf`
- Size: `1371123616` bytes
- SHA256: `b34dde1835752949d6b960528269af93c92fec91c61ea0534fcc73f96c1ed8b2`
- Download URL: `https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf`
