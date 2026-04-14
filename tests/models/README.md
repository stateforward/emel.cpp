# Test Models

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
- Official sibling files: `TE-75M-q5_0.gguf`.
  Its presence is source truth for the upstream release, not proof of maintained support in this
  repo.

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
