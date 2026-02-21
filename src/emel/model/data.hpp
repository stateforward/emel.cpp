#pragma once

#include <array>
#include <cstdint>

namespace emel::model {

struct data {
  static constexpr int32_t k_max_tensors = 65536;
  static constexpr int32_t k_max_name_bytes = 4 * 1024 * 1024;
  static constexpr int32_t k_max_architecture_name = 64;
  static constexpr int32_t k_max_vocab_tokens = 320000;
  static constexpr int32_t k_max_vocab_bytes = 8 * 1024 * 1024;
  static constexpr int32_t k_max_tokenizer_model = 64;
  static constexpr int32_t k_max_tokenizer_pre = 64;
  static constexpr int32_t k_max_merges = 400000;
  static constexpr int32_t k_max_merge_bytes = 8 * 1024 * 1024;
  static constexpr int32_t k_max_precompiled_charsmap_bytes = 1024 * 1024;
  static constexpr int32_t k_max_split_files = 128;
  static constexpr int32_t k_max_metadata_blob_bytes = 4 * 1024 * 1024;
  static constexpr int32_t k_max_metadata_strings = 4096;
  static constexpr int32_t k_max_metadata_list = 256;
  static constexpr int32_t k_max_metadata_entities = 64;
  static constexpr int32_t k_max_metadata_arrays = 512;
  static constexpr int32_t k_max_rope_dimension_sections = 32;
  static constexpr int32_t k_max_xielu_values = 256;
  static constexpr int32_t k_max_clip_image_stats = 16;
  static constexpr int32_t k_max_clip_layer_indexes = 512;

  enum class tokenizer_model : uint8_t {
    NONE = 0,
    SPM = 1,
    BPE = 2,
    WPM = 3,
    UGM = 4,
    RWKV = 5,
    PLAMO2 = 6,
    UNKNOWN = 7,
  };

  enum class tokenizer_pre : uint16_t {
    DEFAULT = 0,
    LLAMA3,
    JAIS2,
    DBRX,
    SMAUG,
    DEEPSEEK_LLM,
    DEEPSEEK_CODER,
    DEEPSEEK3_LLM,
    YOUTU,
    FALCON,
    MPT,
    STARCODER,
    GPT2,
    JAIS,
    REFACT,
    COMMAND_R,
    QWEN2,
    QWEN35,
    STABLELM2,
    OLMO,
    PORO,
    CHATGLM4,
    VIKING,
    TEKKEN,
    SMOLLM,
    CODESHELL,
    BLOOM,
    GPT3_FINNISH,
    EXAONE,
    EXAONE4,
    EXAONE_MOE,
    CHAMELEON,
    MINERVA,
    MEGREZ,
    GPT4O,
    TINY_AYA,
    SUPERBPE,
    TRILLION,
    GRANITE_DOCLING,
    BAILINGMOE,
    SEED_CODER,
    HUNYUAN,
    HUNYUAN_DENSE,
    JOYAI_LLM,
    KIMI_K2,
    GROK_2,
    AFMOE,
    MINIMAX_M2,
    SOLAR_OPEN,
    UNKNOWN,
  };

  struct tensor_record {
    uint32_t name_offset = 0;
    uint32_t name_length = 0;
    int32_t type = 0;
    int32_t n_dims = 0;
    std::array<int64_t, 4> dims = {};
    uint64_t data_offset = 0;
    uint64_t file_offset = 0;
    uint64_t data_size = 0;
    const void * data = nullptr;
    uint16_t file_index = 0;
  };

  struct hparams {
    int32_t n_ctx = 0;
    int32_t n_embd = 0;
    int32_t n_embd_out = 0;
    int32_t n_ff = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_rot = 0;
    int32_t n_layer = 0;
    int32_t n_vocab = 0;
    int32_t n_features = 0;
    int32_t n_leading_dense_block = 0;
    int32_t n_expert_ff = 0;
    int32_t n_expert_shared_ff = 0;
    int32_t n_expert_chunk_ff = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_expert_shared = 0;
    int32_t n_expert_group = 0;
    int32_t n_expert_group_used = 0;
    float expert_weights_scale = 0.0f;
    bool expert_weights_norm = false;
    int32_t expert_gating_func = 0;
    float expert_group_scale = 0.0f;
    int32_t experts_per_group = 0;
    int32_t moe_every_n_layers = 0;
    int32_t nextn_predict_layers = 0;
    int32_t n_deepstack_layers = 0;
    int32_t pooling_type = 0;
    float logit_scale = 0.0f;
    int32_t decoder_start_token_id = -1;
    int32_t decoder_block_count = 0;
    float attn_logit_softcapping = 0.0f;
    float router_logit_softcapping = 0.0f;
    float final_logit_softcapping = 0.0f;
    bool swin_norm = false;
    int32_t rescale_every_n_layers = 0;
    int32_t time_mix_extra_dim = 0;
    int32_t time_decay_extra_dim = 0;
    float residual_scale = 0.0f;
    float embedding_scale = 0.0f;
    int32_t token_shift_count = 0;
    int32_t interleave_moe_layer_step = 0;
    int32_t full_attention_interval = 0;
    float activation_sparsity_scale = 0.0f;
    int32_t altup_active_idx = -1;
    int32_t altup_num_inputs = 0;
    int32_t embd_length_per_layer_input = 0;
    uint32_t swiglu_clamp_exp_count = 0;
    std::array<float, k_max_xielu_values> swiglu_clamp_exp = {};
    uint32_t swiglu_clamp_shexp_count = 0;
    std::array<float, k_max_xielu_values> swiglu_clamp_shexp = {};
    int32_t dense_2_feat_in = 0;
    int32_t dense_2_feat_out = 0;
    int32_t dense_3_feat_in = 0;
    int32_t dense_3_feat_out = 0;
    bool use_parallel_residual = false;
    float attention_max_alibi_bias = 0.0f;
    float attention_clamp_kqv = 0.0f;
    int32_t attention_key_length = 0;
    int32_t attention_value_length = 0;
    float attention_layer_norm_epsilon = 0.0f;
    float attention_layer_norm_rms_epsilon = 0.0f;
    float attention_group_norm_epsilon = 0.0f;
    int32_t attention_group_norm_groups = 0;
    bool attention_causal = false;
    int32_t attention_q_lora_rank = 0;
    int32_t attention_kv_lora_rank = 0;
    int32_t attention_decay_lora_rank = 0;
    int32_t attention_iclr_lora_rank = 0;
    int32_t attention_value_residual_mix_lora_rank = 0;
    int32_t attention_gate_lora_rank = 0;
    int32_t attention_relative_buckets_count = 0;
    int32_t attention_sliding_window = 0;
    int32_t attention_sliding_window_pattern = 0;
    uint32_t attention_sliding_window_pattern_count = 0;
    std::array<uint8_t, k_max_metadata_arrays> attention_sliding_window_pattern_flags = {};
    float attention_scale = 0.0f;
    float attention_output_scale = 0.0f;
    int32_t attention_temperature_length = 0;
    float attention_temperature_scale = 0.0f;
    int32_t attention_key_length_mla = 0;
    int32_t attention_value_length_mla = 0;
    int32_t attention_indexer_head_count = 0;
    int32_t attention_indexer_key_length = 0;
    int32_t attention_indexer_top_k = 0;
    int32_t attention_shared_kv_layers = 0;
    float rope_freq_base = 0.0f;
    float rope_freq_base_swa = 0.0f;
    float rope_scale_linear = 0.0f;
    int32_t rope_dimension_sections_count = 0;
    std::array<int32_t, k_max_rope_dimension_sections> rope_dimension_sections = {};
    float rope_scaling_factor = 0.0f;
    float rope_scaling_attn_factor = 0.0f;
    int32_t rope_scaling_orig_ctx_len = 0;
    bool rope_scaling_finetuned = false;
    float rope_scaling_yarn_log_multiplier = 0.0f;
    float rope_scaling_yarn_ext_factor = 0.0f;
    float rope_scaling_yarn_attn_factor = 0.0f;
    float rope_scaling_yarn_beta_fast = 0.0f;
    float rope_scaling_yarn_beta_slow = 0.0f;
    int32_t ssm_conv_kernel = 0;
    int32_t ssm_inner_size = 0;
    int32_t ssm_state_size = 0;
    int32_t ssm_time_step_rank = 0;
    int32_t ssm_group_count = 0;
    bool ssm_dt_b_c_rms = false;
    int32_t kda_head_dim = 0;
    int32_t wkv_head_size = 0;
    int32_t posnet_embd = 0;
    int32_t posnet_block_count = 0;
    int32_t convnext_embd = 0;
    int32_t convnext_block_count = 0;
    int32_t shortconv_l_cache = 0;
  };

  struct vocab_entry {
    uint32_t text_offset = 0;
    uint32_t text_length = 0;
    float score = 0.0f;
    int32_t type = 0;
  };

  struct vocab {
    uint32_t n_tokens = 0;
    uint32_t n_token_types = 0;
    uint32_t token_bytes_used = 0;
    uint32_t n_merges = 0;
    uint32_t merge_bytes_used = 0;
    uint32_t precompiled_charsmap_size = 0;

    std::array<char, k_max_tokenizer_model> tokenizer_model_name = {};
    std::array<char, k_max_tokenizer_pre> tokenizer_pre_name = {};
    std::array<char, k_max_vocab_bytes> token_storage = {};
    std::array<char, k_max_merge_bytes> merge_storage = {};

    std::array<vocab_entry, k_max_vocab_tokens> entries = {};
    std::array<uint32_t, k_max_merges> merge_offsets = {};
    std::array<uint32_t, k_max_merges> merge_lengths = {};
    std::array<uint8_t, k_max_precompiled_charsmap_bytes> precompiled_charsmap = {};
    static constexpr uint32_t k_attr_flag_bytes = (k_max_vocab_tokens + 7) / 8;
    std::array<uint8_t, k_attr_flag_bytes> lstrip_flags = {};
    std::array<uint8_t, k_attr_flag_bytes> rstrip_flags = {};

    tokenizer_model tokenizer_model_id = tokenizer_model::UNKNOWN;
    tokenizer_pre tokenizer_pre_id = tokenizer_pre::DEFAULT;

    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t eot_id = -1;
    int32_t eom_id = -1;
    int32_t unk_id = -1;
    int32_t sep_id = -1;
    int32_t pad_id = -1;
    int32_t cls_id = -1;
    int32_t mask_id = -1;
    int32_t prefix_id = -1;
    int32_t suffix_id = -1;
    int32_t middle_id = -1;
    int32_t fim_pre_id = -1;
    int32_t fim_suf_id = -1;
    int32_t fim_mid_id = -1;
    int32_t fim_pad_id = -1;
    int32_t fim_rep_id = -1;
    int32_t fim_sep_id = -1;

    bool add_bos = false;
    bool add_eos = false;
    bool add_sep = false;
    bool add_space_prefix = false;
    bool remove_extra_whitespaces = false;
    bool escape_whitespaces = true;
    bool treat_whitespace_as_suffix = false;
    bool ignore_merges = false;
  };

  struct metadata {
    struct string_view {
      uint32_t offset = 0;
      uint32_t length = 0;
    };

    struct named_entity {
      string_view name = {};
      string_view author = {};
      string_view version = {};
      string_view organization = {};
      string_view description = {};
      string_view url = {};
      string_view doi = {};
      string_view uuid = {};
      string_view repo_url = {};
    };

    struct general {
      string_view type = {};
      int32_t quantization_version = -1;
      int32_t alignment = 0;
      int32_t file_type = -1;

      string_view name = {};
      string_view author = {};
      string_view version = {};
      string_view organization = {};
      string_view finetune = {};
      string_view basename = {};
      string_view description = {};
      string_view quantized_by = {};
      string_view size_label = {};

      string_view license = {};
      string_view license_name = {};
      string_view license_link = {};

      string_view url = {};
      string_view doi = {};
      string_view uuid = {};
      string_view repo_url = {};

      string_view source_url = {};
      string_view source_doi = {};
      string_view source_uuid = {};
      string_view source_repo_url = {};
      string_view source_hf_repo = {};

      uint32_t base_model_count = 0;
      std::array<named_entity, k_max_metadata_entities> base_models = {};

      uint32_t dataset_count = 0;
      std::array<named_entity, k_max_metadata_entities> datasets = {};

      uint32_t tag_count = 0;
      std::array<string_view, k_max_metadata_list> tags = {};

      uint32_t language_count = 0;
      std::array<string_view, k_max_metadata_list> languages = {};
    };

    struct sampling {
      string_view sequence = {};
      int32_t top_k = 0;
      float top_p = 0.0f;
      float min_p = 0.0f;
      float xtc_probability = 0.0f;
      float xtc_threshold = 0.0f;
      float temp = 0.0f;
      int32_t penalty_last_n = 0;
      float penalty_repeat = 0.0f;
      int32_t mirostat = 0;
      float mirostat_tau = 0.0f;
      float mirostat_eta = 0.0f;
    };

    struct tokenizer {
      string_view hf_json = {};
      string_view rwkv_world = {};
      string_view chat_template = {};
      uint32_t chat_template_count = 0;
      std::array<string_view, k_max_metadata_list> chat_template_names = {};
      std::array<string_view, k_max_metadata_list> chat_template_values = {};
    };

    struct classifier {
      uint32_t label_count = 0;
      std::array<string_view, k_max_metadata_list> labels = {};
    };

    struct adapter {
      string_view type = {};
      float lora_alpha = 0.0f;
      string_view lora_task_name = {};
      string_view lora_prompt_prefix = {};
      uint32_t alora_invocation_count = 0;
      std::array<uint32_t, k_max_metadata_arrays> alora_invocation_tokens = {};
    };

    struct imatrix {
      int32_t chunk_count = 0;
      int32_t chunk_size = 0;
      uint32_t dataset_count = 0;
      std::array<string_view, k_max_metadata_list> datasets = {};
    };

    struct clip {
      bool has_vision_encoder = false;
      bool has_audio_encoder = false;
      bool has_llava_projector = false;
      bool use_gelu = false;
      bool use_silu = false;
      string_view projector_type = {};
    };

    struct clip_vision {
      string_view projector_type = {};
      int32_t image_size = 0;
      int32_t image_min_pixels = 0;
      int32_t image_max_pixels = 0;
      int32_t preproc_image_size = 0;
      int32_t patch_size = 0;
      int32_t embedding_length = 0;
      int32_t feed_forward_length = 0;
      int32_t projection_dim = 0;
      int32_t block_count = 0;
      int32_t spatial_merge_size = 0;
      int32_t n_wa_pattern = 0;
      int32_t window_size = 0;
      int32_t attention_head_count = 0;
      float attention_layer_norm_epsilon = 0.0f;
      int32_t projector_scale_factor = 0;
      uint32_t image_mean_count = 0;
      std::array<float, k_max_clip_image_stats> image_mean = {};
      uint32_t image_std_count = 0;
      std::array<float, k_max_clip_image_stats> image_std = {};
      uint32_t wa_layer_index_count = 0;
      std::array<uint32_t, k_max_clip_layer_indexes> wa_layer_indexes = {};
      uint32_t deepstack_layer_count = 0;
      std::array<uint8_t, k_max_clip_layer_indexes> deepstack_layers = {};
    };

    struct clip_audio {
      string_view projector_type = {};
      int32_t num_mel_bins = 0;
      int32_t embedding_length = 0;
      int32_t feed_forward_length = 0;
      int32_t projection_dim = 0;
      int32_t block_count = 0;
      int32_t attention_head_count = 0;
      float attention_layer_norm_epsilon = 0.0f;
      int32_t projector_stack_factor = 0;
    };

    struct rope {
      string_view scaling_type = {};
    };

    struct llm_strings {
      string_view tensor_data_layout = {};
    };

    struct xielu {
      uint32_t alpha_p_count = 0;
      std::array<float, k_max_xielu_values> alpha_p = {};
      uint32_t alpha_n_count = 0;
      std::array<float, k_max_xielu_values> alpha_n = {};
      uint32_t beta_count = 0;
      std::array<float, k_max_xielu_values> beta = {};
      uint32_t eps_count = 0;
      std::array<float, k_max_xielu_values> eps = {};
    };

    struct diffusion {
      bool shift_logits = false;
    };

    general general_data = {};
    sampling sampling_data = {};
    tokenizer tokenizer_data = {};
    classifier classifier_data = {};
    adapter adapter_data = {};
    imatrix imatrix_data = {};
    clip clip_data = {};
    clip_vision clip_vision_data = {};
    clip_audio clip_audio_data = {};
    rope rope_data = {};
    llm_strings llm_strings_data = {};
    xielu xielu_data = {};
    diffusion diffusion_data = {};

    uint32_t blob_bytes_used = 0;
    std::array<char, k_max_metadata_blob_bytes> blob = {};
  };

  int32_t n_layers = 0;
  uint32_t n_tensors = 0;
  uint32_t name_bytes_used = 0;
  std::array<char, k_max_architecture_name> architecture_name = {};
  std::array<char, k_max_name_bytes> name_storage = {};
  std::array<tensor_record, k_max_tensors> tensors = {};
  const void * weights_data = nullptr;
  uint64_t weights_size = 0;
  bool weights_mapped = false;
  uint16_t weights_split_count = 1;
  std::array<uint64_t, k_max_split_files> weights_split_sizes = {};
  std::array<uint64_t, k_max_split_files> weights_split_offsets = {};
  hparams params = {};
  vocab vocab_data = {};
  metadata meta = {};
};

}  // namespace emel::model
