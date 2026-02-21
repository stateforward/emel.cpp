#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::encoder::detail {

constexpr int32_t k_token_null = -1;

struct naive_trie {
  struct node {
    std::array<int32_t, 256> next = {};
    bool has_value = false;
    int32_t value = 0;

    node() {
      next.fill(-1);
    }

    const node * traverse(const char c) const {
      const int32_t idx = next[static_cast<uint8_t>(c)];
      if (idx < 0) {
        return nullptr;
      }
      return &nodes_ref->at(static_cast<size_t>(idx));
    }

    node * traverse(const char c) {
      const int32_t idx = next[static_cast<uint8_t>(c)];
      if (idx < 0) {
        return nullptr;
      }
      return &nodes_ref->at(static_cast<size_t>(idx));
    }

    std::vector<node> *nodes_ref = nullptr;
  };

  std::vector<node> nodes = {};

  naive_trie() {
    nodes.emplace_back();
    nodes.back().nodes_ref = &nodes;
  }

  void insert(const char *text, const size_t len, const int32_t value) {
    size_t idx = 0;
    for (size_t i = 0; i < len; ++i) {
      node &node = nodes[idx];
      const uint8_t byte = static_cast<uint8_t>(text[i]);
      if (node.next[byte] < 0) {
        node.next[byte] = static_cast<int32_t>(nodes.size());
        nodes.emplace_back();
        nodes.back().nodes_ref = &nodes;
      }
      idx = static_cast<size_t>(node.next[byte]);
    }
    nodes[idx].has_value = true;
    nodes[idx].value = value;
  }

  const node * traverse(const char c) const {
    const int32_t idx = nodes[0].next[static_cast<uint8_t>(c)];
    if (idx < 0) {
      return nullptr;
    }
    return &nodes[static_cast<size_t>(idx)];
  }
};

using naive_trie = naive_trie;

struct spm_bigram {
  struct comparator {
    bool operator()(const spm_bigram &l, const spm_bigram &r) const {
      return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
  };

  using queue_storage = std::vector<spm_bigram>;
  int left = 0;
  int right = 0;
  float score = 0.0f;
  size_t size = 0;
};

using spm_bigram = spm_bigram;

struct bpe_bigram {
  struct comparator {
    bool operator()(const bpe_bigram &l, const bpe_bigram &r) const {
      return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
  };

  int left = 0;
  int right = 0;
  int rank = 0;
  size_t size = 0;
};

using bpe_bigram = bpe_bigram;

constexpr uint32_t next_pow2(uint32_t v) {
  if (v == 0) {
    return 1;
  }
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

constexpr uint32_t k_token_hash_size = next_pow2(
  static_cast<uint32_t>(emel::model::data::k_max_vocab_tokens * 3u / 2u));
constexpr uint32_t k_merge_hash_size = next_pow2(
  static_cast<uint32_t>(emel::model::data::k_max_merges * 3u / 2u));
static_assert((k_token_hash_size & (k_token_hash_size - 1)) == 0, "token hash size");
static_assert((k_merge_hash_size & (k_merge_hash_size - 1)) == 0, "merge hash size");

struct token_map {
  std::unique_ptr<uint32_t[]> hashes = nullptr;
  std::unique_ptr<int32_t[]> values = nullptr;
  uint32_t count = 0;

  token_map()
      : hashes(std::make_unique<uint32_t[]>(k_token_hash_size)),
        values(std::make_unique<int32_t[]>(k_token_hash_size)) {
    clear();
  }

  void clear() {
    std::fill_n(hashes.get(), k_token_hash_size, 0u);
    std::fill_n(values.get(), k_token_hash_size, k_token_null);
    count = 0;
  }

  bool empty() const {
    return count == 0;
  }
};

struct merge_map {
  std::unique_ptr<uint32_t[]> hashes = nullptr;
  std::unique_ptr<int32_t[]> values = nullptr;
  uint32_t count = 0;

  merge_map()
      : hashes(std::make_unique<uint32_t[]>(k_merge_hash_size)),
        values(std::make_unique<int32_t[]>(k_merge_hash_size)) {
    clear();
  }

  void clear() {
    std::fill_n(hashes.get(), k_merge_hash_size, 0u);
    std::fill_n(values.get(), k_merge_hash_size, k_token_null);
    count = 0;
  }

  bool empty() const {
    return count == 0;
  }
};

constexpr size_t k_max_encode_symbols = 16384;
constexpr size_t k_max_encode_bytes = k_max_encode_symbols * 4;
constexpr size_t k_max_encode_segments = 1024;

struct encode_scratch {
  std::array<uint32_t, k_max_encode_symbols> offsets = {};
  std::array<uint32_t, k_max_encode_symbols> lengths = {};
  std::array<int32_t, k_max_encode_symbols> prev = {};
  std::array<int32_t, k_max_encode_symbols> next = {};
  std::array<char, k_max_encode_bytes> buffer = {};
  std::array<char, k_max_encode_bytes> buffer_alt = {};
  std::array<std::string_view, k_max_encode_segments> segments = {};
  uint32_t symbol_count = 0;
};

struct encode_result {
  int32_t token_count = 0;
  int32_t error = EMEL_OK;
};

}  // namespace emel::encoder::detail
