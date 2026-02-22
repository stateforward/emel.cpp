#pragma once

#include <array>
#include <initializer_list>
#include <string_view>

#include "emel/model/data.hpp"

namespace emel::tokenizer::bpe::detail {

constexpr size_t k_max_regex_exprs = 6;

struct regex_list {
  std::array<std::string_view, k_max_regex_exprs> exprs = {};
  size_t count = 0;
};

inline regex_list regex_for(const emel::model::data::tokenizer_pre pre) {
  auto make = [](std::initializer_list<std::string_view> list) {
    regex_list out{};
    size_t idx = 0;
    for (const auto & expr : list) {
      if (idx >= out.exprs.size()) {
        break;
      }
      out.exprs[idx++] = expr;
    }
    out.count = idx;
    return out;
  };

  using tokenizer_pre = emel::model::data::tokenizer_pre;
  switch (pre) {
    case tokenizer_pre::LLAMA3:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::JAIS2:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)"
          "|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)"
          "|\\s{1,2}(?!\\S)|\\s{1}",
      });
    case tokenizer_pre::DBRX:
    case tokenizer_pre::SMAUG:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::DEEPSEEK_LLM:
      return make({
          "[\\r\\n]",
          "\\s?[A-za-z\\xc2\\xb5\\xc3\\x80-\\xc3\\x96\\xc3\\x98-\\xc3\\xb6\\xc3\\xb8-\\xc6\\xba\\xc6\\xbc-\\xc6\\xbf\\xc7\\x84-\\xca\\x93\\xca\\x95-\\xca\\xaf\\xcd\\xb0-\\xcd\\xb3\\xcd\\xb6\\xcd\\xb7\\xcd\\xbb-\\xcd\\xbd\\xcd\\xbf\\xce\\x86\\xce\\x88-\\xce\\x8a\\xce\\x8c\\xce\\x8e-\\xce\\xa1\\xce\\xa3-\\xcf\\xb5\\xcf\\xb7-\\xd2\\x81\\xd2\\x8a-\\xd4\\xaf\\xd4\\xb1-\\xd5\\x96\\xe1\\x82\\xa0-\\xe1\\x83\\x85\\xe1\\x8e\\xa0-\\xe1\\x8f\\xb5\\xe1\\x8f\\xb8-\\xe1\\x8f\\xbd\\xe1\\xb2\\x90-\\xe1\\xb2\\xba\\xe1\\xb2\\xbd-\\xe1\\xb2\\xbf\\xe1\\xb4\\x80-\\xe1\\xb4\\xab\\xe1\\xb5\\xab-\\xe1\\xb5\\xb7\\xe1\\xb5\\xb9-\\xe1\\xb6\\x9a\\xe1\\xb8\\x80-\\xe1\\xbc\\x95\\xe1\\xbc\\x98-\\xe1\\xbc\\x9d\\xe1\\xbc\\xa0-\\xe1\\xbd\\x85\\xe1\\xbd\\x88-\\xe1\\xbd\\x8d\\xe1\\xbd\\x90-\\xe1\\xbd\\x97\\xe1\\xbd\\x99\\xe1\\xbd\\x9b\\xe1\\xbd\\x9d\\xe1\\xbd\\x9f-\\xe1\\xbd\\xbd\\xe1\\xbe\\x80-\\xe1\\xbe\\xb4\\xe1\\xbe\\xb6-\\xe1\\xbe\\xbc\\xe1\\xbe\\xbe\\xe1\\xbf\\x82-\\xe1\\xbf\\x84\\xe1\\xbf\\x86-\\xe1\\xbf\\x8c\\xe1\\xbf\\x90-\\xe1\\xbf\\x93\\xe1\\xbf\\x96-\\xe1\\xbf\\x9b\\xe1\\xbf\\xa0-\\xe1\\xbf\\xac\\xe1\\xbf\\xb2-\\xe1\\xbf\\xb4\\xe1\\xbf\\xb6-\\xe1\\xbf\\xbc\\xe2\\x84\\x82\\xe2\\x84\\x87\\xe2\\x84\\x8a-\\xe2\\x84\\x93\\xe2\\x84\\x95\\xe2\\x84\\x99-\\xe2\\x84\\x9d\\xe2\\x84\\xa4\\xe2\\x84\\xa6\\xe2\\x84\\xa8\\xe2\\x84\\xaa-\\xe2\\x84\\xad\\xe2\\x84\\xaf-\\xe2\\x84\\xb4\\xe2\\x84\\xb9\\xe2\\x84\\xbc-\\xe2\\x84\\xbf\\xe2\\x85\\x85-\\xe2\\x85\\x89\\xe2\\x85\\x8e\\xe2\\x86\\x83\\xe2\\x86\\x84\\xe2\\xb0\\x80-\\xe2\\xb1\\xbb\\xe2\\xb1\\xbe-\\xe2\\xb3\\xa4\\xe2\\xb3\\xab-\\xe2\\xb3\\xae\\xe2\\xb3\\xb2\\xe2\\xb3\\xb3\\xea\\x99\\x80-\\xea\\x99\\xad\\xea\\x9a\\x80-\\xea\\x9a\\x9b\\xea\\x9c\\xa2-\\xea\\x9d\\xaf\\xea\\x9d\\xb1-\\xea\\x9e\\x87\\xea\\x9e\\x8b-\\xea\\x9e\\x8e\\xea\\xad\\xb0-\\xea\\xae\\xbf\\xef\\xac\\x80-\\xef\\xac\\x86\\xef\\xac\\x93-\\xef\\xac\\x97\\xef\\xbc\\xa1-\\xef\\xbc\\xba\\xef\\xbd\\x81-\\xef\\xbd\\x9a\\xf0\\x90\\x90\\x80-\\xf0\\x90\\x91\\x8f\\xf0\\x90\\x92\\xb0-\\xf0\\x90\\x93\\x93\\xf0\\x90\\x93\\x98-\\xf0\\x90\\x93\\xbb\\xf0\\x90\\xb2\\x80-\\xf0\\x90\\xb2\\xb2\\xf0\\x90\\xb3\\x80-\\xf0\\x90\\xb3\\xb2\\xf0\\x91\\xa2\\xa0-\\xf0\\x91\\xa3\\x9f\\xf0\\x9e\\xa4\\x80-\\xf0\\x9e\\xa5\\x83]+",
          "\\s?[!-/:-~\\xef\\xbc\\x81-\\xef\\xbc\\x8f\\xef\\xbc\\x9a-\\xef\\xbd\\x9e\\xe2\\x80\\x98-\\xe2\\x80\\x9f\\xe3\\x80\\x80-\\xe3\\x80\\x82]+",
          "\\s+$",
          "[\\xe4\\xb8\\x80-\\xe9\\xbe\\xa5\\xe0\\xa0\\x80-\\xe4\\xb8\\x80\\xea\\xb0\\x80-\\xed\\x9f\\xbf]+",
          "\\p{N}+",
      });
    case tokenizer_pre::DEEPSEEK3_LLM:
    case tokenizer_pre::HUNYUAN_DENSE:
    case tokenizer_pre::JOYAI_LLM:
      return make({
          "\\p{N}{1,3}",
          "[\\xe4\\xb8\\x80-\\xe9\\xbe\\xa5\\xe3\\x81\\x80-\\xe3\\x82\\x9f\\xe3\\x82\\xa0-\\xe3\\x83\\xbf]+",
          "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-za-z]+|"
          "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::YOUTU:
      return make({
          "[\\xea\\xb0\\x80-\\xed\\x9e\\xa3\\xe3\\x84\\xb1-\\xe3\\x86\\x8e]+|"
          "[\\xef\\xbc\\x81\\xe2\\x80\\xa6\\xe2\\x80\\x9c\\xe2\\x80\\x9d\\xe2\\x80\\x98\\xe2\\x80\\x99"
          "\\xe2\\x80\\x94\\xef\\xbc\\x9a\\xef\\xbc\\x9b\\xef\\xbc\\x8c\\xe3\\x80\\x81-\\xe3\\x80\\xbf"
          "\\xef\\xb8\\xb0-\\xef\\xb9\\x8f]+|[\\xe3\\x84\\x85-\\xe3\\x84\\xaf]+|"
          "[\\xe4\\xb8\\x80-\\xe9\\xbe\\xa5\\xe3\\x81\\x80-\\xe3\\x82\\x9f\\xe3\\x82\\xa0-\\xe3\\x83\\xbf]+",
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{lu}\\p{lt}\\p{lm}\\p{lo}\\p{M}]*"
          "[\\p{ll}\\p{lm}\\p{lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]"
          "|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{lu}\\p{lt}"
          "\\p{lm}\\p{lo}\\p{M}]+[\\p{ll}\\p{lm}\\p{lo}\\p{M}]*(?:'[sS]|"
          "'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}|"
          " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::DEEPSEEK_CODER:
      return make({
          "[\\r\\n]",
          "\\s?\\p{L}+",
          "\\s?\\p{P}+",
          "[\\xe4\\xb8\\x80-\\xe9\\xbe\\xa5\\xe0\\xa0\\x80-\\xe4\\xb8\\x80\\xea\\xb0\\x80-\\xed\\x9f\\xbf]+",
          "\\p{N}",
      });
    case tokenizer_pre::FALCON:
      return make({
          "[\\p{P}\\$\\+<=>\\^~\\|`]+",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
          "[0-9][0-9][0-9]",
      });
    case tokenizer_pre::STARCODER:
    case tokenizer_pre::REFACT:
    case tokenizer_pre::COMMAND_R:
    case tokenizer_pre::SMOLLM:
    case tokenizer_pre::CODESHELL:
    case tokenizer_pre::EXAONE:
    case tokenizer_pre::MINERVA:
      return make({
          "\\p{N}",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
      });
    case tokenizer_pre::GPT2:
    case tokenizer_pre::MPT:
    case tokenizer_pre::OLMO:
    case tokenizer_pre::JAIS:
    case tokenizer_pre::TRILLION:
    case tokenizer_pre::GRANITE_DOCLING:
      return make({
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
      });
    case tokenizer_pre::QWEN35:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::STABLELM2:
    case tokenizer_pre::QWEN2:
    case tokenizer_pre::HUNYUAN:
    case tokenizer_pre::SOLAR_OPEN:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
          "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::PORO:
    case tokenizer_pre::BLOOM:
    case tokenizer_pre::GPT3_FINNISH:
      return make({
          " ?[^(\\\\s|.,!?\\xe2\\x80\\xa6\\xe3\\x80\\x82\\xef\\xbc\\x8c\\xe3\\x80\\x81\\xe0\\xa5\\xa4\\xe0\\xa5\\xa4\\xd8\\x8c)]+",
      });
    case tokenizer_pre::CHATGLM4:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::VIKING:
      return make({
          " ?[^(\\\\s|.,!?\\xe2\\x80\\xa6\\xe3\\x80\\x82\\xef\\xbc\\x8c\\xe3\\x80\\x81\\xe0\\xa5\\xa4\\xe0\\xa5\\xa4\\xd8\\x8c)]+",
          "\\p{N}",
      });
    case tokenizer_pre::TEKKEN:
      return make({
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|"
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|"
          "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::CHAMELEON:
      return make({
          "<sentinel:[0-9]+>",
          "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
          "([\\t\\n]|    |  )",
          "\\p{N}",
          "[\\p{P}!-/:-@\\[-`{-~]",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
      });
    case tokenizer_pre::GPT4O:
    case tokenizer_pre::MINIMAX_M2:
      return make({
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|"
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|"
          "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::TINY_AYA:
      return make({
          "\\d{1,3}(?=(?:\\d{3})*\\b)",
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{lu}\\p{lt}\\p{lm}\\p{lo}\\p{M}]*"
          "[\\p{ll}\\p{lm}\\p{lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]"
          "|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{lu}\\p{lt}"
          "\\p{lm}\\p{lo}\\p{M}]+[\\p{ll}\\p{lm}\\p{lo}\\p{M}]*(?:'[sS]|"
          "'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}|"
          " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::KIMI_K2:
      return make({
          "\\p{han}+",
      });
    case tokenizer_pre::SUPERBPE:
      return make({
          "\\p{N}+",
          "(?=(\\d{3})+(?!\\d))",
      });
    case tokenizer_pre::BAILINGMOE:
      return make({
          "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::SEED_CODER:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+"
          "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::GROK_2:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::AFMOE:
      return make({
          "\\p{AFMoE_digits}",
          "[\\xe4\\xb8\\x80-\\xe9\\xbf\\xbf\\xe3\\x90\\x80-\\xe4\\xb6\\xbf\\xe8\\xb1\\x88-\\xef\\xab\\xbf\\xe3\\x81\\x80-\\xe3\\x82\\x9f\\xe3\\x82\\xa0-\\xe3\\x83\\xbf\\xef\\xbd\\xa5-\\xef\\xbe\\x9f\\xe2\\xbc\\x80-\\xe2\\xbf\\x9f\\xe0\\xb9\\x80-\\xe0\\xb9\\xbf\\xe0\\xba\\x80-\\xe0\\xbb\\xbf\\xe1\\x80\\x80-\\xe1\\x82\\x9f\\xea\\xa9\\xa0-\\xea\\xa9\\xbf\\xea\\xa7\\xa0-\\xea\\xa7\\xbf\\xea\\x9d\\x80-\\xea\\x9d\\xbf\\xea\\xa0\\x80-\\xea\\xa0\\xbf\\xea\\xa1\\x80-\\xea\\xa1\\xbf\\xea\\xa2\\x80-\\xea\\xa2\\xbf]+",
          "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-za-z]+|"
          "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+"
          "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      });
    case tokenizer_pre::EXAONE_MOE:
      return make({
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
          "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|"
          "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
      });
    default:
      return make({
          "[\\p{P}\\$\\+<=>\\^~\\|]+",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
          "\\p{N}+",
          "[0-9][0-9][0-9]",
      });
  }
}

inline regex_list regex_for(const emel::model::data::vocab & vocab) {
  return regex_for(vocab.tokenizer_pre_id);
}

}  // namespace emel::tokenizer::bpe::detail
