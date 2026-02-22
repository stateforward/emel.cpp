#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

std::filesystem::path models_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "models";
#else
  return std::filesystem::path("tests") / "models";
#endif
}

std::filesystem::path parity_texts_dir() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::filesystem::path root = PARITYCHECKER_REPO_ROOT;
  return root / "tests" / "tokenizer" / "parity_texts";
#else
  return std::filesystem::path("tests") / "tokenizer" / "parity_texts";
#endif
}

bool file_exists(const std::filesystem::path & path) {
  std::FILE * file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  std::fclose(file);
  return true;
}

std::vector<std::string> discover_models() {
  std::vector<std::string> models;
  const auto dir = models_dir();
  if (!std::filesystem::exists(dir)) {
    return models;
  }
  for (const auto & entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto path = entry.path();
    if (path.extension() != ".gguf") {
      continue;
    }
    models.push_back(path.string());
  }
  std::sort(models.begin(), models.end());
  return models;
}

struct parity_case {
  std::string label;
  std::filesystem::path text_path;
  bool add_special = false;
  bool parse_special = false;
};

std::string quote_arg_posix(const std::string & arg) {
  std::string out = "'";
  for (const char c : arg) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

std::string quote_arg_windows(const std::string & arg) {
  std::string out = "\"";
  for (const char c : arg) {
    if (c == '"') {
      out += "\\\"";
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
  return out;
}

std::string special_text_for_model(const std::filesystem::path & model_path) {
  const std::string name = model_path.filename().string();
  const auto texts = parity_texts_dir();
  if (name.find("Llama-") != std::string::npos) {
    return (texts / "special_llama.txt").string();
  }
  if (name.find("distilgpt2") != std::string::npos) {
    return (texts / "special_gpt2.txt").string();
  }
  if (name.find("bert-base-uncased") != std::string::npos) {
    return (texts / "special_bert.txt").string();
  }
  if (name.find("flan-t5") != std::string::npos) {
    return (texts / "special_t5.txt").string();
  }
  if (name.find("rwkv") != std::string::npos) {
    return (texts / "special_rwkv.txt").string();
  }
  return {};
}

std::vector<parity_case> base_cases() {
  const auto texts = parity_texts_dir();
  return {
    {"basic_add_special", texts / "basic.txt", true, false},
    {"basic_no_special", texts / "basic.txt", false, false},
    {"whitespace", texts / "whitespace.txt", true, false},
    {"unicode", texts / "unicode.txt", true, false},
    {"long", texts / "long.txt", false, false},
  };
}

bool run_paritychecker_process(const std::string & model, const parity_case & test_case) {
  std::string command;
#if defined(_WIN32)
  command = ".\\paritychecker --model ";
  command += quote_arg_windows(model);
  command += " --text-file ";
  command += quote_arg_windows(test_case.text_path.string());
#else
  command = "ulimit -s 8192; ./paritychecker --model ";
  command += quote_arg_posix(model);
  command += " --text-file ";
  command += quote_arg_posix(test_case.text_path.string());
#endif
  if (test_case.add_special) {
    command += " --add-special";
  }
  if (test_case.parse_special) {
    command += " --parse-special";
  }
  const int status = std::system(command.c_str());
  if (status == -1) {
    return false;
  }
#if defined(_WIN32)
  return status == 0;
#else
  if (!WIFEXITED(status)) {
    return false;
  }
  return WEXITSTATUS(status) == 0;
#endif
}

}  // namespace

TEST_CASE("paritychecker matches llama tokens across tiny models") {
  const std::vector<std::string> models = discover_models();
  const std::vector<parity_case> cases = base_cases();

  REQUIRE(!models.empty());
  for (const auto & model : models) {
    INFO("model: " << model);
    REQUIRE(file_exists(std::filesystem::path(model)));
    for (const auto & test_case : cases) {
      INFO("case: " << test_case.label);
      REQUIRE(file_exists(test_case.text_path));
      CHECK(run_paritychecker_process(model, test_case));
    }
    const std::string special_text = special_text_for_model(model);
    if (!special_text.empty()) {
      INFO("case: special_parse");
      REQUIRE(file_exists(std::filesystem::path(special_text)));
      parity_case special_case;
      special_case.label = "special_parse";
      special_case.text_path = special_text;
      special_case.add_special = true;
      special_case.parse_special = true;
      CHECK(run_paritychecker_process(model, special_case));
    }
  }
}
