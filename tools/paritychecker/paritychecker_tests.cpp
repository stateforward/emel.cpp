#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cstdlib>
#include <string>

#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

std::string model_path() {
#ifdef PARITYCHECKER_REPO_ROOT
  std::string root = PARITYCHECKER_REPO_ROOT;
  if (!root.empty() && root.back() == '/') {
    root.pop_back();
  }
  return root + "/tests/models/Llama-68M-Chat-v1-Q2_K.gguf";
#else
  return "tests/models/Llama-68M-Chat-v1-Q2_K.gguf";
#endif
}

bool run_paritychecker_process() {
  std::string command;
#if defined(_WIN32)
  command = ".\\paritychecker --model \"";
  command += model_path();
  command += "\" --text \"hello world\" --add-special";
#else
  command = "ulimit -s 8192; ./paritychecker --model '";
  command += model_path();
  command += "' --text 'hello world' --add-special";
#endif
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

TEST_CASE("paritychecker matches llama tokens on tiny gguf") {
  CHECK(run_paritychecker_process());
}
