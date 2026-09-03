#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__linux__)
extern char **environ;
#endif

namespace {

#if defined(__linux__)

struct sha256_context {
  std::array<std::uint32_t, 8> state{
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<unsigned char, 64> block{};
  std::uint64_t byte_count = 0;
  std::size_t block_size = 0;
};

constexpr std::array<std::uint32_t, 64> k_sha256_round_constants{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

constexpr std::uint32_t rotate_right(const std::uint32_t value,
                                     const unsigned int count) noexcept {
  return (value >> count) | (value << (32u - count));
}

void sha256_transform(sha256_context &context) noexcept {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t index = 0; index < 16; ++index) {
    const std::size_t offset = index * 4;
    schedule[index] =
        (static_cast<std::uint32_t>(context.block[offset]) << 24u) |
        (static_cast<std::uint32_t>(context.block[offset + 1]) << 16u) |
        (static_cast<std::uint32_t>(context.block[offset + 2]) << 8u) |
        static_cast<std::uint32_t>(context.block[offset + 3]);
  }
  for (std::size_t index = 16; index < schedule.size(); ++index) {
    const std::uint32_t s0 = rotate_right(schedule[index - 15], 7u) ^
                             rotate_right(schedule[index - 15], 18u) ^
                             (schedule[index - 15] >> 3u);
    const std::uint32_t s1 = rotate_right(schedule[index - 2], 17u) ^
                             rotate_right(schedule[index - 2], 19u) ^
                             (schedule[index - 2] >> 10u);
    schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
  }

  std::uint32_t a = context.state[0];
  std::uint32_t b = context.state[1];
  std::uint32_t c = context.state[2];
  std::uint32_t d = context.state[3];
  std::uint32_t e = context.state[4];
  std::uint32_t f = context.state[5];
  std::uint32_t g = context.state[6];
  std::uint32_t h = context.state[7];
  for (std::size_t index = 0; index < schedule.size(); ++index) {
    const std::uint32_t sigma1 =
        rotate_right(e, 6u) ^ rotate_right(e, 11u) ^ rotate_right(e, 25u);
    const std::uint32_t choice = (e & f) ^ (~e & g);
    const std::uint32_t temporary1 =
        h + sigma1 + choice + k_sha256_round_constants[index] +
        schedule[index];
    const std::uint32_t sigma0 =
        rotate_right(a, 2u) ^ rotate_right(a, 13u) ^ rotate_right(a, 22u);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sigma0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  context.state[0] += a;
  context.state[1] += b;
  context.state[2] += c;
  context.state[3] += d;
  context.state[4] += e;
  context.state[5] += f;
  context.state[6] += g;
  context.state[7] += h;
}

void sha256_update(sha256_context &context, const unsigned char *data,
                   std::size_t size) noexcept {
  context.byte_count += size;
  while (size != 0u) {
    const std::size_t available = context.block.size() - context.block_size;
    const std::size_t copied = size < available ? size : available;
    std::memcpy(context.block.data() + context.block_size, data, copied);
    context.block_size += copied;
    data += copied;
    size -= copied;
    if (context.block_size == context.block.size()) {
      sha256_transform(context);
      context.block_size = 0u;
    }
  }
}

std::array<unsigned char, 32> sha256_finish(sha256_context &context) noexcept {
  const std::uint64_t bit_count = context.byte_count * 8u;
  const unsigned char marker = 0x80u;
  sha256_update(context, &marker, 1u);
  const unsigned char zero = 0u;
  while (context.block_size != 56u) {
    sha256_update(context, &zero, 1u);
  }
  std::array<unsigned char, 8> encoded_length{};
  for (std::size_t index = 0; index < encoded_length.size(); ++index) {
    encoded_length[encoded_length.size() - index - 1u] =
        static_cast<unsigned char>(bit_count >> (index * 8u));
  }
  sha256_update(context, encoded_length.data(), encoded_length.size());

  std::array<unsigned char, 32> digest{};
  for (std::size_t index = 0; index < context.state.size(); ++index) {
    digest[index * 4] = static_cast<unsigned char>(context.state[index] >> 24u);
    digest[index * 4 + 1] =
        static_cast<unsigned char>(context.state[index] >> 16u);
    digest[index * 4 + 2] =
        static_cast<unsigned char>(context.state[index] >> 8u);
    digest[index * 4 + 3] = static_cast<unsigned char>(context.state[index]);
  }
  return digest;
}

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "error: authenticated Needle Python exec: %s\n", message);
  std::exit(126);
}

[[noreturn]] void fail_errno(const char *operation) {
  std::fprintf(stderr, "error: authenticated Needle Python exec: %s: %s\n",
               operation, std::strerror(errno));
  std::exit(126);
}

unsigned char decode_hex(const char value) {
  if (value >= '0' && value <= '9') return static_cast<unsigned char>(value - '0');
  if (value >= 'a' && value <= 'f') return static_cast<unsigned char>(value - 'a' + 10);
  if (value >= 'A' && value <= 'F') return static_cast<unsigned char>(value - 'A' + 10);
  fail("expected SHA-256 must contain exactly 64 hexadecimal characters");
}

std::array<unsigned char, 32> parse_digest(const char *hex) {
  if (std::strlen(hex) != 64u) {
    fail("expected SHA-256 must contain exactly 64 hexadecimal characters");
  }
  std::array<unsigned char, 32> digest{};
  for (std::size_t index = 0; index < digest.size(); ++index) {
    digest[index] = static_cast<unsigned char>(
        (decode_hex(hex[index * 2]) << 4u) | decode_hex(hex[index * 2 + 1]));
  }
  return digest;
}

bool equal_digest(const std::array<unsigned char, 32> &left,
                  const std::array<unsigned char, 32> &right) noexcept {
  unsigned char difference = 0u;
  for (std::size_t index = 0; index < left.size(); ++index) {
    difference = static_cast<unsigned char>(difference | (left[index] ^ right[index]));
  }
  return difference == 0u;
}

void write_all(const int fd, const unsigned char *data, std::size_t size) {
  while (size != 0u) {
    const ssize_t written = ::write(fd, data, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      fail_errno("cannot copy interpreter into memfd");
    }
    if (written == 0) fail("short write while copying interpreter into memfd");
    data += static_cast<std::size_t>(written);
    size -= static_cast<std::size_t>(written);
  }
}

int copy_to_sealed_memfd(const char *source_path) {
  const int source = ::open(source_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (source < 0) fail_errno("cannot open canonical interpreter");
  struct stat metadata_before {};
  if (::fstat(source, &metadata_before) != 0) {
    fail_errno("cannot stat canonical interpreter");
  }
  if (!S_ISREG(metadata_before.st_mode) || (metadata_before.st_mode & 0111) == 0) {
    fail("canonical interpreter must be an executable regular file");
  }

  const int executable = static_cast<int>(
      ::syscall(SYS_memfd_create, "emel-needle-python",
                MFD_ALLOW_SEALING | MFD_CLOEXEC));
  if (executable < 0) {
    fail_errno("memfd_create(MFD_ALLOW_SEALING|MFD_CLOEXEC) failed");
  }

  std::array<unsigned char, 64u * 1024u> buffer{};
  for (;;) {
    const ssize_t count = ::read(source, buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR) continue;
      fail_errno("cannot read canonical interpreter");
    }
    if (count == 0) break;
    write_all(executable, buffer.data(), static_cast<std::size_t>(count));
  }
  struct stat metadata_after {};
  if (::fstat(source, &metadata_after) != 0) {
    fail_errno("cannot restat canonical interpreter");
  }
  if (metadata_before.st_dev != metadata_after.st_dev ||
      metadata_before.st_ino != metadata_after.st_ino ||
      metadata_before.st_size != metadata_after.st_size ||
      metadata_before.st_mtim.tv_sec != metadata_after.st_mtim.tv_sec ||
      metadata_before.st_mtim.tv_nsec != metadata_after.st_mtim.tv_nsec ||
      metadata_before.st_ctim.tv_sec != metadata_after.st_ctim.tv_sec ||
      metadata_before.st_ctim.tv_nsec != metadata_after.st_ctim.tv_nsec) {
    fail("canonical interpreter changed while it was copied");
  }
  if (::close(source) != 0) fail_errno("cannot close canonical interpreter");
  if (::fchmod(executable, 0500) != 0) fail_errno("cannot mark interpreter memfd executable");

  constexpr int required_seals =
      F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  if (::fcntl(executable, F_ADD_SEALS, required_seals) != 0) {
    fail_errno("cannot seal interpreter memfd");
  }
  const int actual_seals = ::fcntl(executable, F_GET_SEALS);
  if (actual_seals < 0) fail_errno("cannot verify interpreter memfd seals");
  if ((actual_seals & required_seals) != required_seals) {
    fail("interpreter memfd is missing required seals");
  }
  return executable;
}

std::array<unsigned char, 32> hash_fd(const int fd) {
  if (::lseek(fd, 0, SEEK_SET) < 0) fail_errno("cannot seek sealed interpreter memfd");
  sha256_context context{};
  std::array<unsigned char, 64u * 1024u> buffer{};
  for (;;) {
    const ssize_t count = ::read(fd, buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR) continue;
      fail_errno("cannot hash sealed interpreter memfd");
    }
    if (count == 0) break;
    sha256_update(context, buffer.data(), static_cast<std::size_t>(count));
  }
  return sha256_finish(context);
}

#ifdef EMEL_AUTHENTICATED_EXEC_TESTING
void run_test_barrier() {
  const char *ready_path = std::getenv("EMEL_AUTHENTICATED_EXEC_TEST_READY");
  const char *continue_path =
      std::getenv("EMEL_AUTHENTICATED_EXEC_TEST_CONTINUE");
  if (ready_path == nullptr && continue_path == nullptr) return;
  if (ready_path == nullptr || continue_path == nullptr) {
    fail("test barrier requires both ready and continue paths");
  }
  const int ready = ::open(ready_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (ready < 0) fail_errno("cannot create authenticated-exec test ready file");
  if (::close(ready) != 0) fail_errno("cannot close authenticated-exec test ready file");
  struct stat metadata {};
  while (::stat(continue_path, &metadata) != 0) {
    if (errno != ENOENT) fail_errno("cannot inspect authenticated-exec test continue file");
    ::usleep(1000u);
  }
}
#endif


bool allowed_environment_name(const char *entry) noexcept {
#ifdef EMEL_AUTHENTICATED_EXEC_TESTING
  constexpr std::array<const char *, 23> names{
      "HOME",           "XDG_CACHE_HOME", "TMPDIR",        "TMP",
      "TEMP",           "LANG",           "LANGUAGE",      "LC_ALL",
      "LC_CTYPE",       "LC_NUMERIC",     "LC_TIME",       "LC_COLLATE",
      "LC_MONETARY",    "LC_MESSAGES",    "LC_PAPER",      "LC_NAME",
      "LC_ADDRESS",     "LC_TELEPHONE",   "LC_MEASUREMENT", "LC_IDENTIFICATION",
      "NEEDLE_THREADS", "EMEL_AUTHENTICATED_EXEC_TEST_READY",
      "EMEL_AUTHENTICATED_EXEC_TEST_CONTINUE"};
#else
  constexpr std::array<const char *, 21> names{
      "HOME",           "XDG_CACHE_HOME", "TMPDIR",        "TMP",
      "TEMP",           "LANG",           "LANGUAGE",      "LC_ALL",
      "LC_CTYPE",       "LC_NUMERIC",     "LC_TIME",       "LC_COLLATE",
      "LC_MONETARY",    "LC_MESSAGES",    "LC_PAPER",      "LC_NAME",
      "LC_ADDRESS",     "LC_TELEPHONE",   "LC_MEASUREMENT", "LC_IDENTIFICATION",
      "NEEDLE_THREADS"};
#endif
  for (const char *name : names) {
    const std::size_t length = std::strlen(name);
    if (std::strncmp(entry, name, length) == 0 && entry[length] == '=') return true;
  }
  return false;
}

std::array<char *, 24> clean_environment() {
  static char python_no_user_site[] = "PYTHONNOUSERSITE=1";
  static char python_no_bytecode[] = "PYTHONDONTWRITEBYTECODE=1";
  std::array<char *, 24> result{};
  std::size_t count = 0u;
  for (char **entry = ::environ; *entry != nullptr; ++entry) {
    if (allowed_environment_name(*entry)) {
      if (count + 3u >= result.size()) fail("too many allowlisted environment entries");
      result[count++] = *entry;
    }
  }
  result[count++] = python_no_user_site;
  result[count++] = python_no_bytecode;
  result[count] = nullptr;
  return result;
}

#endif

} // namespace

int main(int argc, char **argv) {
#if !defined(__linux__)
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "error: authenticated Needle Python exec is supported only on Linux\n");
  return 126;
#else
  if (argc < 5 || std::strcmp(argv[3], "--") != 0) {
    fail("usage: needle_authenticated_exec SOURCE EXPECTED_SHA256 -- ARG...");
  }
  const std::array<unsigned char, 32> expected = parse_digest(argv[2]);
  const int executable = copy_to_sealed_memfd(argv[1]);
  const std::array<unsigned char, 32> actual = hash_fd(executable);
  if (!equal_digest(actual, expected)) fail("configured Needle Python SHA-256 mismatch");
#ifdef EMEL_AUTHENTICATED_EXEC_TESTING
  run_test_barrier();
#endif
  if (::lseek(executable, 0, SEEK_SET) < 0) fail_errno("cannot rewind sealed interpreter memfd");

  std::array<char *, 24> environment = clean_environment();
  ::syscall(SYS_execveat, executable, "", argv + 4, environment.data(),
            AT_EMPTY_PATH);
  fail_errno("execveat(AT_EMPTY_PATH) of sealed interpreter failed");
#endif
}
