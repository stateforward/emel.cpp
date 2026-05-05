#include "emel/io/read/actions.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace emel::io::read::action {

namespace {

struct platform_read_result {
  bool ok = false;
  uint64_t bytes_copied = 0u;
};

bool platform_open(std::string_view path, intptr_t &os_resource_out) noexcept {
  std::array<char, k_max_file_path_bytes + 1u> path_buffer{};
  for (std::size_t i = 0; i < path.size(); ++i) {
    path_buffer[i] = path[i];
  }
  path_buffer[path.size()] = '\0';

#if defined(_WIN32)
  HANDLE handle =
      ::CreateFileA(path_buffer.data(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    os_resource_out = -1;
    return false;
  }
  os_resource_out = reinterpret_cast<intptr_t>(handle);
  return true;
#else
  const int fd = ::open(path_buffer.data(), O_RDONLY);
  if (fd < 0) {
    os_resource_out = -1;
    return false;
  }
  os_resource_out = static_cast<intptr_t>(fd);
  return true;
#endif
}

bool platform_seek(intptr_t os_resource, uint64_t file_offset) noexcept {
#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(os_resource);
  LARGE_INTEGER distance{};
  distance.QuadPart = static_cast<LONGLONG>(file_offset);
  return ::SetFilePointerEx(file_handle, distance, nullptr, FILE_BEGIN) != 0;
#else
  return ::lseek(static_cast<int>(os_resource), static_cast<off_t>(file_offset),
                 SEEK_SET) >= 0;
#endif
}

platform_read_result platform_read_exact(intptr_t os_resource, void *target,
                                         uint64_t byte_size) noexcept {
  platform_read_result result{.ok = true, .bytes_copied = 0u};
  auto *cursor = static_cast<unsigned char *>(target);
  uint64_t remaining = byte_size;

  while (remaining > 0u && result.ok) {
#if defined(_WIN32)
    DWORD chunk = remaining > static_cast<uint64_t>(0xFFFFFFFFu)
                     ? static_cast<DWORD>(0xFFFFFFFFu)
                     : static_cast<DWORD>(remaining);
    DWORD bytes_read = 0u;
    const BOOL read_ok =
        ::ReadFile(reinterpret_cast<HANDLE>(os_resource), cursor, chunk,
                   &bytes_read, nullptr);
    result.ok = read_ok != 0;
    cursor += bytes_read;
    result.bytes_copied += static_cast<uint64_t>(bytes_read);
    remaining -= static_cast<uint64_t>(bytes_read);
    result.ok = result.ok && bytes_read != 0u;
#else
    const auto read_result =
        ::read(static_cast<int>(os_resource), cursor, static_cast<size_t>(remaining));
    result.ok = read_result >= 0;
    const uint64_t bytes_read =
        read_result > 0 ? static_cast<uint64_t>(read_result) : 0u;
    cursor += bytes_read;
    result.bytes_copied += bytes_read;
    remaining -= bytes_read;
    result.ok = result.ok && bytes_read != 0u;
#endif
  }

  result.ok = result.ok || result.bytes_copied > 0u;
  return result;
}

void platform_close(intptr_t os_resource) noexcept {
#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(os_resource);
  ::CloseHandle(file_handle);
#else
  ::close(static_cast<int>(os_resource));
#endif
}

} // namespace

void effect_attempt_open::operator()(const detail::read_tensor_runtime &ev,
                                     context &) const noexcept {
  intptr_t os_resource = -1;
  const bool open_ok = platform_open(ev.request.request.file_path, os_resource);
  ev.status.os_resource = os_resource;
  ev.status.file_open_ok = open_ok;
}

void effect_attempt_seek::operator()(const detail::read_tensor_runtime &ev,
                                     context &) const noexcept {
  ev.status.file_seek_ok =
      platform_seek(ev.status.os_resource, ev.request.request.file_offset);
}

void effect_attempt_read_and_close::operator()(
    const detail::read_tensor_runtime &ev, context &) const noexcept {
  const platform_read_result read_result =
      platform_read_exact(ev.status.os_resource, ev.request.request.target_buffer,
                          ev.request.request.byte_size);
  platform_close(ev.status.os_resource);
  ev.status.os_resource = -1;
  ev.status.bytes_copied = read_result.bytes_copied;
  ev.status.file_read_ok = read_result.ok;
}

void effect_mark_file_seek_failed_and_close::operator()(
    const detail::read_tensor_runtime &ev, context &) const noexcept {
  platform_close(ev.status.os_resource);
  ev.status.os_resource = -1;
  ev.status.err = emel::error::cast(error::file_seek_failed);
  ev.status.ok = false;
}

} // namespace emel::io::read::action
