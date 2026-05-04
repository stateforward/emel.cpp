#include "emel/io/mmap/actions.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace emel::io::mmap::action {

namespace {

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

bool platform_map(intptr_t os_resource, uint64_t file_offset,
                  uint64_t byte_size, void *&base_out,
                  uint64_t &mapped_bytes_out) noexcept {
#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(os_resource);
  const uint64_t total = file_offset + byte_size;
  HANDLE mapping =
      ::CreateFileMappingA(file_handle, nullptr, PAGE_READONLY,
                           static_cast<DWORD>((total >> 32) & 0xFFFFFFFFu),
                           static_cast<DWORD>(total & 0xFFFFFFFFu), nullptr);
  if (mapping == nullptr) {
    base_out = nullptr;
    mapped_bytes_out = 0u;
    return false;
  }
  void *view =
      ::MapViewOfFile(mapping, FILE_MAP_READ,
                      static_cast<DWORD>((file_offset >> 32) & 0xFFFFFFFFu),
                      static_cast<DWORD>(file_offset & 0xFFFFFFFFu),
                      static_cast<SIZE_T>(byte_size));
  ::CloseHandle(mapping);
  if (view == nullptr) {
    base_out = nullptr;
    mapped_bytes_out = 0u;
    return false;
  }
  base_out = view;
  mapped_bytes_out = byte_size;
  return true;
#else
  const int fd = static_cast<int>(os_resource);
  void *addr = ::mmap(nullptr, static_cast<size_t>(byte_size), PROT_READ,
                      MAP_PRIVATE, fd, static_cast<off_t>(file_offset));
  if (addr == MAP_FAILED) {
    base_out = nullptr;
    mapped_bytes_out = 0u;
    return false;
  }
  base_out = addr;
  mapped_bytes_out = byte_size;
  return true;
#endif
}

bool platform_unmap(intptr_t os_resource, void *base,
                    uint64_t mapped_bytes) noexcept {
#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(os_resource);
  const BOOL view_ok = ::UnmapViewOfFile(base);
  const BOOL handle_ok = ::CloseHandle(file_handle);
  (void)mapped_bytes;
  return view_ok != 0 && handle_ok != 0;
#else
  const int munmap_rc = ::munmap(base, static_cast<size_t>(mapped_bytes));
  const int close_rc = ::close(static_cast<int>(os_resource));
  return munmap_rc == 0 && close_rc == 0;
#endif
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

void effect_reserve_top_free_slot_then_attempt_open::operator()(
    const detail::map_tensor_runtime &ev, context &ctx) const noexcept {
  ctx.free_count -= 1u;
  const uint32_t slot_index = ctx.free_stack[ctx.free_count];
  ctx.slots[slot_index].in_use = true;
  ctx.slots[slot_index].tensor_id = -1;
  ev.status.reserved_slot = slot_index;

  intptr_t os_resource = -1;
  const bool open_ok = platform_open(ev.request.request.file_path, os_resource);
  ev.status.os_resource = os_resource;
  ev.status.file_open_ok = open_ok;
}

void effect_attempt_mapping::operator()(const detail::map_tensor_runtime &ev,
                                        context &) const noexcept {
  void *base = nullptr;
  uint64_t mapped_bytes = 0u;
  const bool mapping_ok =
      platform_map(ev.status.os_resource, ev.request.request.file_offset,
                   ev.request.request.byte_size, base, mapped_bytes);
  ev.status.mapped_base = base;
  ev.status.mapped_bytes = mapped_bytes;
  ev.status.mapping_ok = mapping_ok;
}

void effect_close_open_resource_and_release_slot_on_mapping_failure::operator()(
    const detail::map_tensor_runtime &ev, context &ctx) const noexcept {
  platform_close(ev.status.os_resource);
  auto &slot_ref = ctx.slots[ev.status.reserved_slot];
  slot_ref.in_use = false;
  slot_ref.tensor_id = -1;
  slot_ref.base = nullptr;
  slot_ref.mapped_bytes = 0u;
  slot_ref.os_resource = -1;
  slot_ref.file_offset = 0u;
  slot_ref.requested_bytes = 0u;
  ctx.free_stack[ctx.free_count] = ev.status.reserved_slot;
  ctx.free_count += 1u;
  ev.status.err = emel::error::cast(error::mapping_failed);
  ev.status.ok = false;
}

void effect_attempt_unmap::operator()(const detail::release_mapping_runtime &ev,
                                      context &ctx) const noexcept {
  const auto &slot_ref = ctx.slots[ev.status.target_slot];
  ev.status.unmap_base = slot_ref.base;
  ev.status.unmap_bytes = slot_ref.mapped_bytes;
  ev.status.os_resource = slot_ref.os_resource;
  const bool unmap_ok = platform_unmap(
      ev.status.os_resource, ev.status.unmap_base, ev.status.unmap_bytes);
  ev.status.unmap_ok = unmap_ok;
}

} // namespace emel::io::mmap::action
