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

uint64_t platform_required_offset_alignment() noexcept {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  ::GetSystemInfo(&info);
  return info.dwAllocationGranularity != 0u
             ? static_cast<uint64_t>(info.dwAllocationGranularity)
             : k_required_offset_alignment;
#else
  const long page_size = ::sysconf(_SC_PAGE_SIZE);
  return page_size > 0 ? static_cast<uint64_t>(page_size)
                       : k_required_offset_alignment;
#endif
}

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

bool platform_file_size(intptr_t os_resource,
                        uint64_t &file_size_bytes_out) noexcept {
#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(os_resource);
  LARGE_INTEGER size{};
  if (::GetFileSizeEx(file_handle, &size) == 0 || size.QuadPart < 0) {
    file_size_bytes_out = 0u;
    return false;
  }
  file_size_bytes_out = static_cast<uint64_t>(size.QuadPart);
  return true;
#else
  struct stat st{};
  if (::fstat(static_cast<int>(os_resource), &st) != 0 || st.st_size < 0) {
    file_size_bytes_out = 0u;
    return false;
  }
  file_size_bytes_out = static_cast<uint64_t>(st.st_size);
  return true;
#endif
}

unmap_result platform_unmap(intptr_t os_resource, void *base,
                                     uint64_t mapped_bytes) noexcept {
  unmap_result result{};
  const bool mapping_absent = base == nullptr || mapped_bytes == 0u;
  const bool resource_absent = os_resource == -1;
#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(os_resource);
  result.unmap_base_released = mapping_absent || ::UnmapViewOfFile(base) != 0;
  result.os_resource_released =
      resource_absent || ::CloseHandle(file_handle) != 0;
#else
  result.unmap_base_released =
      mapping_absent || ::munmap(base, static_cast<size_t>(mapped_bytes)) == 0;
  result.os_resource_released =
      resource_absent || ::close(static_cast<int>(os_resource)) == 0;
#endif
  result.ok = result.unmap_base_released && result.os_resource_released;
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

// One platform helper per advice kind: the kind is routed by explicit guards
// in the transition table, never branched on here.
bool platform_advise_sequential(void *base, uint64_t offset,
                                uint64_t length) noexcept {
#if defined(_WIN32)
  // Win32 has no sequential-readahead hint for a private read-only view; the
  // hint degrades to a successful no-op.
  (void)base;
  (void)offset;
  (void)length;
  return true;
#else
  unsigned char *window = static_cast<unsigned char *>(base) + offset;
  return ::posix_madvise(window, static_cast<size_t>(length),
                         POSIX_MADV_SEQUENTIAL) == 0;
#endif
}

bool platform_advise_willneed(void *base, uint64_t offset,
                              uint64_t length) noexcept {
  unsigned char *window = static_cast<unsigned char *>(base) + offset;
#if defined(_WIN32)
  WIN32_MEMORY_RANGE_ENTRY range{};
  range.VirtualAddress = window;
  range.NumberOfBytes = static_cast<SIZE_T>(length);
  return ::PrefetchVirtualMemory(::GetCurrentProcess(), 1u, &range, 0) != 0;
#else
  return ::posix_madvise(window, static_cast<size_t>(length),
                         POSIX_MADV_WILLNEED) == 0;
#endif
}

bool platform_advise_dontneed(void *base, uint64_t offset,
                              uint64_t length) noexcept {
#if defined(_WIN32)
  // Win32 reclaims pages of a private view under memory pressure on its own;
  // the hint degrades to a successful no-op.
  (void)base;
  (void)offset;
  (void)length;
  return true;
#else
  unsigned char *window = static_cast<unsigned char *>(base) + offset;
  return ::posix_madvise(window, static_cast<size_t>(length),
                         POSIX_MADV_DONTNEED) == 0;
#endif
}

} // namespace

platform_ops default_platform_ops() noexcept {
  platform_ops ops{};
  ops.open = &platform_open;
  ops.file_size = &platform_file_size;
  ops.map = &platform_map;
  ops.unmap = &platform_unmap;
  ops.close = &platform_close;
  ops.advise_sequential = &platform_advise_sequential;
  ops.advise_willneed = &platform_advise_willneed;
  ops.advise_dontneed = &platform_advise_dontneed;
  return ops;
}

context::context() noexcept : context(default_platform_ops()) {}

context::context(const platform_ops &platform_in) noexcept
    : required_offset_alignment(platform_required_offset_alignment()),
      platform(platform_in) {
  // An injector overriding a single op keeps the production behavior for the
  // rest: null fields seed from the defaults so no effect ever calls through
  // a null platform pointer.
  const platform_ops defaults = default_platform_ops();
  if (platform.open == nullptr) {
    platform.open = defaults.open;
  }
  if (platform.file_size == nullptr) {
    platform.file_size = defaults.file_size;
  }
  if (platform.map == nullptr) {
    platform.map = defaults.map;
  }
  if (platform.unmap == nullptr) {
    platform.unmap = defaults.unmap;
  }
  if (platform.close == nullptr) {
    platform.close = defaults.close;
  }
  if (platform.advise_sequential == nullptr) {
    platform.advise_sequential = defaults.advise_sequential;
  }
  if (platform.advise_willneed == nullptr) {
    platform.advise_willneed = defaults.advise_willneed;
  }
  if (platform.advise_dontneed == nullptr) {
    platform.advise_dontneed = defaults.advise_dontneed;
  }
  for (uint32_t i = 0; i < k_max_mappings; ++i) {
    free_stack[i] = (k_max_mappings - 1u) - i;
  }
  free_count = k_max_mappings;
}

context::~context() noexcept {
  for (auto &slot_ref : slots) {
    if (!slot_ref.in_use) {
      continue;
    }

    (void)platform.unmap(slot_ref.os_resource, slot_ref.base,
                         slot_ref.mapped_bytes);

    slot_ref.in_use = false;
    slot_ref.tensor_id = -1;
    slot_ref.base = nullptr;
    slot_ref.mapped_bytes = 0u;
    slot_ref.os_resource = -1;
    slot_ref.file_offset = 0u;
    slot_ref.requested_bytes = 0u;
  }
}

void effect_reserve_top_free_slot_then_attempt_open::operator()(
    const detail::map_tensor_runtime &ev, context &ctx) const noexcept {
  ctx.free_count -= 1u;
  const uint32_t slot_index = ctx.free_stack[ctx.free_count];
  ctx.slots[slot_index].in_use = true;
  ctx.slots[slot_index].tensor_id = -1;
  ev.status.reserved_slot = slot_index;

  intptr_t os_resource = -1;
  const bool open_ok =
      ctx.platform.open(ev.request.request.file_path, os_resource);
  ev.status.os_resource = os_resource;
  ev.status.file_open_ok = open_ok;
}

void effect_measure_open_file_size::operator()(
    const detail::map_tensor_runtime &ev, context &ctx) const noexcept {
  uint64_t file_size_bytes = 0u;
  const bool size_ok =
      ctx.platform.file_size(ev.status.os_resource, file_size_bytes);
  ev.status.file_size_bytes = file_size_bytes;
  ev.status.file_size_ok = size_ok;
}

void effect_attempt_mapping::operator()(const detail::map_tensor_runtime &ev,
                                        context &ctx) const noexcept {
  void *base = nullptr;
  uint64_t mapped_bytes = 0u;
  const bool mapping_ok =
      ctx.platform.map(ev.status.os_resource, ev.request.request.file_offset,
                       ev.request.request.byte_size, base, mapped_bytes);
  ev.status.mapped_base = base;
  ev.status.mapped_bytes = mapped_bytes;
  ev.status.mapping_ok = mapping_ok;
}

void effect_close_open_resource_and_release_slot_on_file_span_failure::
operator()(const detail::map_tensor_runtime &ev, context &ctx) const noexcept {
  ctx.platform.close(ev.status.os_resource);
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
  ev.status.err = emel::error::cast(error::unsupported_resource);
  ev.status.ok = false;
}

void effect_close_open_resource_and_release_slot_on_mapping_failure::operator()(
    const detail::map_tensor_runtime &ev, context &ctx) const noexcept {
  ctx.platform.close(ev.status.os_resource);
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
  const unmap_result result = ctx.platform.unmap(
      ev.status.os_resource, ev.status.unmap_base, ev.status.unmap_bytes);
  ev.status.unmap_ok = result.ok;
  ev.status.unmap_base_released = result.unmap_base_released;
  ev.status.os_resource_released = result.os_resource_released;
}

void effect_mark_unmap_failed_and_release_slot::operator()(
    const detail::release_mapping_runtime &ev, context &ctx) const noexcept {
  auto &slot_ref = ctx.slots[ev.status.target_slot];
  const std::uintptr_t keep_mapping =
      static_cast<std::uintptr_t>(!ev.status.unmap_base_released);
  const std::uintptr_t mapping_mask = 0u - keep_mapping;
  slot_ref.base = reinterpret_cast<void *>(
      reinterpret_cast<std::uintptr_t>(slot_ref.base) & mapping_mask);
  slot_ref.mapped_bytes *= static_cast<uint64_t>(keep_mapping);
  const intptr_t keep_resource =
      static_cast<intptr_t>(!ev.status.os_resource_released);
  slot_ref.os_resource = (slot_ref.os_resource * keep_resource) -
                         static_cast<intptr_t>(ev.status.os_resource_released);
  ev.status.err = emel::error::cast(error::unmap_failed);
  ev.status.ok = false;
}

void effect_attempt_advise_sequential::operator()(
    const detail::advise_mapping_runtime &ev, context &ctx) const noexcept {
  const slot &slot_ref = ctx.slots[ev.request.handle];
  ev.status.advise_ok = ctx.platform.advise_sequential(
      slot_ref.base, ev.request.offset, ev.request.length);
}

void effect_attempt_advise_willneed::operator()(
    const detail::advise_mapping_runtime &ev, context &ctx) const noexcept {
  const slot &slot_ref = ctx.slots[ev.request.handle];
  ev.status.advise_ok = ctx.platform.advise_willneed(
      slot_ref.base, ev.request.offset, ev.request.length);
}

void effect_attempt_advise_dontneed::operator()(
    const detail::advise_mapping_runtime &ev, context &ctx) const noexcept {
  const slot &slot_ref = ctx.slots[ev.request.handle];
  ev.status.advise_ok = ctx.platform.advise_dontneed(
      slot_ref.base, ev.request.offset, ev.request.length);
}

} // namespace emel::io::mmap::action
