#!/usr/bin/env bash
# Shared CMake target configuration for Zig builds.
#
# Zig 0.15 on macOS derives an exact host-version deployment target when no
# explicit compiler target is provided. That can produce binaries whose
# LC_BUILD_VERSION is newer than the active Xcode SDK and which dyld rejects
# before main. CMake's OSX deployment setting alone is insufficient because
# Zig ignores the resulting -mmacosx-version-min flag; CMAKE_*_COMPILER_TARGET
# is required so CMake passes --target to Zig.

EMEL_ZIG_CMAKE_PLATFORM_ARGS=()
EMEL_ZIG_MACOS_DEPLOYMENT_TARGET=""

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v xcrun >/dev/null 2>&1; then
    echo "error: xcrun is required to configure Zig for macOS" >&2
    return 1 2>/dev/null || exit 1
  fi

  emel_zig_macos_sysroot="${EMEL_MACOS_SYSROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
  EMEL_ZIG_MACOS_DEPLOYMENT_TARGET="${EMEL_MACOS_DEPLOYMENT_TARGET:-$(
    xcrun --sdk macosx --show-sdk-version
  )}"

  if [[ ! -d "$emel_zig_macos_sysroot" ]]; then
    echo "error: active macOS SDK does not exist: $emel_zig_macos_sysroot" >&2
    return 1 2>/dev/null || exit 1
  fi
  if [[ ! "$EMEL_ZIG_MACOS_DEPLOYMENT_TARGET" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "error: invalid macOS deployment target: $EMEL_ZIG_MACOS_DEPLOYMENT_TARGET" >&2
    return 1 2>/dev/null || exit 1
  fi

  case "$(uname -m)" in
    arm64|aarch64)
      emel_zig_macos_arch="aarch64"
      ;;
    x86_64|amd64)
      emel_zig_macos_arch="x86_64"
      ;;
    *)
      echo "error: unsupported macOS architecture for Zig: $(uname -m)" >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac

  emel_zig_macos_target="${emel_zig_macos_arch}-macos.${EMEL_ZIG_MACOS_DEPLOYMENT_TARGET}"
  emel_zig_macos_frameworks="$emel_zig_macos_sysroot/System/Library/Frameworks"

  if [[ ! -d "$emel_zig_macos_frameworks" ]]; then
    echo "error: active macOS SDK framework directory does not exist: $emel_zig_macos_frameworks" >&2
    return 1 2>/dev/null || exit 1
  fi

  # Zig's explicit Darwin target does not add the selected Xcode SDK's
  # usr/include directory, and its Clang frontend rejects an enum forward-
  # declaration extension used by those SDK headers. Keep the deployment
  # target explicit while binding the complete matching SDK header surface.
  EMEL_ZIG_CMAKE_PLATFORM_ARGS+=(
    "-DCMAKE_OSX_SYSROOT=$emel_zig_macos_sysroot"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=$EMEL_ZIG_MACOS_DEPLOYMENT_TARGET"
    "-DCMAKE_C_COMPILER_TARGET=$emel_zig_macos_target"
    "-DCMAKE_CXX_COMPILER_TARGET=$emel_zig_macos_target"
    "-DCMAKE_C_STANDARD_INCLUDE_DIRECTORIES=$emel_zig_macos_sysroot/usr/include"
    "-DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=$emel_zig_macos_sysroot/usr/include"
    "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -Wno-elaborated-enum-base"
    "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -Wno-elaborated-enum-base"
    "-DCMAKE_EXE_LINKER_FLAGS=-F$emel_zig_macos_frameworks -Wl,-dead_strip"
    "-DCMAKE_SHARED_LINKER_FLAGS=-F$emel_zig_macos_frameworks"
    "-DCMAKE_MODULE_LINKER_FLAGS=-F$emel_zig_macos_frameworks"
  )
fi
