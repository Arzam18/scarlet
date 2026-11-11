#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TARGET="aarch64-linux-android"
API="${ANDROID_API_LEVEL:-24}"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [[ -z "$NDK" || ! -d "$NDK/toolchains/llvm/prebuilt/linux-x86_64" ]]; then
  echo "ERROR: set ANDROID_NDK_HOME to an installed Linux Android NDK" >&2
  exit 1
fi
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
LINKER="$TOOLCHAIN/aarch64-linux-android${API}-clang"
if [[ ! -x "$LINKER" ]]; then
  echo "ERROR: missing Android linker: $LINKER" >&2
  exit 1
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$LINKER"
export CC_aarch64_linux_android="$LINKER"
export AR_aarch64_linux_android="$TOOLCHAIN/llvm-ar"
export RUSTFLAGS="-C target-cpu=generic -C target-feature=+neon"

rustup target add "$TARGET"
# Compile AArch64-only unit tests as well, even though Android test binaries
# cannot be executed on the GitHub host without a device/emulator.
cargo test --locked --release --target "$TARGET" --lib --no-run
cargo build --locked --release --target "$TARGET" --bin scarlet

BIN="$ROOT/target/$TARGET/release/scarlet"
if [[ ! -s "$BIN" ]]; then
  echo "ERROR: Android engine binary was not created" >&2
  exit 1
fi
"$TOOLCHAIN/llvm-readelf" -h "$BIN" | grep -q 'Machine:.*AArch64' || {
  echo "ERROR: output is not an AArch64 ELF" >&2; exit 1;
}
"$TOOLCHAIN/llvm-objdump" -d "$BIN" > "$ROOT/target/scarlet-android-arm64.disasm"
if ! grep -Eq '\b(add|sub|mul|smull|addv)\s+v[0-9]+' "$ROOT/target/scarlet-android-arm64.disasm"; then
  echo "ERROR: could not confirm NEON vector instructions in the binary" >&2
  exit 1
fi
if ! grep -Eq '\bsdot\b' "$ROOT/target/scarlet-android-arm64.disasm"; then
  echo "ERROR: runtime-dispatched Dotprod kernel did not emit SDOT" >&2
  exit 1
fi

mkdir -p "$ROOT/dist"
cp "$BIN" "$ROOT/dist/scarlet-android-arm64"
file "$ROOT/dist/scarlet-android-arm64"
echo "PASS: Android ARM64 binary created; NEON confirmed; SDOT Dotprod kernel present behind runtime feature and network-range checks."
echo "Output: $ROOT/dist/scarlet-android-arm64"
