#!/bin/sh
# basal-rs installer and updater (macOS on Apple Silicon, Linux x86_64 with an NVIDIA GPU):
#
#   curl -fsSL https://raw.githubusercontent.com/itsoltech/basal-rs/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/itsoltech/basal-rs/main/install.sh | sh -s -- --version 0.1.0
#
# Downloads the release package from GitHub Releases, checks its SHA-256, installs it into PREFIX (default ~/.local:
# PREFIX/bin/basal, on Linux also PREFIX/libexec/basal/basal-cuda), then runs `basal setup` (on Linux the CUDA
# libraries from NVIDIA) and `basal doctor`. Running it again updates the installation.
#
# Options: --version X.Y.Z (default: the latest release), --prefix DIR, --no-setup, --dry-run (print what would be
# done). Environment: BASAL_RELEASE_URL=URL or a local directory holding the packages and SHA256SUMS (instead of
# GitHub Releases; for testing), GITHUB_TOKEN (optional, for the GitHub API rate limit).
set -eu

REPO=itsoltech/basal-rs
PREFIX="${HOME}/.local"
VERSION=""
SETUP=1
DRY=0

say() { printf 'basal-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }
run() { if [ "$DRY" = 1 ]; then say "would run: $*"; else "$@"; fi; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="${2:?--version needs a value}"; VERSION="${VERSION#v}"; shift 2 ;;
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --no-setup) SETUP=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0" 2>/dev/null || true; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done

os=$(uname -s); arch=$(uname -m)
case "$os/$arch" in
  Darwin/arm64) target=aarch64-apple-darwin ;;
  Linux/x86_64) target=x86_64-linux ;;
  Darwin/*) die "basal-rs needs Apple Silicon (M1 or newer); this Mac is $arch" ;;
  Linux/*) die "no Linux package for $arch yet (x86_64 only); the container image ghcr.io/$REPO may work" ;;
  *) die "unsupported system $os/$arch (macOS on Apple Silicon, Linux x86_64)" ;;
esac

if command -v curl > /dev/null 2>&1; then
  fetch() { curl --fail-with-body --location --silent --show-error ${GITHUB_TOKEN:+-H "Authorization: Bearer $GITHUB_TOKEN"} -o "$2" "$1"; }
elif command -v wget > /dev/null 2>&1; then
  fetch() { wget -q ${GITHUB_TOKEN:+--header="Authorization: Bearer $GITHUB_TOKEN"} -O "$2" "$1"; }
else
  die "curl or wget is needed"
fi
if command -v sha256sum > /dev/null 2>&1; then sha() { sha256sum "$1" | cut -d' ' -f1; }
else sha() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi

tmp=$(mktemp -d)
trap 'rm -r -f "$tmp"' EXIT

# where the packages come from: a local directory, a URL, or GitHub Releases
src="${BASAL_RELEASE_URL:-}"
get() {  # get FILE DEST
  case "$src" in
    "") fetch "https://github.com/$REPO/releases/download/v$VERSION/$1" "$2" ;;
    http://*|https://*) fetch "$src/$1" "$2" ;;
    *) cp "$src/$1" "$2" ;;
  esac
}
if [ -z "$VERSION" ]; then
  if [ -n "$src" ]; then
    # the newest package in the directory listing of SHA256SUMS
    get SHA256SUMS "$tmp/SHA256SUMS" || die "no SHA256SUMS in $src"
    VERSION=$(sed -n "s/.*basal-\([0-9][0-9.]*[0-9a-z.-]*\)-$target\.tar\.gz/\1/p" "$tmp/SHA256SUMS" | sort -V | tail -1)
  else
    fetch "https://api.github.com/repos/$REPO/releases/latest" "$tmp/latest.json" || die "cannot reach the GitHub API"
    VERSION=$(sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' "$tmp/latest.json" | head -1)
  fi
  [ -n "$VERSION" ] || die "no release found"
fi

pkg="basal-$VERSION-$target.tar.gz"
if [ -x "$PREFIX/bin/basal" ]; then
  say "installed: $("$PREFIX/bin/basal" --version 2>/dev/null || echo unknown); installing $VERSION"
else
  say "installing basal $VERSION ($target) into $PREFIX"
fi
if command -v brew > /dev/null 2>&1 && brew list --formula basal-rs > /dev/null 2>&1; then
  say "note: basal-rs is also installed with Homebrew (update it with: brew upgrade basal-rs)"
fi

get "$pkg" "$tmp/$pkg" || die "cannot download $pkg"
[ -f "$tmp/SHA256SUMS" ] || get SHA256SUMS "$tmp/SHA256SUMS" || die "cannot download SHA256SUMS"
want=$(awk -v f="$pkg" '$2 == f || $2 == "*"f {print $1}' "$tmp/SHA256SUMS")
[ -n "$want" ] || die "$pkg is not listed in SHA256SUMS"
got=$(sha "$tmp/$pkg")
[ "$want" = "$got" ] || die "$pkg: SHA-256 $got, expected $want"
say "SHA-256 ok"

tar -xzf "$tmp/$pkg" -C "$tmp"
dir="$tmp/basal-$VERSION-$target"
[ -x "$dir/bin/basal" ] || die "$pkg has no bin/basal"

# install: copy next to the old files, then rename (a running server keeps its binary)
run mkdir -p "$PREFIX/bin"
run cp "$dir/bin/basal" "$PREFIX/bin/.basal.new"
run mv -f "$PREFIX/bin/.basal.new" "$PREFIX/bin/basal"
if [ -x "$dir/libexec/basal/basal-cuda" ]; then
  run mkdir -p "$PREFIX/libexec/basal"
  run cp "$dir/libexec/basal/basal-cuda" "$PREFIX/libexec/basal/.basal-cuda.new"
  run mv -f "$PREFIX/libexec/basal/.basal-cuda.new" "$PREFIX/libexec/basal/basal-cuda"
fi
run mkdir -p "$PREFIX/share/doc/basal"
run cp "$dir/LICENSE" "$dir/NOTICE" "$PREFIX/share/doc/basal/"
[ "$DRY" = 1 ] && { say "dry run: nothing installed"; exit 0; }
say "installed $("$PREFIX/bin/basal" --version)"

case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) say "add $PREFIX/bin to PATH, e.g.: echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.profile" ;;
esac

if [ "$SETUP" = 1 ]; then
  "$PREFIX/bin/basal" setup || say "basal setup failed; run it again after fixing the problem above"
  "$PREFIX/bin/basal" doctor || true
fi
say "start the server: basal serve   (http://127.0.0.1:8000)"
