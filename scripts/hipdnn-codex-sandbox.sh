#!/bin/bash
# Sandboxed Codex runner for hipDNNEP implementer work using bwrap.
#
# This is the Codex counterpart to scripts/hipdnn-sandbox.sh. It keeps the
# Claude workflow intact and provides an additive Codex-specific entrypoint.
#
# Restricts the agent to read-write access on a bead worktree and build
# directory while providing read-only access to the main checkout, SDKs,
# and system tools. Codex runs with
# --dangerously-bypass-approvals-and-sandbox inside the sandbox, so the outer
# bwrap filesystem restrictions are the effective permission model.
#
# SETUP (one-time, requires sudo):
#   sudo ./setup-bwrap-apparmor.sh
#
# Usage:
#   hipdnn-codex-sandbox <bead-id> [-- codex-args...]
#
# Examples:
#   hipdnn-codex-sandbox bd-ffh -- "Implement bd-ffh"
#   hipdnn-codex-sandbox bd-k85 -- --resume
#
# The bead-id determines which worktree gets read-write access:
#   /home/mahesh/onnxruntime/hipDNNEP-<bead-id>/

set -euo pipefail

BEAD_ID=""
CODEX_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      CODEX_ARGS=("$@")
      break
      ;;
    *)
      if [[ -z "$BEAD_ID" ]]; then
        BEAD_ID="$1"
      else
        echo "Error: unexpected argument '$1'" >&2
        echo "Usage: hipdnn-codex-sandbox <bead-id> [-- codex-args...]" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$BEAD_ID" ]]; then
  echo "Error: bead-id is required." >&2
  echo "Usage: hipdnn-codex-sandbox <bead-id> [-- codex-args...]" >&2
  exit 1
fi

MAIN_CHECKOUT="/home/mahesh/onnxruntime/hipDNNEP"
WORKTREE="/home/mahesh/onnxruntime/hipDNNEP-${BEAD_ID}"
BUILD_BASE="/home/mahesh/onnxruntime/build/hipDNNEP"
TORCH_MLIR_INSTALL="/home/mahesh/onnxruntime/build/torch-mlir/install"
VENV="${MAIN_CHECKOUT}/.venv"

THEROCK="${THEROCK_DIST:-/home/mahesh/TheRock/build/MaheshRelWithDebInfo/dist/rocm}"
ORT_ROOT="${ONNXRUNTIME_ROOT:-/home/mahesh/onnxruntime/onnxruntime}"

CODEX_BIN="$(command -v codex 2>/dev/null || echo "$HOME/.nvm/versions/node/v25.0.0/bin/codex")"
NVM_NODE_DIR="$HOME/.nvm/versions/node/v25.0.0"

if ! /usr/bin/bwrap --ro-bind / / -- true 2>/dev/null; then
  echo "Error: bwrap cannot create user namespaces." >&2
  echo "Run the one-time AppArmor setup:" >&2
  echo "  sudo ${MAIN_CHECKOUT}/scripts/setup-bwrap-apparmor.sh" >&2
  exit 1
fi

if [[ ! -x "$CODEX_BIN" ]]; then
  echo "Error: codex binary not found at ${CODEX_BIN}" >&2
  exit 1
fi

if [[ ! -d "$MAIN_CHECKOUT" ]]; then
  echo "Error: main checkout not found at ${MAIN_CHECKOUT}" >&2
  exit 1
fi

EXTRA_BINDS=()

mkdir -p "$WORKTREE"
EXTRA_BINDS+=(--bind "$WORKTREE" "$WORKTREE")

mkdir -p "$MAIN_CHECKOUT/.git/worktrees"

mkdir -p "$BUILD_BASE"
EXTRA_BINDS+=(--bind "$BUILD_BASE" "$BUILD_BASE")

if [[ -d "$TORCH_MLIR_INSTALL" ]]; then
  EXTRA_BINDS+=(--ro-bind "$TORCH_MLIR_INSTALL" "$TORCH_MLIR_INSTALL")
fi

if [[ -d "$THEROCK" ]]; then
  THEROCK_BUILD_ROOT="$(cd "$THEROCK/../.." && pwd)"
  EXTRA_BINDS+=(--ro-bind "$THEROCK_BUILD_ROOT" "$THEROCK_BUILD_ROOT")
fi

THEROCK_SRC="/home/mahesh/TheRock/TheRock"
if [[ -d "$THEROCK_SRC" ]]; then
  EXTRA_BINDS+=(--ro-bind "$THEROCK_SRC" "$THEROCK_SRC")
fi

if [[ -d "$ORT_ROOT" ]]; then
  EXTRA_BINDS+=(--ro-bind "$ORT_ROOT" "$ORT_ROOT")
fi

if [[ -d "$VENV" ]]; then
  EXTRA_BINDS+=(--ro-bind "$VENV" "$VENV")
fi

if [[ -d "$HOME/.local/state/codex" ]]; then
  EXTRA_BINDS+=(--bind "$HOME/.local/state/codex" "$HOME/.local/state/codex")
fi

if [[ -d "/home/mahesh/iree/build/RelWithDebInfo/tools" ]]; then
  EXTRA_BINDS+=(--ro-bind "/home/mahesh/iree/build/RelWithDebInfo/tools" \
                          "/home/mahesh/iree/build/RelWithDebInfo/tools")
fi

echo "Sandboxed codex for bead ${BEAD_ID}"
echo "  Main checkout: ${MAIN_CHECKOUT} (read-only)"
echo "  Worktree:      ${WORKTREE} (read-write)"
echo "  Build dir:     ${BUILD_BASE} (read-write)"

exec /usr/bin/bwrap \
  --ro-bind /usr /usr \
  --symlink usr/bin /bin \
  --symlink usr/lib /lib \
  --symlink usr/lib64 /lib64 \
  --ro-bind /etc /etc \
  --ro-bind /opt /opt \
  --proc /proc \
  --ro-bind /sys /sys \
  --dev /dev \
  --dev-bind-try /dev/dri /dev/dri \
  --dev-bind-try /dev/kfd /dev/kfd \
  --bind /tmp /tmp \
  --ro-bind /run/systemd/resolve /run/systemd/resolve \
  \
  --tmpfs "$HOME" \
  --ro-bind "$HOME/.bashrc" "$HOME/.bashrc" \
  --ro-bind "$HOME/.gitconfig" "$HOME/.gitconfig" \
  --ro-bind "$HOME/.ssh" "$HOME/.ssh" \
  --ro-bind "$HOME/.local" "$HOME/.local" \
  --ro-bind "$HOME/.nvm" "$HOME/.nvm" \
  \
  --bind "$HOME/.codex" "$HOME/.codex" \
  --bind-try "$HOME/.config/codex" "$HOME/.config/codex" \
  --bind "$HOME/.config/gh" "$HOME/.config/gh" \
  --bind "$HOME/.cache" "$HOME/.cache" \
  \
  --ro-bind "$MAIN_CHECKOUT" "$MAIN_CHECKOUT" \
  --bind "$MAIN_CHECKOUT/.beads" "$MAIN_CHECKOUT/.beads" \
  --bind "$MAIN_CHECKOUT/.git" "$MAIN_CHECKOUT/.git" \
  \
  "${EXTRA_BINDS[@]}" \
  \
  --setenv HOME "$HOME" \
  --setenv THEROCK_DIST "$THEROCK" \
  --setenv ONNXRUNTIME_ROOT "$ORT_ROOT" \
  --setenv PATH "${VENV}/bin:$HOME/.local/bin:${NVM_NODE_DIR}/bin:/usr/local/bin:/usr/bin:/bin" \
  --setenv OPENAI_API_KEY "${OPENAI_API_KEY:-}" \
  --unsetenv VSCODE_GIT_ASKPASS_MAIN \
  --unsetenv VSCODE_GIT_ASKPASS_NODE \
  \
  --chdir "$MAIN_CHECKOUT" \
  --die-with-parent \
  -- \
  "$CODEX_BIN" --dangerously-bypass-approvals-and-sandbox "${CODEX_ARGS[@]}"
