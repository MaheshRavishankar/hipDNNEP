#!/bin/bash
# One-time AppArmor setup to allow bubblewrap (bwrap) to create user namespaces.
#
# Ubuntu 24.04+ restricts unprivileged user namespaces via AppArmor.
# This script creates a permissive profile for bwrap so it can create
# mount/user namespaces without root.
#
# Usage (requires sudo):
#   sudo ./setup-bwrap-apparmor.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Error: this script must be run as root (sudo)." >&2
  exit 1
fi

PROFILE_DIR="/etc/apparmor.d"
PROFILE_PATH="${PROFILE_DIR}/bwrap"

if [[ -f "$PROFILE_PATH" ]]; then
  echo "AppArmor profile for bwrap already exists at ${PROFILE_PATH}."
  echo "To reinstall, remove it first: sudo rm ${PROFILE_PATH}"
  exit 0
fi

echo "Creating AppArmor profile for bwrap..."

cat > "$PROFILE_PATH" << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,

  include if exists <local/bwrap>
}
EOF

echo "Reloading AppArmor..."
apparmor_parser -r "$PROFILE_PATH"

echo "Done. Verifying..."
if su -c '/usr/bin/bwrap --ro-bind / / -- true' "${SUDO_USER:-$(logname)}" 2>/dev/null; then
  echo "bwrap user namespaces are working."
else
  echo "Warning: bwrap test failed. You may need to reboot or check dmesg." >&2
  exit 1
fi
