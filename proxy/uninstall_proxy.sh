#!/usr/bin/env bash
# Usage: sudo proxy/uninstall_proxy.sh
# Stops and removes llmproxy.service, /opt/llmproxy, /etc/llmproxy (with the key file) and user llmkey.
set -euo pipefail
if [ "$(id -u)" != 0 ]; then
    echo "run with sudo" >&2
    exit 1
fi
systemctl disable --now llmproxy.service 2>/dev/null || true
rm -f /etc/systemd/system/llmproxy.service
systemctl daemon-reload
if [ -f /etc/llmproxy/key ]; then
    shred -u /etc/llmproxy/key
fi
rm -rf /opt/llmproxy /etc/llmproxy
if id llmkey >/dev/null 2>&1; then
    userdel llmkey
fi
echo "llmproxy removed"
