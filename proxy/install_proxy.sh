#!/usr/bin/env bash
# Usage: sudo proxy/install_proxy.sh [port] [model]      (defaults: 8765 claude-opus-5-5)
#   REPLACE_KEY=1 asks for a new key even if one is installed.
# Creates system user llmkey, copies llmproxy.py to /opt/llmproxy (root-owned), writes /etc/llmproxy/config.json
# and the key file (llmkey, mode 400), installs and starts llmproxy.service. Read this file and llmproxy.py first.
set -euo pipefail
if [ "$(id -u)" != 0 ]; then
    echo "run with sudo" >&2
    exit 1
fi
PORT=${1:-8765}
MODEL=${2:-claude-opus-5-5}
HERE=$(cd "$(dirname "$0")" && pwd)

if ! id llmkey >/dev/null 2>&1; then
    useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin llmkey
fi
install -d -o root -g root -m 755 /opt/llmproxy
install -o root -g root -m 644 "$HERE/llmproxy.py" /opt/llmproxy/llmproxy.py
install -d -o root -g llmkey -m 750 /etc/llmproxy
printf '{"port": %d, "models": ["%s"], "betas": ["server-side-fallback-2026-07-01"], "timeout_s": 900}\n' \
    "$PORT" "$MODEL" > /etc/llmproxy/config.json
chown root:llmkey /etc/llmproxy/config.json
chmod 640 /etc/llmproxy/config.json

if [ ! -s /etc/llmproxy/key ] || [ "${REPLACE_KEY:-0}" = 1 ]; then
    read -rs -p "Anthropic API key (input hidden): " KEY
    echo
    if [ -z "$KEY" ]; then
        echo "empty key" >&2
        exit 1
    fi
    (umask 077 && printf '%s' "$KEY" > /etc/llmproxy/key)
    unset KEY
    chown llmkey:llmkey /etc/llmproxy/key
    chmod 400 /etc/llmproxy/key
fi

install -o root -g root -m 644 "$HERE/llmproxy.service" /etc/systemd/system/llmproxy.service
systemctl daemon-reload
systemctl enable llmproxy.service
systemctl restart llmproxy.service
sleep 1
systemctl --no-pager --lines=3 status llmproxy.service
