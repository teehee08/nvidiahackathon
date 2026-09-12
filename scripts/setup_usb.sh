#!/usr/bin/env bash
# setup_usb.sh — Run this once at the venue, with the USB drive mounted, to
# stand up the fully offline environment on the Dell Pro Max GB10.
#
# Usage: ./scripts/setup_usb.sh /path/to/mounted/usb

set -euo pipefail

USB_ROOT="${1:?Usage: $0 /path/to/mounted/usb}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== TalentForge AI offline setup =="
echo "USB root: $USB_ROOT"
echo "Repo root: $REPO_ROOT"

echo "--> Installing Ollama (arm64) if not already present"
if ! command -v ollama &>/dev/null; then
    if [ -f "$USB_ROOT/bin/ollama-linux-arm64.tar.zst" ]; then
        sudo tar --zstd -C /usr -xf "$USB_ROOT/bin/ollama-linux-arm64.tar.zst"
    elif [ -f "$USB_ROOT/bin/ollama-linux-arm64.tgz" ]; then
        sudo tar -C /usr -xzf "$USB_ROOT/bin/ollama-linux-arm64.tgz"
    else
        echo "ERROR: Ollama ARM64 archive not found in $USB_ROOT/bin" >&2
        exit 1
    fi
fi

echo "--> Installing tectonic (arm64) if not already present"
if ! command -v tectonic &>/dev/null; then
    mkdir -p /tmp/tectonic_extract
    tar -C /tmp/tectonic_extract -xzf "$USB_ROOT/bin/tectonic-aarch64.tar.gz"
    sudo mv /tmp/tectonic_extract/tectonic /usr/local/bin/tectonic
fi

echo "--> Restoring warmed Tectonic package cache (avoids first-compile network hit)"
mkdir -p "$HOME/.cache/Tectonic"
if [ -d "$USB_ROOT/tectonic_cache" ]; then
    cp -r "$USB_ROOT/tectonic_cache/." "$HOME/.cache/Tectonic/"
fi

echo "--> Installing poppler-utils (arm64) if not already present"
if ! command -v pdftotext &>/dev/null; then
    sudo dpkg -i "$USB_ROOT"/bin/poppler-utils*.deb || sudo apt-get install -f -y
fi

echo "--> Copying model weights into models/"
mkdir -p "$REPO_ROOT/models/llm" "$REPO_ROOT/models/embeddings"
cp -r "$USB_ROOT/models/llm/." "$REPO_ROOT/models/llm/"
cp -r "$USB_ROOT/models/embeddings/." "$REPO_ROOT/models/embeddings/"

echo "--> Installing Python deps fully offline"
pip install --no-index --find-links="$USB_ROOT/wheels" -r "$REPO_ROOT/requirements.txt"

echo "--> Starting Ollama server"
(ollama serve &>/tmp/ollama.log &)
sleep 3

echo "--> Registering local model with Ollama"
if [ -f "$REPO_ROOT/models/llm/qwen2.5-14b/Modelfile" ]; then
    ollama create talentforge-qwen -f "$REPO_ROOT/models/llm/qwen2.5-14b/Modelfile"
else
    echo "WARNING: Modelfile not found — copy models/Modelfile.template into"
    echo "models/llm/qwen2.5-14b/Modelfile and adjust the FROM path first."
fi

echo "== Setup complete. Run ./scripts/verify_pipeline.sh to smoke-test. =="
