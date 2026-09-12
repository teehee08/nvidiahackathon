# Dell Offline Setup

Use these steps after copying the contents of the local `usb` folder to a thumb drive.

## USB Layout

The root of the thumb drive should contain:

```text
bin/
models/
repo/
wheels/
tectonic_cache/
```

If you copied the entire `usb` folder instead, the path will include `/usb` at the end.

## 1. Find the USB Drive

On the Dell, open a terminal and run:

```bash
ls -la /media/$USER
```

Set `USB_ROOT` to the mounted drive name:

```bash
export USB_ROOT=/media/$USER/YOUR_USB_NAME
```

If the payload is inside a nested `usb` directory, use:

```bash
export USB_ROOT=/media/$USER/YOUR_USB_NAME/usb
```

Confirm that the payload is visible:

```bash
test -f "$USB_ROOT/bin/ollama-linux-arm64.tar.zst" && echo "USB payload found"
```

## 2. Copy and Set Up the Project

Copy the repository from the USB drive to the Dell's home directory:

```bash
cp -a "$USB_ROOT/repo/talentforge-ai" "$HOME/talentforge-ai"
cd "$HOME/talentforge-ai"
```

Run the setup script:

```bash
chmod +x scripts/*.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

./scripts/setup_usb.sh "$USB_ROOT"
```

The setup script installs the staged ARM64 Ollama, Tectonic, Poppler, and Python dependencies; copies the models; starts Ollama; and registers the `talentforge-qwen` model.

## 3. Validate the Offline Installation

Run this with Wi-Fi disabled if possible:

```bash
./scripts/verify_pipeline.sh
```

The expected final message is:

```text
== All checks passed. You're ready for eHub. ==
```

## 4. Start the Application

In the first terminal:

```bash
cd "$HOME/talentforge-ai"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd "$HOME/talentforge-ai"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

streamlit run frontend/app_streamlit.py
```

Open the local address shown by Streamlit, usually:

```text
http://localhost:8501
```

## Notes

- The Dell Pro Max with GB10 is ARM64. Use the staged ARM64 wheels and binaries from the USB drive.
- Do not install the USB wheels into an x86 Windows Python environment.
- Keep a second USB copy as a backup.
- The setup script is located at `scripts/setup_usb.sh`.
- The validation script is located at `scripts/verify_pipeline.sh`.
