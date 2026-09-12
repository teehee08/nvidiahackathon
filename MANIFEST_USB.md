# TalentForge AI — USB Thumbdrive Pre-Download Manifest

**Read this first:** the Dell Pro Max with GB10 is an **ARM64 (aarch64)** machine
(NVIDIA Grace CPU, 20 cores) running **NVIDIA DGX OS** (Ubuntu-based), with a
Blackwell GPU (6,144 CUDA cores) and 128GB unified memory. This is *not* a
standard x86_64 desktop. Every wheel and binary you stage must be an
**aarch64/arm64 build**, or `pip`/the binary will silently fail or refuse to
install once you're offline. Do this staging on any machine with internet
access, then copy the whole USB payload over.

Target USB layout:

```
usb/
├── models/
│   ├── llm/
│   ├── embeddings/
├── wheels/                 # arch=aarch64 python wheels
├── bin/                    # arch=aarch64 binaries (tectonic, poppler)
└── repo/                   # this project, zipped or git-bundled
```

---

## 1. Primary LLM (reasoning / rewriting agent)

GB10 has 128GB unified memory shared CPU+GPU, so you have real headroom — pick
one 8B–14B model as your default, and optionally a second as a fallback/judge.

| Purpose | Model | Format | Quant | Approx. size |
|---|---|---|---|---|
| **Primary (recommended)** | `Qwen/Qwen2.5-14B-Instruct-GGUF` | GGUF | `Q5_K_M` | ~10.5 GB |
| Faster fallback | `Qwen/Qwen2.5-7B-Instruct-GGUF` | GGUF | `Q6_K` | ~6.3 GB |
| Alt / strong instruction-following | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | GGUF | `Q6_K` | ~6.6 GB |
| Alt, long-context JD ingestion | `bartowski/Mistral-Nemo-Instruct-2407-GGUF` (12B) | GGUF | `Q5_K_M` | ~8.7 GB |

Why Qwen2.5-14B as primary: strongest JSON/structured-output adherence of
this size class as of early 2026, which matters because every phase of this
pipeline (Phase 1/2/3 in `agent_prompts.py`) demands strict JSON — a model
that drifts from schema will break your pipeline live on stage. Keep the
7B as a hot-swap if the 14B is too slow for live demo latency.

**Download (do this with internet, before you leave):**

```bash
# Option A: huggingface-cli (recommended, resumable)
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF \
    qwen2.5-14b-instruct-q5_k_m.gguf \
    --local-dir ./usb/models/llm/qwen2.5-14b

huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q6_k.gguf \
    --local-dir ./usb/models/llm/qwen2.5-7b
```

> Double-check exact quant filenames on the model's "Files" tab before
> downloading — GGUF repo filenames vary slightly by uploader/quant round.

## 2. Embedding model (JD ↔ snippet similarity)

| Purpose | Model | Notes |
|---|---|---|
| **Recommended** | `BAAI/bge-small-en-v1.5` | 384-dim, strong retrieval quality/speed tradeoff, small enough to keep resident alongside the LLM |
| Lighter alt | `sentence-transformers/all-MiniLM-L6-v2` | Smaller, slightly weaker semantic separation on technical JD text |

```bash
huggingface-cli download BAAI/bge-small-en-v1.5 \
    --local-dir ./usb/models/embeddings/bge-small-en-v1.5
```

Both load fine via `sentence-transformers` fully offline once cached — see
`matcher.py`, which sets `local_files_only=True`.

## 3. Local inference runtime

You have two reasonable paths on GB10/DGX OS. Pick one as primary, stage both
binaries if you have room (WiFi is weak, not nonexistent — redundancy is
cheap insurance).

- **Ollama (recommended for hackathon speed):** ships an official
  `linux-arm64` build, has a built-in local HTTP server, and handles GGUF
  loading/quant management for you.
  ```bash
  curl -fsSL https://ollama.com/install.sh -o ./usb/bin/ollama_install.sh
  # on a connected linux/arm64 box, or extract the arm64 tarball directly:
  curl -L https://ollama.com/download/ollama-linux-arm64.tar.zst \
  -o ./usb/bin/ollama-linux-arm64.tar.zst
  ```
  At the venue: `tar --zstd -C /usr -xf ollama-linux-arm64.tar.zst`, then
  `ollama create talentforge-qwen -f Modelfile` pointing at your staged GGUF
  (see `scripts/setup_usb.sh`).

- **llama-cpp-python (fallback / for tighter Python-native control):** you
  need a source build or an aarch64+CUDA prebuilt wheel — **generic PyPI
  wheels for this package are x86_64 only**, so plan to build it once,
  online, on an aarch64 box (or the GB10 itself before the event) with:
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" pip wheel llama-cpp-python \
      --no-binary llama-cpp-python -w ./usb/wheels/
  ```
  Do this build *ahead of time* — compiling with CUDA flags takes several
  minutes and you don't want to be doing it for the first time on venue WiFi.

## 4. LaTeX → PDF compilation (offline, deterministic)

- **`tectonic`** (recommended): single self-contained binary, no TeX Live
  tree to manage, fetches packages once and caches them — but that caching
  needs network the *first* time, so pre-warm the cache before you leave:
  ```bash
  # aarch64 static binary
  curl -L https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-<version>-aarch64-unknown-linux-gnu.tar.gz \
      -o ./usb/bin/tectonic-aarch64.tar.gz
  ```
  Then, still online, compile your chosen template **once** with
  `tectonic --keep-logs resume.tex` so its bundle/package cache
  (`~/.cache/Tectonic`) is fully populated — **copy that cache directory
  onto the USB too**, or first-compile at the venue will try to hit the
  network and stall on weak WiFi.

- **Fallback: portable TeX Live (aarch64).** Heavier (multi-GB) but zero
  runtime network dependency at all once installed:
  ```bash
  # texlive.net's aarch64 net-installer, run once online:
  # install-tl --profile=texlive.profile --repository <mirror>
  ```
  Only worth it as a backup if you're nervous about Tectonic's cache being
  incomplete; it's meaningfully bigger to carry.

- **`poppler-utils`** (for `pdftotext`, used to sanity-check ATS-extractable
  text from your generated PDF, and to parse uploaded PDF resumes as a
  fallback to `pymupdf`):
  ```bash
  apt-get download poppler-utils:arm64 && cp *.deb ./usb/bin/
  ```

## 5. Python wheels — zero PyPI dependency at the venue

```bash
# Run this on an aarch64 Linux box (or via Docker --platform linux/arm64
# on any host) so the wheels match GB10's architecture:
pip download -r requirements.txt -d ./usb/wheels \
    --platform manylinux_2_28_aarch64 \
    --python-version 3.11 \
    --only-binary=:all:
```

Packages without prebuilt `manylinux_aarch64` wheels (rare, but check
`torch`, `sentence-transformers`'s C-extension deps, and `llama-cpp-python`)
need the `--no-binary` source-build path shown above, done ahead of time.

At the venue, install fully offline with:
```bash
pip install --no-index --find-links=./usb/wheels -r requirements.txt
```

## 6. Final pre-departure checklist for the drive itself

- [ ] `models/llm/qwen2.5-14b/*.gguf` (+ 7B fallback)
- [ ] `models/embeddings/bge-small-en-v1.5/` (full HF snapshot, not just weights)
- [ ] `bin/ollama-linux-arm64.tar.zst` (or legacy `.tgz`)
- [ ] `bin/tectonic-aarch64.tar.gz` + warmed `~/.cache/Tectonic` copied over
- [ ] `bin/poppler-utils*.deb` (arm64)
- [ ] `wheels/*.whl` (arm64, matching Python 3.11 or whatever DGX OS ships)
- [ ] `repo/talentforge-ai/` (this codebase, git-bundled: `git bundle create talentforge.bundle --all`)
- [ ] A second physical USB drive with the same payload (Murphy's law insurance for a 24-hour hackathon)
