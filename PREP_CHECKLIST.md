# Pre-Hackathon Prep Checklist

Run this tonight, in order, **with internet access**, before you stage the
USB drive. Each step tells you what "success" looks like.

## 1. Verify architecture assumptions

```bash
uname -m        # must print aarch64 if you're testing on the actual GB10
python3 --version
```

If you're prepping on an x86_64 laptop tonight (likely), remember every
binary/wheel you download must still target **aarch64** — see
`MANIFEST_USB.md`. Don't just `pip download` on your laptop and assume it
transfers; use `--platform manylinux_2_28_aarch64` explicitly.

## 2. Run the offline-safe unit tests right now (no models needed)

```bash
cd talentforge-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # normal PyPI, since you have internet now
pytest tests/ -v
```

Expected: all tests pass. `test_matcher.py` and `test_agent_prompts.py`
mock the embedding model and LLM respectively, so they validate your
actual pipeline _logic_ (segmentation, grounding checks, schema
validation, LaTeX escaping) independent of whether models are staged yet.
`test_latex_generator.py`'s compile test auto-skips if no LaTeX compiler
is on your current machine — that's fine tonight, it matters at the venue.

## 3. Download and stage models (see MANIFEST_USB.md for exact commands)

```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF \
    qwen2.5-14b-instruct-q5_k_m.gguf --local-dir ./usb/models/llm/qwen2.5-14b
huggingface-cli download BAAI/bge-small-en-v1.5 \
    --local-dir ./usb/models/embeddings/bge-small-en-v1.5
```

Success check: both directories are non-empty and the `.gguf` file size
roughly matches what's listed on the HF "Files" tab (a truncated download
is the #1 cause of "works tonight, corrupt tomorrow").

## 4. Stage the LaTeX toolchain AND warm its cache

```bash
curl -L <tectonic-aarch64-release-url> -o tectonic-aarch64.tar.gz
tar xzf tectonic-aarch64.tar.gz
./tectonic --keep-logs templates/resume_template.tex.jinja   # will fail — that's fine, it's a .jinja not .tex
```

Actually compile once for real, so the package cache gets populated:

```bash
python3 -c "
from backend.latex_generator import render_tex
from tests.test_latex_generator import sample_resume
" # or just run pytest -k full_compile if tectonic is on PATH
```

Then confirm the cache exists and copy it to the USB:

```bash
ls -la ~/.cache/Tectonic
cp -r ~/.cache/Tectonic ./usb/tectonic_cache
```

Success check: `~/.cache/Tectonic` has real package files in it, not just
an empty directory.

## 5. Build llama-cpp-python's aarch64+CUDA wheel NOW if you plan to use it

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip wheel llama-cpp-python \
    --no-binary llama-cpp-python -w ./usb/wheels/
```

This step is slow (several minutes) and network + compiler dependent —
do not attempt this for the first time at the venue. If you're going the
Ollama route instead (recommended), skip this step entirely.

## 6. Full offline dry run — simulate venue conditions

Turn off WiFi, then:

```bash
ollama serve &
ollama create talentforge-qwen -f models/Modelfile.template   # adjust FROM path first
./scripts/verify_pipeline.sh
```

Success check: script prints `All checks passed. You're ready for eHub.`
with WiFi disabled the entire time. If step 3 (embedding model load) or
step 5 (full pipeline) fails specifically because of a network call, that's
exactly the class of bug you want to catch tonight, not on stage.

## 7. Time-box a live demo rehearsal

Run the Streamlit app end-to-end with WiFi off, using a real messy resume
(yours) and a real JD you don't control the wording of (copy-paste one from
a company site), and time it:

```bash
streamlit run frontend/app_streamlit.py
```

If the 14B model's full 3-phase pipeline takes uncomfortably long for a
live demo, switch `LLMConfig.model_name` to `talentforge-qwen-7b` as your
demo default and keep the 14B as an "if judges ask about output quality"
comparison.

## 8. Pack

- [ ] Two physical copies of the USB payload
- [ ] Printed copy of `MANIFEST_USB.md` and this checklist (venue WiFi may
      be too weak to load your own GitHub repo page)
- [ ] Charger for the GB10 (280W USB-C — bring the right brick)
- [ ] A backup laptop with the repo cloned locally, in case the GB10 itself
      has setup issues at check-in
