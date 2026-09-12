# models/ — USB mount point

This directory is intentionally empty in the repo. At the venue, copy the
staged weights from the USB drive here (or symlink):

```
models/
├── llm/
│   ├── qwen2.5-14b/qwen2.5-14b-instruct-q5_k_m.gguf
│   └── qwen2.5-7b/qwen2.5-7b-instruct-q6_k.gguf
└── embeddings/
    └── bge-small-en-v1.5/   (full HF snapshot directory)
```

`matcher.py` expects `models/embeddings/bge-small-en-v1.5` to exist and
loads it with `local_files_only=True` — it will raise a clear
`FileNotFoundError` rather than hang trying to hit the network if this
directory is missing or incomplete.

`agent_prompts.py`'s `LocalLLMClient` talks to Ollama by model tag
(`talentforge-qwen` by default), not a raw file path — register the GGUF
with Ollama first via `scripts/setup_usb.sh` or manually:

```bash
ollama create talentforge-qwen -f models/llm/qwen2.5-14b/Modelfile
```

See `MANIFEST_USB.md` for exact download commands and `PREP_CHECKLIST.md`
for the verification sequence.
