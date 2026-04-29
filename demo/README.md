---
title: nano-cohere-transcribe
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Pure-PyTorch ASR for Cohere Transcribe (14 languages)
hf_oauth: false
suggested_hardware: a10g-small
---

# nano-cohere-transcribe — Hugging Face Space

Demo Space for [`nano-cohere-transcribe`](https://github.com/Deep-unlearning/nano-cohere-transcribe),
a pure-PyTorch port of [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

## Setup (Space owner — one time)

The model is **gated**. The Space won't load without a token.

1. Visit the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and accept the license under *the same account* that owns the Space.
2. In **Space Settings → Variables and secrets**, add:
   - Secret name: `HF_TOKEN`
   - Value: a [User Access Token](https://huggingface.co/settings/tokens) with `read` scope.
3. Pick hardware: at minimum **T4-small** for usable latency, **A10G-small** for the headline numbers. ZeroGPU works but expect ~2.5 s per call cold-start overhead because of the dynamic GPU allocation.

## Local development

```bash
git clone https://github.com/Deep-unlearning/nano-cohere-transcribe
cd nano-cohere-transcribe/demo
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
hf auth login
python app.py
```
