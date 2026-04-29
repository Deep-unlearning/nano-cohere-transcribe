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
suggested_hardware: zero-a10g
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
3. Pick hardware. **ZeroGPU** is the recommended default — the app declares a 180-s GPU slot per call, which covers cold-load + warm-up + audio up to a few minutes. For low-latency interactive use or very long-form clips, upgrade to **A10G-small** or **A100-large**.

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
