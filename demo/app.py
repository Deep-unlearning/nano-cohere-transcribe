"""Gradio Space demo for nano-cohere-transcribe.

Designed to work on:
- ZeroGPU spaces (via `import spaces` + @spaces.GPU decorator)
- Paid GPU spaces (T4 / A10G / A100) — same code, decorator becomes a no-op
- Local CPU/GPU (run `python app.py`)

The model weights are gated. The Space needs an HF token in its
Settings → Variables and secrets as `HF_TOKEN` so
``huggingface_hub.snapshot_download`` can pull the checkpoint.
"""
from __future__ import annotations

# IMPORTANT: `import spaces` must come *before* any torch import — on ZeroGPU
# it monkey-patches torch to lazily initialise CUDA inside @spaces.GPU
# functions. If torch is imported first the patch is a no-op and the
# ``torch.device("cuda")`` calls inside our loader will fail at module load.
try:
    import spaces  # noqa: F401

    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import os
import time

import gradio as gr
import torch

from nano_cohere_transcribe import from_pretrained
from nano_cohere_transcribe.audio import load_audio_16k_mono
from nano_cohere_transcribe.tokenizer import SUPPORTED_LANGUAGES


LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "ar": "Arabic",
    "ja": "Japanese",
    "zh": "Chinese",
    "vi": "Vietnamese",
    "ko": "Korean",
}

MODEL_REPO = "CohereLabs/cohere-transcribe-03-2026"

# On any HF Space we target cuda — ZeroGPU's shim defers actual device
# allocation until we're inside @spaces.GPU. Locally, fall back to whatever
# the user has.
if HAS_SPACES or torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Module-level cache. Populated on first request inside the GPU context.
_model = None


def _ensure_model():
    """Lazily load the model on first call.

    On ZeroGPU this must run inside an @spaces.GPU function so CUDA is
    available; module-level CUDA allocations would fail. Globals persist
    between requests so subsequent calls reuse the loaded weights.
    """
    global _model
    if _model is None:
        _model = from_pretrained(MODEL_REPO, device=DEVICE, warmup=False)
    return _model


def _transcribe(audio_path, language, punctuation, batch_size):
    if audio_path is None:
        return "Please upload an audio file or record from the microphone.", ""

    model = _ensure_model()
    waveform = load_audio_16k_mono(audio_path)

    t0 = time.perf_counter()
    text = model.transcribe(
        waveform,
        language=language,
        punctuation=punctuation,
        batch_size=int(batch_size),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    audio_s = waveform.shape[0] / 16000.0
    rtfx = audio_s / dt if dt > 0 else float("nan")
    info = (
        f"**{audio_s:.1f}s** of audio  •  "
        f"transcribed in **{dt:.2f}s**  •  RTFx **{rtfx:.0f}×**"
    )
    return text, info


# 180 s ZeroGPU slot covers cold-load (~3 s) + warmup + transcription of
# clips up to a few minutes. Longer clips would need a longer slot.
if HAS_SPACES:
    transcribe = spaces.GPU(_transcribe, duration=180)
else:
    transcribe = _transcribe


language_choices = [(LANGUAGE_NAMES[c], c) for c in SUPPORTED_LANGUAGES]

with gr.Blocks(title="nano-cohere-transcribe") as demo:
    gr.Markdown(
        """
        # nano-cohere-transcribe

        Pure-PyTorch inference for [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
        — 14 languages, energy-based chunking for long audio, CUDA-graph
        decoder for low-latency streaming.

        Code: [github.com/Deep-unlearning/nano-cohere-transcribe](https://github.com/Deep-unlearning/nano-cohere-transcribe)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                label="Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )
            language = gr.Dropdown(
                label="Language",
                choices=language_choices,
                value="en",
            )
            with gr.Accordion("Advanced", open=False):
                punctuation = gr.Checkbox(label="Punctuation & casing", value=True)
                batch_size = gr.Slider(
                    label="Chunk batch size (parallelism for long audio)",
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=8,
                    info=(
                        "Long audio is auto-split at quiet points into ~35 s "
                        "chunks. This controls how many chunks the GPU "
                        "processes per forward pass."
                    ),
                )
            submit = gr.Button("Transcribe", variant="primary")

        with gr.Column(scale=1):
            text_out = gr.Textbox(label="Transcript", lines=12, max_lines=40)
            info_out = gr.Markdown()

    submit.click(
        transcribe,
        inputs=[audio, language, punctuation, batch_size],
        outputs=[text_out, info_out],
    )


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
