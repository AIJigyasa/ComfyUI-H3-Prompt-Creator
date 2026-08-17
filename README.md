# ComfyUI H3 Prompt Creator

A multimodal MiniMax H3 prompt engine for ComfyUI with three nodes:

1. **H3 Video Prompt Creator** — idea + optional first/last frame. Automatically selects T2VA/I2VA/FL2VA/L2VA and writes the complete H3 three-field prompt.
2. **H3 Full-Reference Video Prompt Creator** — idea + optional IMAGE/VIDEO/AUDIO references. Builds a reference dossier, resolves `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`, and writes the six-section full-reference prompt.
3. **H3 Use Case Prompt Creator** — specialized H3 workflows based on the official MiniMax H3 skills.

## Source guides

The prompt rules are encoded from two official documents:

- *Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)*
- *Full-Reference Mode Rewrite Output Format Guide*

They live as `VIDEO_SYSTEM` and `FULL_REF_SYSTEM` in `h3_prompt_creator.py`. Rules a language model reliably ignores are additionally enforced in code after generation: shot timing, reference-label numbering, subject de-duplication, and refusal to cite an asset that was never connected. **If you revise the guides, update both the system prompt and the matching enforcement.**

## Providers

All three nodes can run against a local model or a hosted API. Pick with the `provider` widget:

| Provider | Needs | Default model | Env var |
|---|---|---|---|
| Ollama (Local) | nothing | `qwen3-vl:8b` (from `ollama_model`) | — |
| OpenAI | key | `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | key | `claude-opus-5` | `ANTHROPIC_API_KEY` |
| OpenRouter | key | `anthropic/claude-sonnet-5` | `OPENROUTER_API_KEY` |
| Google Gemini | key | `gemini-2.0-flash` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| Built-in deterministic | nothing | — | — |

**Prefer the environment variable over the `api_key` widget.** A key typed into the widget is saved into the workflow JSON and travels with the file if you share or export the workflow. Leave `api_key` blank and the node reads the variable above.

`api_model` overrides the default. It must name a **vision-capable** model whenever reference images are connected. `ollama_url` / `ollama_model` / `num_ctx` / `keep_alive` apply only to the local provider.

Everything is plain HTTP through `urllib` — no vendor SDKs, so the node adds no pip dependencies. Provider-specific handling worth knowing:

- **Anthropic**: `temperature` is omitted for models that reject it (Claude Opus 5, Sonnet 5, Opus 4.8/4.7, Fable 5 return a 400 if it is sent). Thinking is on by default on current Claude models and shares the `max_output_tokens` budget with the reply, so leave that generous. A `stop_reason: refusal` is reported rather than silently returning empty.
- **OpenAI**: if the model rejects `max_tokens`, the request is retried with `max_completion_tokens`; if it rejects a non-default `temperature`, that is dropped and retried.
- **Gemini**: uses `responseMimeType: application/json` for JSON replies.
- A missing key fails immediately with a clear message, before any slow call.

## Local AI pipeline

- **Qwen3-VL 8B via Ollama**: image analysis, sampled-video visual analysis, reference reasoning, and final H3 structuring.
- **faster-whisper**: local speech/lyrics transcription with timestamps.
- **librosa**: local audio/music feature analysis.
- **FFmpeg**: optional extraction of embedded audio from reference VIDEO sources.

The H3 nodes intentionally separate reusable subject identity from incidental source-image background/pose/camera/lighting unless the user explicitly asks to preserve those details. This reduces background/pose leakage when a reference is being used only for character identity.

## Install

**ComfyUI Manager** — search for "MiniMax H3 Prompt Creator" and install, then restart ComfyUI.

**Manually** — clone into `ComfyUI/custom_nodes/` and restart:

```bash
git clone https://github.com/AIJigyasa/ComfyUI-H3-Prompt-Creator
```

The node installs with **no Python dependencies** — the prompt engine talks to every backend over plain HTTP.

Audio analysis is opt-in, because faster-whisper and librosa are large. Without them the nodes simply report that transcription is unavailable; everything else works. To enable it, install into the ComfyUI environment:

```bash
pip install -r requirements-audio.txt
```

(The file is deliberately not called `requirements.txt` — ComfyUI Manager installs that automatically, which would make every install a heavy one.)

FFmpeg should be available on PATH for embedded audio extraction from reference videos.

## Ollama

Install a vision-capable Qwen model, e.g.:

```bash
ollama pull qwen3-vl:8b
```

Default Ollama server: `http://127.0.0.1:11434`

The nodes fall back to deterministic H3 formatting if Ollama is unavailable, but reference/media understanding is naturally better with Qwen3-VL available.

## Audio

Reference audio is never passed directly to Qwen3-VL. Instead:

- faster-whisper creates transcript + timestamps for speech/lyrics.
- librosa extracts duration, tempo estimate, spectral centroid/rolloff, zero-crossing and RMS dynamics.
- FFmpeg can extract audio embedded in a reference VIDEO.

The resulting evidence is supplied to Qwen3-VL as structured text for H3 reasoning.


## Troubleshooting Qwen3-VL on Ollama

This package uses Ollama's local HTTP API at `http://127.0.0.1:11434`. Qwen3-VL must be installed in Ollama and the installed name must match the node's `ollama_model` value (for example `qwen3-vl:8b`). The model dropdown is populated from your running Ollama server at ComfyUI start; use the **Refresh Ollama Models** button after pulling a new model.

Recommended checks in a terminal:

```bash
ollama list
```

### If the node runs forever and produces nothing

Three separate causes were behind this; all are fixed in the node, but they are worth understanding because they affect any Ollama integration.

**1. The context window (the big one).** If a request does not set `num_ctx`, Ollama sizes the context from the model's maximum — 262144 tokens for Qwen3-VL. That needs roughly 25 GB of KV cache, so on a 12 GB card most layers spill to system RAM. Check with:

```bash
ollama ps
```

A healthy row reads `100% GPU`. If it reads something like `62%/38% CPU/GPU`, generation drops from ~50 tok/s to ~5 tok/s and a single prompt can take 15 minutes. The `num_ctx` widget (default 8192) pins this. It auto-raises in fixed steps when many reference images are connected, so the model stays loaded instead of reloading on every run.

**2. The answer arrives in the wrong field.** On current Ollama builds, Qwen3-VL returns its reply in `message.thinking` and leaves `message.content` empty — even with `"think": false`. Code that reads only `message.content` gets an empty string every time and falls back to boilerplate. The node now reads whichever field is populated.

**3. Truncated JSON.** When generation hits `num_predict`, the JSON object is cut off mid-string and normal parsing discards the whole response. The node repairs unterminated JSON so a truncated generation is still usable, and reports clearly when it cannot.

### If you get dozens of near-identical subjects and no actual prompt

A reference that shows the same person more than once — a multi-pose contact sheet, a turnaround, a collage — makes the vision model treat each pose as a separate person. It then restates the same description until the token budget is gone, so `subject_definitions` fills with 27 copies of one woman and `detailed_description` is never written.

**Each reference asset is analyzed in its own request.** Sending every image in one batch makes the model cross-attribute — describing image 2's person under `<Picture 1>`. One asset per request removes the ambiguity, at the cost of one model call per connected reference. Turn off `enable_reference_analysis` if you would rather have the speed.

A character sheet counts as one subject. The analyzer is told that the same person in different poses, angles, crops **or outfits** is a single subject, described as a whole person with the outfit variants noted inside that one definition — not one subject per panel.

Further defences, in order:

- The analyzer is told that one person is one subject regardless of pose, and that "left/middle/right" are not different people.
- The schema caps subjects (3 per asset, 6 global) and the analyzer gets its own smaller token budget, so a runaway dossier cannot eat the whole run.
- Near-identical subjects are merged in code after generation, ignoring positional wording. Genuinely different people are preserved. The merge count appears in `generation_notes`.
- If `summary`, `retention_analysis` or `detailed_description` still come back empty, the node retries once, naming the empty fields and capping the subject list.
- Reference labels are renumbered across the whole prompt so `<Subject A>` becomes `<Subject 1>` and every section agrees.
- Shot timestamps that fall outside the target duration are respaced inside it, and Shot 1's timestamp is stripped.

### Speed reference

On an RTX 3060 12 GB with `qwen3-vl:8b` at `num_ctx` 8192:

| Operation | Time |
|---|---|
| Text-only prompt (T2VA) | ~5 s |
| One reference image (I2VA) | ~15 s |
| Three reference images, full-reference node | ~50 s |

If runs are much slower than this, check `ollama ps` for CPU offload first.

### Other controls

- **enable_reference_analysis** — the nodes make two model calls: a reference-dossier pass, then the prompt itself. Turning this off halves generation time when references are simple.
- **max_output_tokens** — raise if generation stops mid-sentence; the node reports when it hits the limit.
- Reference images are downscaled to 768px on the long edge before being sent, since Qwen3-VL's vision cost scales with resolution.
