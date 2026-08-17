import json, base64, io, re
from typing import Any

from .h3_prompt_creator import (
    VIDEO_SYSTEM,
    BackendConfig,
    _backend_chat,
    _clean,
    _json_schema,
    _model_choices,
    _preflight,
    _render_video,
    _safe_json,
)
from .h3_providers import OLLAMA, PROVIDERS

try:
    import torch
except Exception:
    torch = None


USE_CASES = [
    "Minimalist Product Ad",
    "Brand Promo Video",
    "3D Animation Short",
    "Co-op Game Intro",
    "Handdrawn Live-Action Fusion",
    "Music Video / Lyric Typography",
    "Paper Collage Explainer",
    "Papercraft Stop-Motion Explainer",
    "Custom H3 Use Case",
]

COMMON_FIELDS = {
    "idea": "Core idea / goal",
    "duration": "Target duration in seconds",
    "aspect_ratio": "Aspect ratio",
    "language": "Output language",
    "extra": "Extra constraints / creative notes",
}

USE_CASE_GUIDES = {
    "Minimalist Product Ad": {
        "required": ["product_image", "product_name", "variant", "copy", "template"],
        "fields": ["product_name", "variant", "copy", "template", "platform"],
        "summary": "Product images/materials; confirm variant, duration, ratio, Apple-style template and in-frame copy; build product facts, narrative spine, motion language, three independent anchors, precise beat/text storyboard. H3 is the default video model.",
        "references": "Product image is the primary identity source. Keep body color/material/structure consistent. Do not use a 4-panel anchor sheet.",
    },
    "Brand Promo Video": {
        "required": ["brand_name", "campaign_focus"],
        # "copy_language" was listed here but no widget existed, so it was never
        # collected. On-screen copy language is covered by the common "language".
        "fields": ["brand_name", "official_url", "campaign_focus", "audience", "cta", "narration_language", "claims"],
        "summary": "Collect/verify logo, fonts, colors, product/UI assets, official facts, audience, campaign focus, duration, ratio. Build truth sheet, provenance, story spine and exact beats before H3 prompt.",
        "references": "Identity-bearing assets should be treated as verified/user-provided sources; do not invent logos, UI, claims or metrics.",
    },
    "3D Animation Short": {
        "required": ["story"],
        "fields": ["story", "character_count", "world_style", "ending", "dialogue_style"],
        "summary": "Plan an end-to-end stylized 3D short with project brief, story outline, character/environment cards, standardized shot planning, continuity, timing, camera, performance, audio and final review.",
        "references": "Character and environment refs should be isolated into cards so identity and world continuity remain stable across shots.",
    },
    "Co-op Game Intro": {
        "required": ["player1", "player2", "game_title", "style"],
        "fields": ["player1", "player2", "game_title", "style", "ui_copy", "menu_theme"],
        "summary": "Collect two player names, game title, visual style, optional character refs. Build a confirmation-image concept using a fixed 16:9 menu framework, then refill the final H3 video prompt from the approved direction.",
        "references": "Character refs are identity-only: face silhouette, hairstyle, proportions, distinctive traits. Do not inherit photo realism, original lighting, camera quality or source style.",
    },
    "Handdrawn Live-Action Fusion": {
        "required": ["contact_object", "mood"],
        "fields": ["contact_object", "mood", "space", "drawn_entity", "language_constraint"],
        "summary": "Finished 15s 16:9 fusion clip: live-action space + glowing hand-drawn entity, clear contact in first 0-3s, continuous morphing, escape, delayed handheld chase; non-horror, cute/life-like.",
        "references": "Single continuous space or adjacent area; no unrelated scene cuts. Preserve morphing continuity and delayed camera pursuit.",
    },
    "Music Video / Lyric Typography": {
        "required": ["music_or_lyrics"],
        "fields": ["music_or_lyrics", "genre", "vocal_mode", "platform", "typography_direction", "character_direction", "reference_roles"],
        "summary": "Lock music/lyrics, duration, ratio, creative contract, character/scene/typography reference roles, beat-reactive multi-shot structure, typography and stitching. For >15s, use multi-shot with master audio continuity.",
        "references": "Character, scene and typography references have separate narrow jobs and must not cross-contaminate.",
    },
    "Paper Collage Explainer": {
        "required": ["topic", "learning_goal"],
        "fields": ["topic", "learning_goal", "audience", "visual_metaphor", "paper_style", "narration_language"],
        "summary": "Turn science/education/general knowledge into layered paper-collage explanation with clear learning goal, visual metaphor, character/asset plan, scene prompts, timing and sound.",
        "references": "Use references to establish collage material language, character/scene construction and continuity; avoid importing irrelevant background.",
    },
    "Papercraft Stop-Motion Explainer": {
        "required": ["topic", "learning_goal"],
        "fields": ["topic", "learning_goal", "audience", "paper_material", "diorama_style", "lesson_language"],
        "summary": "Create tactile handmade paper/cut-paper educational content with project brief, story, characters, layered diorama sets, props, storyboard, camera, sound, review and staged approvals.",
        "references": "Keep handmade material/character/set language consistent across scenes; separate reference roles and use approved assets as anchors.",
    },
    "Custom H3 Use Case": {
        "required": [],
        "fields": [],
        "summary": "Use the core H3 prompt-writing rules while adapting to a user-defined production use case.",
        "references": "Infer the minimum relevant reference roles from supplied media and user intent.",
    },
}


def _img_b64(image) -> str:
    if torch is None:
        raise RuntimeError("torch unavailable")
    arr = image[0] if getattr(image, "ndim", 0) == 4 else image
    arr = (arr.clamp(0, 1) * 255).byte().cpu().numpy()
    from PIL import Image
    im = Image.fromarray(arr).convert("RGB")
    # Cap the long edge: vision tokens scale with resolution.
    if max(im.size) > 768:
        scale = 768.0 / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _first_image(images):
    if images is None:
        return None
    try:
        if hasattr(images, "shape") and len(images.shape) == 4 and images.shape[0] > 0:
            return images[0:1]
    except Exception:
        return None
    return None


def _sample_video(video, max_frames=8):
    frames = []
    if video is None:
        return frames
    frames_tensor = None
    try:
        if isinstance(video, dict):
            frames_tensor = video.get("images") or video.get("frames")
        elif hasattr(video, "images"):
            frames_tensor = video.images
        elif hasattr(video, "frames"):
            frames_tensor = video.frames
    except Exception:
        pass
    if frames_tensor is None:
        return frames
    try:
        n = int(frames_tensor.shape[0])
        idxs = sorted(set([int(i * max(0, n-1) / max_frames) for i in range(max_frames)]))
        for i in idxs:
            frames.append(_img_b64(frames_tensor[i:i+1]))
    except Exception:
        pass
    return frames


# The three H3 fields plus the two report strings this node returns. Asking for
# the fields separately (instead of one prose blob) lets the shared renderer
# apply the same shot-timing and reference-label enforcement as the other nodes.
USE_CASE_SCHEMA = _json_schema(
    {
        "integrated_multimodal_description": {"type": "string"},
        "overall_soundscape": {"type": "string"},
        "non_diegetic_music": {"type": "string"},
        "production_brief": {"type": "string"},
        "use_case_analysis": {"type": "string"},
    },
    [
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
        "production_brief",
        "use_case_analysis",
    ],
)


def _extract_json(text: str):
    # Shared parser also repairs JSON truncated by the token limit.
    return _safe_json(text)


_H3_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")


def _split_h3_blob(text: str) -> dict:
    """Recover the three H3 fields from a single prose blob.

    Kept so a model that answers with the older "h3_prompt" string still yields
    structured fields for the shared renderer.
    """
    out = {}
    for i, key in enumerate(_H3_FIELDS):
        nxt = _H3_FIELDS[i + 1] if i + 1 < len(_H3_FIELDS) else None
        pattern = rf"{key}\s*:\s*(.*?)(?=\n\s*{nxt}\s*:|\Z)" if nxt else rf"{key}\s*:\s*(.*)\Z"
        m = re.search(pattern, text, re.S | re.I)
        if m and m.group(1).strip():
            out[key] = m.group(1).strip()
    if not out and text.strip():
        out["integrated_multimodal_description"] = text.strip()
    return out


def _as_h3_text(value: Any) -> str:
    """Coerce the model's h3_prompt into pasteable H3 text.

    Local models sometimes answer with a nested object instead of a string. A raw
    JSON blob is useless downstream, so rebuild the three H3 fields from it.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_as_h3_text(v) for v in value).strip()
    if isinstance(value, dict):
        if any(k in value for k in _H3_FIELDS):
            parts = []
            for key in _H3_FIELDS:
                inner = value.get(key)
                parts.append(f"{key}: {_as_h3_text(inner) if inner else 'N/A'}")
            return "\n\n".join(parts)
        # No recognisable H3 fields: flatten to readable lines rather than JSON.
        return "\n".join(f"{k}: {_as_h3_text(v)}" for k, v in value.items()).strip()
    return "" if value is None else str(value).strip()


class H3UseCasePromptCreator:
    @classmethod
    def INPUT_TYPES(cls):
        # Only the common fields are required. Every use-case-specific field is
        # optional so the UI can hide the ones this use case does not use, and
        # Python still has a default if the frontend omits them.
        return {
            "required": {
                "use_case": (USE_CASES, {"default": "Minimalist Product Ad"}),
                "idea": ("STRING", {"multiline": True, "default": "", "tooltip": COMMON_FIELDS["idea"]}),
                "duration": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "match reference"], {"default": "16:9"}),
                "language": ("STRING", {"default": "English"}),
                "extra": ("STRING", {"multiline": True, "default": "", "tooltip": COMMON_FIELDS["extra"]}),
                "provider": (PROVIDERS, {
                    "default": OLLAMA,
                    "tooltip": "Ollama (Local) needs no key. The hosted providers need api_key + api_model.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "leave blank to use an environment variable",
                    "tooltip": "Blank reads OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY. A key typed here is saved into the workflow JSON.",
                }),
                "api_model": ("STRING", {
                    "default": "",
                    "placeholder": "blank = provider default",
                    "tooltip": "Model for hosted providers; ignored by Ollama. Must be a vision model when reference images are connected.",
                }),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": ((_model_choices() or ["qwen3-vl:8b"]), {"default": (_model_choices() or ["qwen3-vl:8b"])[0]}),
                "temperature": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "request_timeout": ("INT", {"default": 300, "min": 30, "max": 3600, "step": 10}),
                "num_ctx": ([4096, 8192, 16384, 32768], {
                    "default": 8192,
                    "tooltip": "Ollama context window. Leaving this to Ollama's default forces the KV cache off the GPU and makes generation extremely slow.",
                }),
                "max_output_tokens": ("INT", {"default": 4096, "min": 256, "max": 12000, "step": 256}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_image_6": ("IMAGE",),
                "reference_video": ("VIDEO",),
                "reference_audio": ("AUDIO",),
                "product_name": ("STRING", {"default": ""}),
                "brand_name": ("STRING", {"default": ""}),
                "official_url": ("STRING", {"default": ""}),
                "campaign_focus": ("STRING", {"multiline": True, "default": ""}),
                "audience": ("STRING", {"default": ""}),
                "cta": ("STRING", {"default": ""}),
                "claims": ("STRING", {"multiline": True, "default": ""}),
                "variant": ("STRING", {"default": ""}),
                "copy": ("STRING", {"default": ""}),
                "template": (["White-tech", "Dark rim-light", "Brand color field", "Light lifestyle scene"], {"default": "White-tech"}),
                "story": ("STRING", {"multiline": True, "default": ""}),
                "character_count": ("INT", {"default": 2, "min": 1, "max": 20}),
                "world_style": ("STRING", {"default": ""}),
                "ending": ("STRING", {"default": ""}),
                "dialogue_style": ("STRING", {"default": ""}),
                "player1": ("STRING", {"default": ""}),
                "player2": ("STRING", {"default": ""}),
                "game_title": ("STRING", {"default": ""}),
                "style": ("STRING", {"default": ""}),
                "ui_copy": ("STRING", {"default": ""}),
                "menu_theme": ("STRING", {"default": ""}),
                "contact_object": ("STRING", {"default": ""}),
                "mood": ("STRING", {"default": ""}),
                "space": ("STRING", {"default": ""}),
                "drawn_entity": ("STRING", {"default": ""}),
                "language_constraint": ("STRING", {"default": ""}),
                "music_or_lyrics": ("STRING", {"multiline": True, "default": ""}),
                "genre": ("STRING", {"default": ""}),
                "vocal_mode": ("STRING", {"default": ""}),
                "platform": ("STRING", {"default": ""}),
                "typography_direction": ("STRING", {"default": ""}),
                "character_direction": ("STRING", {"default": ""}),
                "reference_roles": ("STRING", {"multiline": True, "default": ""}),
                "topic": ("STRING", {"default": ""}),
                "learning_goal": ("STRING", {"default": ""}),
                "visual_metaphor": ("STRING", {"default": ""}),
                "paper_style": ("STRING", {"default": ""}),
                "narration_language": ("STRING", {"default": ""}),
                "paper_material": ("STRING", {"default": ""}),
                "diorama_style": ("STRING", {"default": ""}),
                "lesson_language": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "production_brief", "use_case_analysis")
    FUNCTION = "generate"
    CATEGORY = "H3 / Use Case"

    def generate(self, use_case, idea, duration, aspect_ratio, language, extra="",
                 provider=OLLAMA, api_key="", api_model="",
                 ollama_url="http://127.0.0.1:11434", ollama_model="qwen3-vl:8b",
                 temperature=0.25, request_timeout=300, num_ctx=8192, max_output_tokens=4096,
                 reference_image_1=None, reference_image_2=None, reference_image_3=None,
                 reference_image_4=None, reference_image_5=None, reference_image_6=None,
                 reference_video=None, reference_audio=None, **fields):
        cfg = USE_CASE_GUIDES[use_case]
        # Only this use case's fields are read, so values left over from another
        # use case can never leak into the prompt.
        field_lines = []
        for f in cfg["fields"]:
            v = fields.get(f, "")
            if isinstance(v, (str, int, float)) and str(v).strip() != "":
                field_lines.append(f"{f}: {v}")
        missing = [f for f in cfg.get("required", []) if not str(fields.get(f, "")).strip()]
        if _clean(extra):
            field_lines.append(f"extra: {_clean(extra)}")

        images = []
        for ref in [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5, reference_image_6]:
            if ref is not None:
                try: images.append(_img_b64(ref))
                except Exception: pass
        video_frames = _sample_video(reference_video)
        images.extend(video_frames[:4])

        input_inventory = []
        if images: input_inventory.append(f"visual evidence: {len(images)} image/frame samples")
        if reference_video is not None: input_inventory.append("reference video connected")
        if reference_audio is not None: input_inventory.append("reference audio connected")

        # Inherit the full T2VA guide (camera vocabulary, cut phrasing, speaker
        # and timing rules) instead of restating a thinner version of it here.
        system = f"""{VIDEO_SYSTEM}

--- USE-CASE LAYER ---
You are additionally acting as a MiniMax H3 use-case prompt director.
USE CASE: {use_case}
GUIDE SUMMARY: {cfg['summary']}
REFERENCE PRINCIPLES: {cfg['references']}
Do not invent unsupported brand claims, logos, metrics or source facts. For reference images,
keep character/object identity separate from incidental pose/background unless this use case
explicitly requires them. Do not create generic filler sections.

Return a JSON object with exactly these five string keys:
integrated_multimodal_description, overall_soundscape, non_diegetic_music,
production_brief, use_case_analysis.
Every value MUST be a plain text string, never a nested object or array.
The first three are the H3 prompt fields and follow every rule above.
production_brief summarises the plan behind the prompt.
use_case_analysis explains how the output satisfies this use case and what each connected
reference contributes.
"""
        user = f"""Create the best executable H3 video prompt for this use case.
Idea: {idea}
Duration: {duration:.2f}s
Aspect ratio: {aspect_ratio}
Language: {language}
Timing constraint: the video is exactly {duration:.2f} seconds. Every shot timestamp must be
below {duration:.2f} seconds, Shot 1 has no timestamp, and there are at most {max(1, int(duration // 3) + 1)} shots.
Use-case fields:
""" + ("\n".join(field_lines) if field_lines else "(none)") + "\nInputs:\n" + ("; ".join(input_inventory) or "none") + "\n\nRules: The prompt must be directly usable, detailed but executable, timeline-aware, and obey the selected use-case's constraints. If references are connected, explain what each reference contributes and prevent unrelated background/pose/style leakage."

        cfg = BackendConfig(
            provider, ollama_url, ollama_model, api_key, api_model,
            temperature, "10m", int(request_timeout), int(max_output_tokens), int(num_ctx),
        )
        notes = []
        if missing:
            notes.append(f"missing recommended field(s) for {use_case}: {', '.join(missing)}")
        try:
            preflight_note = _preflight(cfg)
            if preflight_note:
                notes.append(preflight_note)
            text = _backend_chat(
                cfg, system, user,
                images=images or None,
                response_format=USE_CASE_SCHEMA,
            )
            obj = _extract_json(text)
            # Older prompts returned one prose blob under "h3_prompt"; accept both.
            if obj and obj.get("h3_prompt") and not obj.get("integrated_multimodal_description"):
                obj.update(_split_h3_blob(_as_h3_text(obj["h3_prompt"])))
            if obj and _clean(obj.get("integrated_multimodal_description")):
                notes.append(f"{cfg.provider}: {cfg.model_label}")
                if input_inventory:
                    notes.append(" | ".join(input_inventory))
                # Shared renderer applies shot-timing repair and label normalisation.
                h3_text = _render_video(obj, "T2VA", float(duration))
                return (
                    h3_text,
                    _as_h3_text(obj.get("production_brief", "")),
                    _as_h3_text(obj.get("use_case_analysis", "")) + "\n" + " | ".join(notes),
                )
        except Exception as exc:
            notes.append(f"Ollama fallback: {exc}")

        # Deterministic fallback; concise but valid H3 structure.
        desc = f"[Shot 1] Cinematic {use_case.lower()} video, {aspect_ratio}, {duration:.2f} seconds. {idea.strip() or 'Create a polished, coherent scene based on the supplied use case.'}"
        if field_lines: desc += " " + "; ".join(field_lines[:8]) + "."
        prompt = f"integrated_multimodal_description: {desc}\n\noverall_soundscape: Natural diegetic ambience and physical action sounds appropriate to the scene.\n\nnon_diegetic_music: A restrained, use-case-appropriate instrumental score with clear pacing and a controlled ending."
        brief = f"Use case: {use_case}\nDuration: {duration:.2f}s\nAspect ratio: {aspect_ratio}\n" + "\n".join(field_lines)
        return prompt, brief, "Deterministic fallback used. " + " | ".join(notes)


NODE_CLASS_MAPPINGS = {"H3UseCasePromptCreator": H3UseCasePromptCreator}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UseCasePromptCreator": "H3 Use Case Prompt Creator"}
