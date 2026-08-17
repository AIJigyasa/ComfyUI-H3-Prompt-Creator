"""ComfyUI MiniMax H3 Prompt Creator.

Two focused nodes:
1) H3 Video Prompt Creator
   - automatic T2VA / I2VA / FL2VA / L2VA selection from first/last frame inputs
   - optional local Ollama/VLM analysis
2) H3 Full-Reference Video Prompt Creator
   - image, video and audio reference inputs
   - automatic reference-label mapping + six-section H3 full-reference output
   - optional local Ollama/VLM analysis

The implementation follows the user-supplied H3 prompt-writing guides.  The
Ollama integration is intentionally HTTP-only: no Ollama Python package is
required.
"""
from __future__ import annotations

import base64
import difflib
import io
import json
import math
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .h3_audio_analysis import analyze_audio_features, extract_audio_from_video_source, transcribe_audio
from . import h3_providers
from .h3_providers import CLOUD_PROVIDERS, DETERMINISTIC, OLLAMA, PROVIDERS

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import av
except Exception:  # pragma: no cover
    av = None


# ---------------------------------------------------------------------------
# Small JSON-schema helper needed by both the analyzer and generators.
# ---------------------------------------------------------------------------

def _json_schema(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# H3 system prompts, distilled directly from the supplied user guides.
# ---------------------------------------------------------------------------

REFERENCE_ANALYZER_SYSTEM = r"""
You are the reference-analysis stage of an H3 video prompt system.
Your job is NOT to write the final video prompt. Build a clean reference dossier that a
second model will use to write it.

CRITICAL RULE: separate reusable subject identity from incidental image composition.
For a character/object reference, describe only attributes that should travel with that
subject into a new video: identity, face, hair, skin/material, body/build when relevant,
clothing, accessories, colors, distinctive marks, and other persistent design traits.
Do NOT copy the source image's background, room, location, furniture, pose, camera angle,
framing, or lighting into the subject description unless the user explicitly says that
those are part of what should be preserved.

If the source is a concrete first/last/keyframe, separately describe frame-anchor facts
that belong to the picture itself. Those facts must remain separate from reusable subject
identity so they do not leak into character-only references.

For reference video evidence, separate reusable subjects from source-video properties:
subject appearance vs. camera movement, editing rhythm, action path, temporal structure,
and environment. Do not treat the source video's background as a target background unless
requested.

IDENTITY DEDUPLICATION — mandatory, this is the most common failure:
- One physical person or object = exactly ONE subject candidate, however many times it appears.
- A character sheet, contact sheet, multi-pose sheet, turnaround or collage showing the SAME
  person is ONE subject candidate. Default to exactly one subject per image. Only produce a
  second subject when the image genuinely contains a DIFFERENT person, identifiable by a
  different face, hair colour, hair length or skin tone.
- Never split a subject by pose, camera angle, crop, framing, position ("left/middle/right"
  are not different people), OR BY OUTFIT. The same woman in a pink blazer in one panel and a
  beige suit in another is ONE person, not two. When outfits differ across panels, write one
  subject describing the person overall and list the outfit variants inside that single
  description.
- Describe the person as a whole — face, hair, skin tone, build and their clothing options.
  Do not describe each panel or shot separately.
- If two candidates would share face, hair and skin tone, they are the same person. Merge them.
- Never emit two candidates whose descriptions read almost the same. If you are about to repeat a
  description you already wrote, stop: you have already covered that subject.
- Background crowd members who are not individually important are one collective candidate, or
  are omitted entirely. Do not enumerate a crowd.
- Emit at most 3 subject candidates per asset and at most 6 global subjects in total. Fewer is better.

For audio evidence, use the supplied local analysis fields: transcript/segments from faster-whisper,
voice/speech presence, duration, RMS/peak and music/audio features from librosa. Distinguish dialogue,
singing/lyrics, music, ambience and sound effects only when supported by the evidence. Never invent
exact dialogue or lyrics.

Only describe evidence that the supplied asset actually contains. For an IMAGE asset,
leave video_structure and audio_characteristics out entirely — never describe motion,
editing, camera movement over time, or sound for a still image. Fill frame_anchor only
when the image genuinely acts as a concrete frame or composition anchor.

Return JSON only.
""".strip()

REFERENCE_ANALYZER_SCHEMA = _json_schema(
    {
        "assets": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "asset_type": {"type": "string"},
                "role_candidates": {"type": "array", "items": {"type": "string"}},
                "subject_candidates": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "type": {"type": "string"},
                        "identity_description": {"type": "string"},
                        "persistent_traits": {"type": "array", "items": {"type": "string"}},
                        "incidental_details_to_exclude": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["candidate_id", "type", "identity_description", "persistent_traits", "incidental_details_to_exclude"],
                    "additionalProperties": False,
                }, "maxItems": 3},
                "frame_anchor": {"type": "string"},
                "video_structure": {"type": "string"},
                "audio_characteristics": {"type": "string"},
            },
            # frame_anchor / video_structure / audio_characteristics are deliberately
            # NOT required: forcing them onto an IMAGE asset makes the analyzer invent
            # video and audio evidence, which then leaks <Video N>/<Audio N> into the
            # finished prompt for a task that has neither.
            "required": ["asset_id", "asset_type", "role_candidates", "subject_candidates"],
            "additionalProperties": False,
        }},
        "global_subjects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "canonical_name": {"type": "string"},
                "description": {"type": "string"},
                "identity_anchors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidate_id", "canonical_name", "description", "identity_anchors"],
            "additionalProperties": False,
        }, "maxItems": 6},
        "notes": {"type": "string"},
    },
    ["assets", "global_subjects", "notes"],
)

VIDEO_SYSTEM = r"""
You are an expert MiniMax H3 video prompt engineer. Follow the supplied H3
Video Prompt Writing Guide exactly.

TASK MODES are inferred from supplied keyframes:
- no first frame + no last frame = T2VA
- first frame only = I2VA
- first + last frame = FL2VA
- last frame only = L2VA
Never invent a different mode.

FINAL OUTPUT MUST HAVE EXACTLY THESE THREE FIELDS IN THIS ORDER:
1) integrated_multimodal_description
2) overall_soundscape
3) non_diegetic_music

T2VA has no alignment instruction.
I2VA begins with exactly one first-line alignment instruction stating that Picture 1
from Shot 1 is fully referenced at 0.00 seconds.
FL2VA begins with one first-line alignment instruction mapping Picture 1 to 0.00
seconds and Picture 2 to the final duration.
L2VA begins with one first-line alignment instruction mapping Picture 1 to the
final duration and the actual final shot.
Put one blank line after an alignment instruction.

The integrated_multimodal_description is a playback timeline, not a plot summary.

HARD TIMING RULES — these are constraints, not suggestions:
- The video is exactly the supplied target duration. Every timestamp must be strictly
  less than that duration. Never write a timestamp at or beyond it.
- Shot 1 carries NO timestamp. Write it as "[Shot 1]" with no time.
- Every later shot is written as "[Shot N] At MM:SS.mmm" with square brackets and
  strictly increasing times.
- Fit the action to the duration. A short duration means one or two shots, not six.
  Roughly one shot per 3-4 seconds is the upper limit on shot count.
Camera movement must be natural prose; motion type, amplitude and speed may be
included when meaningful. Do not stack camera labels as keywords.

At the start of [Shot 1] state the overall style and initial composition. Style
vocabulary: Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor,
vintage film. Example: "[Shot 1] Live-action, cinematic, a medium-wide shot frames..."

CUTS: use one of these exact phrasings — "the camera cuts to", "the shot cuts to",
"the shot transitions to", "the shot changes to", "the shot switches to". A cut must
introduce new information about subject, space, state, viewpoint or time. If only the
distance or a slight angle changes, use camera motion instead of a cut.

CAMERA MOTION = motion type + amplitude + speed, written as natural English action
inside the shot, never stacked as labels at the end of a sentence.
Motion types: Zoom In / Zoom Out, Push In / Pull Out, Pan Left / Pan Right,
Truck Left / Truck Right, Tilt Up / Tilt Down, Pedestal Up / Pedestal Down, Arc Shot,
Tracking Shot, Static Shot, Shake Slightly / Shake Strongly, POV,
Roll Clockwise / Roll Counterclockwise.
Amplitude: "with small amplitude" / "with large amplitude".
Speed: "at slow speed" / "at fast speed".
Omit amplitude and speed when they are medium/normal.
Example: "The camera pushes in with small amplitude at slow speed toward the letter."

Stable vocal IDs are (S1), (S2), etc. When already-numbered speakers speak together use
a compound ID such as (S1,S2). A speaker keeps the same ID across shots; characters who
never vocalize get no ID. When a speaker first appears, establish identity outside <d>
(type, age, gender, on/off-screen, pitch, timbre, rate, accent).
Actual spoken content appears only inside <d>[Language] ...</d>, containing nothing but
the language tag and the words themselves. Preserve the supplied dialogue verbatim.
For voiceover use the exact phrase "says in an off-screen voiceover", and immediately
after that <d> block state that the on-screen character's lips remain completely closed.
Use <scenetrans> at both connecting points when a line crosses a cut, and explicitly say
the audio continues across the cut. Use <cutoff> when speech is truncated at the end.
<scenetrans> and <cutoff> are ONLY for spoken or sung content. Never use them to mark a
camera move, a cut, or any visual transition — a video with no dialogue contains neither tag.
Visible on-screen text is quoted in English double quotation marks and preserved verbatim.

For keyframes:
- I2VA: establish the first image exactly, then develop forward.
- FL2VA: build one continuous, observable path from the first image to the last;
  favor a single shot unless a cut is explicitly needed.
- L2VA: infer a plausible earlier state and gradually converge on the final image.

overall_soundscape is 1–4 English sentences covering ambient sound, physical action
sounds and non-verbal human sounds across the video. Do not repeat dialogue/singing.
Use N/A only when complete silence is explicitly intended.

non_diegetic_music is 1–3 English sentences describing audience-only score through
instrumentation, tempo, rhythm and dynamic change. Do not use abstract mood words and do
not explain the emotional function of the score. Singing, instruments, radio, television
or phone music that the characters can hear are diegetic and belong in the multimodal
description, not here. Use N/A when there is no non-diegetic music.

The user may provide only a simple idea. In that case, intelligently invent the
supporting cinematic details needed to produce a complete H3-ready prompt, while
remaining faithful to the idea.
""".strip()

FULL_REF_SYSTEM = r"""
You are an expert MiniMax H3 full-reference video prompt engineer. Follow the
supplied Full-Reference Mode Rewrite Output Format Guide exactly.

FINAL OUTPUT MUST HAVE EXACTLY SIX SECTIONS IN THIS ORDER:
subject_definitions
summary
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music

Use reference labels consistently:
<Subject N> = reusable visible content abstracted from reference assets (people, animals,
  objects, scenes, environments, clothing, props, interfaces, effects, styles, actions,
  expressions, poses).
<Picture N> = a reference image used as a concrete target frame or shot-planning anchor.
<Video N> = whole-video source/continuation/editing/temporal structure.
<Audio N> = standalone audio asset or an enabled synchronized track that is copied or referenced.

Once a label is assigned, keep its meaning stable in every section.
<Video N> and <Audio N> are numbered independently; matching indices do not imply pairing,
and different indices do not prevent them coming from the same source asset.
NEVER invent a label for an asset that was not supplied. An ordinary reference video does
NOT create an <Audio N> merely because the file contains sound.

subject_definitions gives each tracked item its own line, written as natural English that
states what the label denotes, its reference role and the main features to follow — for
example "<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue
cardigan, and a thin silver necklace." One subject may be defined by several assets:
"<Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion
comes from <Video 1>." If a <Picture N> or <Video N> only identifies the source of another
item and is not used separately later, cite it inside that item's definition instead of
giving it its own line.

SUBJECT COUNT DISCIPLINE — mandatory, this is the most common failure:
- Define each distinct person or object exactly ONCE. Most prompts need 1-3 subjects.
  Never define more than 6. If you have more, you are splitting one person into many.
- Several poses, crops or frames of the same person are ONE <Subject N>. "Left/middle/right"
  or "pose 1/pose 2" are the same person, not different subjects.
- Never write two subject definitions with nearly identical wording. If a description you are
  about to write repeats one you already wrote, reuse the existing <Subject N> label instead.
- subject_definitions is a short list, not a catalogue: keep it under 120 words in total.
- detailed_description is the section that matters most. Budget your output so that summary,
  retention_analysis and detailed_description are always written in full. Never spend your
  length on subject_definitions and leave the later sections empty.

IMPORTANT REFERENCE-ROLE RULES:
- A character/object/style image is NOT automatically a <Picture N>. If it only defines reusable content, extract that content as <Subject N>.
- A standalone <Picture N> is reserved for a concrete first frame, keyframe, last frame, edited frame, or composition anchor.
- Subject definitions must describe reusable identity/design traits only. Explicitly exclude incidental background, pose, camera angle, framing and lighting unless the user requests them as preserved attributes.
- Use <Video N> for whole-video source/continuation/edit/camera structure, while extracting reusable visible people/objects/environments as <Subject N> when appropriate.
- Use <Audio N> for source audio relationships; do not invent exact speech/lyrics.

summary is one short English paragraph beginning with a square-bracketed task-type prefix.
Combine task types with " + " only when the task genuinely contains multiple relationships,
and never repeat a type. Available task types:
- keyframe completion: an image serves as a concrete first/key/last/edited frame anchor.
- reference generation: an image, video or audio guides character, scene, style, action,
  camera or storyboard WITHOUT being a concrete frame or an edited/continued source video.
- video editing: an existing source video is directly modified.
- video continuation: new content continues/extends/resumes from an existing source video.
- audio reuse: the same audio signal is reused in full or in part.
- audio reference: only style, timbre, content, texture, beat or continuity is referenced.
The mere presence of a video or audio asset does not create the matching task type. A
reference video supplying only camera movement, cuts or rhythm is reference generation.
IF AND ONLY IF the chosen task type includes "video editing", the summary begins, after
the prefix, with "The target video is an edited version of <Video 1>." For every other
task type that sentence is forbidden — never write it when no source video is being
edited, and never write it when no video asset is connected at all.
Do not introduce new reference labels in the summary.

retention_analysis uses one line per reference label. Visible content uses only these
markers: fully_preserved, partially_preserved, attribute_transfer, weak_reference.
Audio uses only: fully_copy, partially_copy, reference, weak_reference.
Do not invent other retention markers. Write each line as the label, the shots it appears
in, then the marker and an explanation after " - ", for example:
"<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ..."
"<Picture 2> ([Shot 1] first frame): fully_preserved - ..."
"<Video 1> (cut and pacing structure): weak_reference - ..."
Never write a speaker ID (Sx) in retention_analysis. Newly added actions, backgrounds or
plot events in the target video are NOT losses of reference fidelity.

The detailed_description is the main shot-by-shot playback description. Establish
composition, subject appearance/position, environment, lighting, action/state changes,
camera movement, sound and dialogue, and insert reference labels wherever they apply.

Unlike T2VA, the style opening is established in ONE OR TWO English sentences BEFORE
[Shot 1], not inside it. For example:
"The target video is in a cinematic, literary music-video style with soft lighting and a
slightly desaturated color palette.
[Shot 1] The scene opens in a crowded urban street..."

Insert each reference label at its first appearance and wherever its role applies. At the
first clear appearance of an important <Subject N>, describe its referenced characteristics,
frame position and current action. Do not redefine a label in later shots. Use natural
phrasing for frame anchors: "the shot begins from <Picture 1>", "the shot's keyframe
corresponds to <Picture 2>", "the shot ends on <Picture 3>". Cite <Audio N> in the shot or
audio phase where its relationship is active.

When a referenced subject speaks, keep BOTH labels: "<Subject 2> (S1) turns and says,
<d>[English] ...</d>". Where a speaker matches no defined subject, use a stable voice
description followed by (Sx). Verbal content existing only inside a directly reused
soundtrack uses <Audio N> as the source and gets NO (Sx). Write [unclear] for unintelligible
spans rather than guessing. Camera, cut, voiceover, on-screen text and <scenetrans>/<cutoff>
rules are identical to the T2VA guide.

HARD TIMING RULES — these are constraints, not suggestions:
- Every timestamp must be strictly less than the supplied target duration.
- Shot 1 carries NO timestamp; write it as "[Shot 1]".
- Later shots use "[Shot N] At MM:SS.mmm," with square brackets, strictly increasing.
- Fit the shot count to the duration, roughly one shot per 3-4 seconds at most.
Stable vocal IDs are global (S1), (S2), etc. Dialogue/lyrics only go inside
<d>[Language] ...</d>.

For generation tasks, detailed_description should normally be 350–500 English words
unless the task is unusually dialogue-dense or is primarily a direct source-video edit.

When a reference video is supplied, do not pretend it is merely an image: use <Video N>
to describe its source/edit/continuation role and use sampled visual frames as evidence.
When reference audio is supplied, use <Audio N> consistently and describe whether it is
copied, partially copied, or used as a reference. If only a waveform is available and
speech content cannot be recovered, do not invent words.

The user may provide only a simple idea. Infer a coherent reference-aware target video
from the supplied assets and idea, while never inventing a reference asset that was not supplied.
""".strip()


def _clean(v: Any) -> str:
    if v is None:
        return ""
    return str(v).replace("\r\n", "\n").strip()


def _duration(v: float) -> float:
    try:
        return max(0.1, float(v))
    except Exception:
        return 6.0


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    rem = seconds - mins * 60
    return f"{mins:02d}:{rem:06.3f}"


def _fmt_two(seconds: float) -> str:
    return f"{float(seconds):.2f}"


def _sentence(v: Any) -> str:
    text = _clean(v)
    if not text:
        return ""
    if text.upper() == "N/A":
        return "N/A"
    return text if text.endswith((".", "!", "?")) else text + "."


def _strip_fences(text: str) -> str:
    text = _clean(text)
    text = re.sub(r"^```(?:json|text|txt)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _repair_json(text: str) -> str:
    """Close an unterminated JSON object.

    Local models routinely hit the token limit mid-object. Recovering the
    complete keys is far better than discarding the whole generation.
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()

    repaired = text
    if escaped:
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    # Drop a dangling comma or a key with no value yet.
    repaired = re.sub(r'(?:,\s*"[^"]*"\s*:\s*)$', "", repaired)
    repaired = re.sub(r'(?:"[^"]*"\s*:\s*)$', "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)
    return repaired + "".join(reversed(stack))


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_fences(text)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    if start >= 0:
        # Truncated generations never reach a closing brace.
        candidates.append(_repair_json(text[start:]))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# Image / video / audio evidence helpers.
# ---------------------------------------------------------------------------


def _tensor_to_pil(image_tensor: Any, index: int = 0) -> Optional[Image.Image]:
    if Image is None or np is None or image_tensor is None:
        return None
    try:
        arr = image_tensor
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3:
            frame = arr
        elif arr.ndim >= 4:
            frame = arr[min(index, arr.shape[0] - 1)]
        else:
            return None
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = np.asarray(frame, dtype=np.float32)
        if frame.max() <= 1.5:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return Image.fromarray(frame, "RGB")
    except Exception:
        return None


# Qwen3-VL tokenizes by resolution, so a full-size frame can cost thousands of
# vision tokens. Capping the long edge keeps requests inside num_ctx and fast.
MAX_IMAGE_SIDE = 768


def _pil_to_b64(image: Image.Image, quality: int = 88, max_side: int = MAX_IMAGE_SIDE) -> str:
    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
        image = image.resize(new_size, resample)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_batch_to_b64(image_tensor: Any, max_images: int = 2) -> List[str]:
    pil_frames: List[str] = []
    if image_tensor is None:
        return pil_frames
    try:
        count = int(image_tensor.shape[0]) if hasattr(image_tensor, "shape") and len(image_tensor.shape) >= 4 else 1
    except Exception:
        count = 1
    indices = [0]
    if count > 1 and max_images > 1:
        indices.append(count - 1)
    for idx in indices[:max_images]:
        img = _tensor_to_pil(image_tensor, idx)
        if img is not None:
            pil_frames.append(_pil_to_b64(img))
    return pil_frames


def _extract_video_source(video: Any) -> Any:
    if video is None:
        return None
    for name in ("get_stream_source", "get_source", "get_path"):
        fn = getattr(video, name, None)
        if callable(fn):
            try:
                source = fn()
                if source:
                    return source
            except Exception:
                pass
    if isinstance(video, str):
        return video
    return None


def _sample_video_frames(video: Any, max_frames: int = 8) -> Tuple[List[str], Dict[str, Any]]:
    """Sample a small, evenly spaced set of JPEG frames from ComfyUI VIDEO.

    Current ComfyUI VIDEO objects expose a stream source; older/third-party
    video objects often expose the same method. This function stays best-effort.
    """
    source = _extract_video_source(video)
    meta: Dict[str, Any] = {"frame_count": None, "fps": None, "duration": None}
    if source is None or av is None:
        return [], meta

    frames: List[str] = []
    try:
        with av.open(source, mode="r") as container:
            if not container.streams.video:
                return [], meta
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            meta["fps"] = fps
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            meta["duration"] = duration

            # Uniform sampling by decoded frame index avoids requiring seek support.
            all_frames: List[Any] = []
            for frame in container.decode(stream):
                all_frames.append(frame)
                # Hard safety cap for unusually long videos.
                if len(all_frames) >= max(300, max_frames * 40):
                    break
            total = len(all_frames)
            meta["frame_count"] = total
            if total == 0:
                return [], meta
            if total <= max_frames:
                indices = list(range(total))
            else:
                indices = np.linspace(0, total - 1, max_frames).astype(int).tolist() if np is not None else [int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
            for idx in indices:
                frame = all_frames[idx]
                pil = Image.fromarray(frame.to_ndarray(format="rgb24")) if Image is not None else None
                if pil is not None:
                    frames.append(_pil_to_b64(pil))
    except Exception:
        return frames, meta
    return frames, meta


def _audio_metadata(audio: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "sample_rate": None,
        "channels": None,
        "samples": None,
        "duration_seconds": None,
        "rms": None,
        "peak": None,
    }
    if not isinstance(audio, dict):
        return meta
    waveform = audio.get("waveform")
    try:
        sr = int(audio.get("sample_rate", 0))
        meta["sample_rate"] = sr
        if waveform is None:
            return meta
        arr = waveform
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32) if np is not None else None
        if arr is None:
            return meta
        # [B,C,T] is the canonical ComfyUI AUDIO shape.
        if arr.ndim == 3:
            channels = arr.shape[1]
            samples = arr.shape[2]
            flat = arr.reshape(-1)
        elif arr.ndim == 2:
            channels = arr.shape[0]
            samples = arr.shape[1]
            flat = arr.reshape(-1)
        else:
            channels = 1
            samples = arr.size
            flat = arr.reshape(-1)
        meta["channels"] = int(channels)
        meta["samples"] = int(samples)
        if sr:
            meta["duration_seconds"] = round(samples / sr, 3)
        if flat.size:
            meta["rms"] = round(float(np.sqrt(np.mean(flat * flat))), 5)
            meta["peak"] = round(float(np.max(np.abs(flat))), 5)
    except Exception:
        pass
    try:
        features = analyze_audio_features(audio)
        if features.get("available"):
            meta["feature_analysis"] = features
    except Exception as exc:
        meta["feature_analysis_error"] = str(exc)
    return meta


def _audio_spectrogram_b64(audio: Any) -> Optional[str]:
    """Create a compact grayscale spectrogram image when numpy/PIL are available."""
    if np is None or Image is None or not isinstance(audio, dict):
        return None
    try:
        waveform = audio.get("waveform")
        if waveform is None:
            return None
        arr = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            x = arr[0].mean(axis=0)
        elif arr.ndim == 2:
            x = arr.mean(axis=0)
        else:
            x = arr.reshape(-1)
        if x.size < 256:
            return None
        # Limit to ~15s for compact requests.
        sr = int(audio.get("sample_rate", 44100) or 44100)
        max_samples = min(x.size, int(sr * 15))
        x = x[:max_samples]
        n_fft = 512
        hop = 256
        if x.size < n_fft:
            return None
        windows = 1 + (x.size - n_fft) // hop
        spec = np.empty((n_fft // 2 + 1, windows), dtype=np.float32)
        window = np.hanning(n_fft).astype(np.float32)
        for i in range(windows):
            seg = x[i * hop : i * hop + n_fft] * window
            spec[:, i] = np.abs(np.fft.rfft(seg))
        spec = np.log1p(spec)
        spec -= spec.min()
        if spec.max() > 0:
            spec /= spec.max()
        img = (255.0 * (1.0 - spec)).astype(np.uint8)
        # Make it visually readable without external plotting dependencies.
        pil = Image.fromarray(img, mode="L").resize((768, 256))
        return _pil_to_b64(pil.convert("RGB"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ollama HTTP client.
# ---------------------------------------------------------------------------


class BackendConfig:
    """Everything needed to reach the selected backend.

    Carried as one object so adding a provider does not mean threading another
    parameter through every generation function.
    """

    def __init__(
        self,
        provider: str = OLLAMA,
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen3-vl:8b",
        api_key: str = "",
        api_model: str = "",
        temperature: float = 0.25,
        keep_alive: str = "10m",
        timeout: int = 600,
        max_output_tokens: int = 4096,
        num_ctx: int = 8192,
    ) -> None:
        self.provider = provider
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.api_key = h3_providers.api_key_for(provider, api_key)
        self.api_model = _clean(api_model) or h3_providers.default_model(provider)
        self.temperature = float(temperature)
        self.keep_alive = keep_alive
        self.timeout = int(timeout)
        self.max_output_tokens = int(max_output_tokens)
        self.num_ctx = int(num_ctx)

    @property
    def is_local(self) -> bool:
        return self.provider == OLLAMA

    @property
    def is_deterministic(self) -> bool:
        return self.provider == DETERMINISTIC

    @property
    def model_label(self) -> str:
        return self.ollama_model if self.is_local else self.api_model


def _backend_chat(
    cfg: BackendConfig,
    system: str,
    user: str,
    images: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Single entry point for every backend. Returns the raw text reply."""
    budget = int(max_output_tokens or cfg.max_output_tokens)
    if cfg.is_local:
        return _ollama_chat(
            cfg.ollama_url,
            cfg.ollama_model,
            system,
            user,
            images=images or None,
            temperature=cfg.temperature,
            keep_alive=cfg.keep_alive,
            timeout=cfg.timeout,
            response_format=response_format,
            max_output_tokens=budget,
            num_ctx=cfg.num_ctx,
        )
    return h3_providers.cloud_chat(
        cfg.provider,
        cfg.api_key,
        cfg.api_model,
        system,
        user,
        images=images or None,
        temperature=cfg.temperature,
        max_tokens=budget,
        timeout=cfg.timeout,
        want_json=True,
    )


def _notes(items: List[str]) -> str:
    """Join generation notes, dropping the blanks that produce ' | | '."""
    return " | ".join(_clean(item) for item in items if _clean(item))


def _fit_num_ctx(num_ctx: int, image_count: int, max_output_tokens: int = 4096) -> int:
    """Grow the context window so the attached images actually fit.

    A 768px frame costs Qwen3-VL roughly 1k vision tokens. Six references plus
    sampled video frames overflow an 8192 window and the prompt gets silently
    truncated, so raise the window (never lower the user's choice).

    The result snaps to fixed buckets: Ollama reloads the model whenever num_ctx
    changes, so a stable value keeps it resident between runs.
    """
    needed = 1536 + int(image_count) * 1024 + int(max_output_tokens)
    target = max(int(num_ctx), needed)
    for bucket in (4096, 8192, 16384, 32768):
        if target <= bucket:
            return bucket
    return 32768


def _model_matches(model: str, installed: List[str]) -> bool:
    """Ollama treats 'name' and 'name:latest' as the same model."""
    wanted = _clean(model)
    if not wanted:
        return False
    variants = {wanted, wanted + ":latest", wanted.rsplit(":latest", 1)[0]}
    return any(name in variants for name in installed)


def _ollama_health(base_url: str, model: str) -> Dict[str, Any]:
    """Fast preflight so the node fails clearly instead of appearing stuck."""
    base = base_url.rstrip("/")
    result: Dict[str, Any] = {"ok": False, "version": "", "models": [], "model_available": False}
    try:
        req = urllib.request.Request(base + "/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result["version"] = str(json.loads(resp.read().decode("utf-8")).get("version", "unknown"))
        req = urllib.request.Request(base + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        result["models"] = [str(m.get("name")) for m in obj.get("models", []) if m.get("name")]
        result["model_available"] = _model_matches(model, result["models"])
        result["ok"] = True
        if not result["model_available"]:
            raise RuntimeError(
                f"Ollama is running, but model '{model}' is not installed. "
                f"Installed models: {', '.join(result['models']) or 'none'}"
            )
        return result
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama preflight HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to Ollama at {base_url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama preflight timed out at {base_url}.") from exc


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def _final_answer(content: str, thinking: str) -> str:
    """Return the model's answer regardless of which channel it arrived in.

    Recent Ollama builds route the whole reply of some thinking-capable models
    (Qwen3-VL among them) into ``message.thinking`` and leave ``message.content``
    empty, even with ``think: false``. Reading only ``content`` yields nothing.
    """
    content = _clean(content)
    if content:
        stripped = _clean(_THINK_BLOCK.sub("", content))
        return stripped or content
    thinking = _clean(thinking)
    if not thinking:
        return ""
    match = _THINK_BLOCK.search(thinking)
    if match:
        after = _clean(thinking[match.end():])
        if after:
            return after
        return _clean(match.group(1))
    return _clean(re.sub(r"</?think>", "", thinking))


def _ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user_text: str,
    images: Optional[List[str]] = None,
    temperature: float = 0.25,
    keep_alive: str = "10m",
    timeout: int = 600,
    response_format: Optional[Dict[str, Any]] = None,
    max_output_tokens: int = 4096,
    num_ctx: int = 8192,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    msg: Dict[str, Any] = {"role": "user", "content": user_text}
    if images:
        msg["images"] = images

    base_payload: Dict[str, Any] = {
        "model": _clean(model),
        "messages": [{"role": "system", "content": system}, msg],
        "stream": False,
        # Qwen3-VL is a thinking-capable model. We do not need a reasoning trace
        # for prompt generation; disabling it keeps generation short and fast.
        "think": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_output_tokens),
            # Without an explicit num_ctx, Ollama sizes the context from the
            # model's maximum (262144 for Qwen3-VL). The resulting KV cache is
            # far larger than a consumer GPU, so most layers spill to CPU and
            # generation slows to a crawl. This single option is the difference
            # between ~5 tok/s and ~50 tok/s on a 12 GB card.
            "num_ctx": int(num_ctx),
            # Vision models fall into degenerate loops on reference tasks,
            # restating the same subject until the token budget is exhausted and
            # the later sections never get written. This penalises that directly.
            # Kept mild: higher values also penalise the repeated digits in
            # timestamps and push the model into nonsense times.
            "repeat_penalty": 1.08,
            "repeat_last_n": 320,
        },
    }

    def do_request(format_value: Any) -> Tuple[str, str]:
        payload = dict(base_payload)
        payload["format"] = format_value
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        message = obj.get("message") or {}
        answer = _final_answer(message.get("content", ""), message.get("thinking", ""))
        return answer, _clean(obj.get("done_reason"))

    attempts: List[Any] = []
    if response_format is not None:
        attempts.append(response_format)
    attempts.append("json")

    last_error = ""
    for index, format_value in enumerate(attempts):
        is_last = index == len(attempts) - 1
        try:
            answer, done_reason = do_request(format_value)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # Some Ollama builds/models reject schema-constrained output. Retry
            # with ordinary JSON mode while keeping the schema in the prompt.
            if exc.code in (400, 422) and not is_last:
                last_error = f"HTTP {exc.code}: {detail[:200]}"
                continue
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail[:700]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {timeout}s. Lower num_ctx/max_output_tokens "
                f"or check that '{model}' fits in VRAM."
            ) from exc

        if answer:
            if done_reason == "length" and _safe_json(answer) is None:
                raise RuntimeError(
                    f"Ollama hit the {max_output_tokens}-token limit before finishing valid JSON. "
                    "Raise max_output_tokens, or lower temperature so the model stops rambling."
                )
            return answer
        last_error = "empty response"

    raise RuntimeError(
        f"Ollama returned an empty answer ({last_error or 'no content'}). "
        f"Check that '{model}' is a chat/vision model."
    )


def _ollama_models(base_url: str) -> List[str]:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return [str(m.get("name")) for m in obj.get("models", []) if m.get("name")]
    except Exception:
        return []



# ---------------------------------------------------------------------------
# Subject de-duplication.
#
# Vision models split one person into many "subjects" when a reference is a
# multi-pose contact sheet, then repeat near-identical descriptions until the
# token budget is gone and the real prompt sections never get written. Prompt
# rules reduce this; these helpers guarantee it.
# ---------------------------------------------------------------------------

MAX_GLOBAL_SUBJECTS = 6
MAX_ASSET_SUBJECTS = 3
_DUP_RATIO = 0.82

# Positional wording is how the model labels the same person in different poses.
_POSITIONAL = re.compile(
    r"\b(left|middle|centre|center|right|first|second|third|front|back|rear|"
    r"pose\s*\d+|figure\s*\d+|panel\s*\d+|view\s*\d+)\b",
    re.I,
)


def _norm_desc(text: str) -> str:
    """Normalise a description so poses/positions do not make it look unique."""
    text = _clean(text).lower()
    text = _POSITIONAL.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(sorted(set(text.split())))


def _near_duplicate(a: str, b: str, ratio: float = _DUP_RATIO) -> bool:
    na, nb = _norm_desc(a), _norm_desc(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= ratio


def _dedupe_by_text(items: List[Any], key, limit: int) -> List[Any]:
    kept: List[Any] = []
    seen: List[str] = []
    for item in items:
        text = _clean(key(item))
        if not text:
            continue
        if any(_near_duplicate(text, prior) for prior in seen):
            continue
        kept.append(item)
        seen.append(text)
        if len(kept) >= limit:
            break
    return kept


def _dedupe_dossier(obj: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    """Collapse repeated subjects in the analyzer output. Returns (obj, removed)."""
    if not isinstance(obj, dict):
        return obj, 0
    removed = 0

    globals_in = [g for g in (obj.get("global_subjects") or []) if isinstance(g, dict)]
    globals_out = _dedupe_by_text(
        globals_in,
        lambda g: f"{g.get('canonical_name', '')} {g.get('description', '')}",
        MAX_GLOBAL_SUBJECTS,
    )
    removed += len(globals_in) - len(globals_out)
    obj["global_subjects"] = globals_out

    for asset in obj.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        cands = [c for c in (asset.get("subject_candidates") or []) if isinstance(c, dict)]
        kept = _dedupe_by_text(cands, lambda c: c.get("identity_description", ""), MAX_ASSET_SUBJECTS)
        removed += len(cands) - len(kept)
        asset["subject_candidates"] = kept
    return obj, removed


def _dedupe_subject_definitions(text: str) -> str:
    """Collapse repeated '<Subject N> = ...' entries in the final output.

    Deliberately stricter than the dossier pass: later sections reference these
    labels, so dropping a merely-similar definition would leave a dangling
    <Subject N>. Only near-verbatim repeats are collapsed here.
    """
    text = _clean(text)
    # The guide's form is "<Subject 1> is the young woman in <Picture 1>, ...";
    # models also emit "<Subject 1> = ...". Accept either and keep the connector.
    # Split only where a label actually starts a definition: a bare mention such
    # as "the same form appears as <Subject 4> and <Subject 5>" must not split.
    entries = re.split(r"(?=<Subject\s*\d+>\s*(?:=|is\s))", text)
    if len(entries) < 2:
        return text
    preamble = entries[0].strip() if not entries[0].lstrip().startswith("<Subject") else ""
    defs = [e.strip() for e in entries if e.strip().startswith("<Subject")]
    if not defs:
        return text
    kept: List[str] = []
    seen: List[str] = []
    for entry in defs:
        remainder = re.sub(r"^<Subject\s*\d+>", "", entry)
        body = remainder.lstrip(" =\t").strip()
        if any(_near_duplicate(body, prior, ratio=0.93) for prior in seen):
            continue
        kept.append(remainder.rstrip())
        seen.append(body)
    renumbered = [f"<Subject {i + 1}>{remainder}" for i, remainder in enumerate(kept)]
    return "\n".join(([preamble] if preamble else []) + renumbered).strip()


_LABEL_RE = re.compile(r"<\s*(Subject|Picture|Video|Audio)\s*([A-Za-z0-9]{1,3})\s*>", re.I)
_BARE_LABEL_RE = re.compile(r"\b(Subject|Picture|Video|Audio)\s+([A-Z])\b")
_HTML_RE = re.compile(r"</?\s*(br|p|div|span|b|i|em|strong)\s*/?>", re.I)


def _normalize_h3_labels(text: str, duration: float = 0.0) -> str:
    """Force H3 reference labels to the numbered form the spec requires.

    Models drift to <Subject A>/<Subject B> or leak stray HTML. Labels must be
    numbered and stable, so the whole prompt is renumbered in one pass to keep
    every section consistent with the others.
    """
    text = _clean(text)
    if not text:
        return text
    text = _HTML_RE.sub(" ", text)
    # <scenetrans>/<cutoff> mark speech continuing across a cut. With no <d>
    # dialogue anywhere in the prompt they are always wrong, and models emit
    # them as generic transition markers.
    if not re.search(r"<d>", text, re.I):
        text = re.sub(r"\s*</?(?:scenetrans|cutoff)>\s*", " ", text, flags=re.I)

    # First-appearance order per label type, across the entire prompt.
    order: Dict[str, List[str]] = {}
    for kind, ident in _LABEL_RE.findall(text):
        seen = order.setdefault(kind.capitalize(), [])
        key = ident.upper()
        if key not in seen:
            seen.append(key)
    for kind, ident in _BARE_LABEL_RE.findall(text):
        seen = order.setdefault(kind.capitalize(), [])
        if ident.upper() not in seen:
            seen.append(ident.upper())

    def renumber(match: "re.Match[str]") -> str:
        kind = match.group(1).capitalize()
        ident = match.group(2).upper()
        seq = order.get(kind, [])
        return f"<{kind} {seq.index(ident) + 1}>" if ident in seq else match.group(0)

    text = _LABEL_RE.sub(renumber, text)
    text = _BARE_LABEL_RE.sub(renumber, text)

    # Shot 1 must never carry a timestamp.
    text = re.sub(r"(\[Shot\s*1\])[,\s]*At\s*\d{1,2}:\d{2}(?:\.\d{1,3})?[,\s]*", r"\1 ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# A shot HEADING starts a sentence or line. A bare "[Shot 1]" in the middle of a
# sentence ("matching the framing of [Shot 1]") is a citation and must be left
# alone — timestamping it corrupts the prose.
_SHOT_MARKER_RE = re.compile(
    r"(?:^|(?<=[.!?;\n]))(\s*)\[Shot\s*(\d+)\]\s*,?\s*"
    r"(?:At\s*(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\s*,?)?\s*",
    re.I | re.M,
)


def _fix_shot_times(text: str, duration: float) -> str:
    """Enforce the guide's shot-timing format on a description.

    Shot 1 must have no timestamp; every later shot must carry a strictly
    increasing cut time inside the duration. Models both omit those timestamps
    and emit times past the end of the video, so the whole sequence is respaced
    when anything is wrong.

    Only ever call this on a description body. It must not touch
    retention_analysis, where bare "[Shot 1], [Shot 3]" citations are correct.
    """
    text = _clean(text)
    matches = list(_SHOT_MARKER_RE.finditer(text))
    if not matches:
        return text
    duration = _duration(duration)

    times: List[Optional[float]] = [
        (int(m.group(3)) * 60 + float(m.group(4))) if m.group(3) is not None else None
        for m in matches
    ]
    later = times[1:]
    already_valid = (
        times[0] is None
        and all(t is not None for t in later)
        and all(0 < t < duration for t in later)  # type: ignore[operator]
        and all(b > a for a, b in zip(later, later[1:]))  # type: ignore[operator]
    )
    if already_valid:
        return text

    step = duration / len(matches)
    out: List[str] = []
    cursor = 0
    for i, match in enumerate(matches):
        out.append(text[cursor : match.start()])
        lead = "\n" if "\n" in (match.group(1) or "") else (" " if match.start() > 0 else "")
        number = match.group(2)
        out.append(
            f"{lead}[Shot {number}] " if i == 0
            else f"{lead}[Shot {number}] At {_fmt_time(step * i)}, "
        )
        cursor = match.end()
    out.append(text[cursor:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()


_EDIT_SENTENCE_RE = re.compile(
    r"\s*The target video is an edited version of <Video\s*\d+>\.\s*", re.I
)


def _strip_unavailable_labels(
    text: str, has_image: bool, has_video: bool, has_audio: bool
) -> Tuple[str, List[str]]:
    """Remove references to assets that were never connected.

    The guide forbids inventing a label for an unsupplied asset. The editing
    boilerplate is a standalone sentence and is safe to delete outright; any
    other stray label is reported rather than cut, since removing it mid-sentence
    would damage the prompt.
    """
    text = _clean(text)
    warnings: List[str] = []
    if not has_video and _EDIT_SENTENCE_RE.search(text):
        text = _EDIT_SENTENCE_RE.sub(" ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
    if not has_audio and re.search(r"<Audio\s*\d+>", text, re.I):
        # Retention clauses leak into the audio sections as
        # "<Audio 1> (music reference): partially_copy - ...". Drop the clause and
        # the dangling label; the descriptive remainder is still usable prose.
        text = re.sub(
            r"<Audio\s*\d+>\s*(?:\([^)]*\))?\s*:\s*"
            r"(?:fully_copy|partially_copy|reference|weak_reference)\s*[-–]\s*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"<Audio\s*\d+>", "the soundtrack", text, flags=re.I)
        text = re.sub(r"[ \t]{2,}", " ", text)
    for label, present in (("Video", has_video), ("Audio", has_audio), ("Picture", has_image)):
        if not present and re.search(rf"<{label}\s*\d+>", text, re.I):
            warnings.append(f"<{label} N> referenced but no {label.lower()} asset connected")
    return text, warnings


def _first_sentence(text: str, limit: int = 160) -> str:
    text = _clean(text).replace("\n", " ")
    match = re.search(r"^(.{20,}?[.!?])(\s|$)", text)
    out = match.group(1) if match else text
    return out[:limit].rstrip()


def _asset_bindings(obj: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """One authoritative line per asset: which person/content that asset holds.

    The generator swapped two people because the dossier listed subjects as a
    flat pile with no asset attached. This mapping makes the binding explicit.
    """
    bindings: Dict[str, str] = {}
    if not obj:
        return bindings
    for asset in obj.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        aid = _clean(asset.get("asset_id"))
        if not aid:
            continue
        for subj in asset.get("subject_candidates") or []:
            desc = _clean(subj.get("identity_description")) if isinstance(subj, dict) else ""
            if desc and aid not in bindings:
                bindings[aid] = _first_sentence(desc)
    for glob in obj.get("global_subjects") or []:
        if not isinstance(glob, dict):
            continue
        aid = _clean(glob.get("source_asset"))
        desc = _clean(glob.get("description")) or _clean(glob.get("canonical_name"))
        if aid and desc and aid not in bindings:
            bindings[aid] = _first_sentence(desc)
    return bindings


def _reference_dossier_text(obj: Optional[Dict[str, Any]]) -> str:
    """Render the dossier grouped BY ASSET.

    Every subject must appear underneath the asset it came from. A flat list
    lets the generator reassign a person to the wrong picture, and model-chosen
    candidate ids collide across assets (both returned "Subject-0"), so ids are
    regenerated per asset here.
    """
    if not obj:
        return "No reference dossier available."

    blocks: Dict[str, List[str]] = {}
    order: List[str] = []

    def block(asset_id: str) -> List[str]:
        if asset_id not in blocks:
            blocks[asset_id] = []
            order.append(asset_id)
        return blocks[asset_id]

    for asset in obj.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        aid = _clean(asset.get("asset_id")) or "unidentified asset"
        lines = block(aid)
        atype = _clean(asset.get("asset_type"))
        if atype:
            lines.append(f"type: {atype}")
        roles = ", ".join(_clean(x) for x in (asset.get("role_candidates") or []) if _clean(x))
        if roles:
            lines.append(f"role candidates: {roles}")
        tag = re.sub(r"[^A-Za-z0-9]", "", aid) or "A"
        for index, subj in enumerate(asset.get("subject_candidates") or [], 1):
            if not isinstance(subj, dict):
                continue
            desc = _clean(subj.get("identity_description"))
            if not desc:
                continue
            persist = ", ".join(_clean(x) for x in (subj.get("persistent_traits") or []) if _clean(x))
            exclude = ", ".join(_clean(x) for x in (subj.get("incidental_details_to_exclude") or []) if _clean(x))
            lines.append(f"subject {tag}-{index} (belongs to {aid}): {desc}")
            if persist:
                lines.append(f"  persistent traits: {persist}")
            if exclude:
                lines.append(f"  exclude from identity: {exclude}")
        for key, label in (
            ("frame_anchor", "frame-anchor facts"),
            ("video_structure", "video structure"),
            ("audio_characteristics", "audio characteristics"),
        ):
            value = _clean(asset.get(key))
            if value:
                lines.append(f"{label}: {value}")

    for glob in obj.get("global_subjects") or []:
        if not isinstance(glob, dict):
            continue
        aid = _clean(glob.get("source_asset")) or "unassigned"
        name = _clean(glob.get("canonical_name"))
        desc = _clean(glob.get("description"))
        if not (name or desc):
            continue
        anchors = ", ".join(_clean(x) for x in (glob.get("identity_anchors") or []) if _clean(x))
        lines = block(aid)
        lines.append(f"named subject (belongs to {aid}): {name} - {desc}")
        if anchors:
            lines.append(f"  identity anchors: {anchors}")

    if not order:
        return "No reference dossier available."

    out: List[str] = [
        "REFERENCE DOSSIER - each block below belongs to exactly ONE asset.",
        "Never move a subject from one asset to another. If a block says a person",
        "belongs to <Picture 2>, that person is in <Picture 2> and nowhere else.",
        "",
    ]
    bindings = _asset_bindings(obj)
    if bindings:
        out.append("QUICK BINDING:")
        out.extend(f"  {aid} = {summary}" for aid, summary in bindings.items())
        out.append("")
    for aid in order:
        out.append(f"=== {aid} ===")
        out.extend(blocks[aid])
        out.append("")
    notes = _clean(obj.get("notes"))
    if notes:
        out.append(f"Analysis notes: {notes}")

    text = "\n".join(out).strip() or "No reference dossier available."
    # A bloated dossier teaches the generator to enumerate subjects too.
    if len(text) > 5000:
        text = text[:5000].rstrip() + "\n\n[dossier truncated]"
    return text


def _audio_analysis_text(transcription: Optional[Dict[str, Any]], features: Optional[Dict[str, Any]], source_label: str = "AUDIO") -> str:
    lines = [f"{source_label} ANALYSIS"]
    t = transcription or {}
    f = features or {}
    if t.get("available"):
        lines.append(f"Speech transcription ({t.get('language') or 'unknown'}, model {t.get('model') or 'unknown'}):")
        lines.append(t.get("text") or "[No speech text detected]")
        segs = t.get("segments") or []
        if segs:
            lines.append("Timed segments:")
            for seg in segs[:40]:
                lines.append(f"[{seg.get('start', 0):.3f}-{seg.get('end', 0):.3f}] {seg.get('text', '')}")
    else:
        lines.append(f"Speech transcription unavailable: {t.get('reason', 'not run')}")
    fa = f.get("feature_analysis") if isinstance(f, dict) else None
    if fa:
        lines.append("Audio/music features:")
        for key in ["duration_seconds", "sample_rate", "rms", "peak", "tempo_bpm_estimate", "spectral_centroid_hz", "spectral_rolloff_hz", "zero_crossing_rate", "dynamic_rms_mean", "dynamic_rms_std"]:
            if key in fa and fa.get(key) is not None:
                lines.append(f"{key}: {fa[key]}")
        lines.append(f"feature_backend: {fa.get('analysis_backend', 'unknown')}")
    if f.get("_extracted_from_video"):
        lines.append("Source: audio track extracted from reference video with FFmpeg.")
    return "\n".join(lines)


def _analyze_references(
    cfg: BackendConfig,
    units: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Analyze each reference asset in its OWN request.

    Sending every image in one batch makes the model cross-attribute — it
    described image 2's blonde woman under <Picture 1>. One asset per request
    removes the ambiguity entirely: the only images in the request are the ones
    the asset actually owns.
    """
    units = [u for u in units if u.get("images") or u.get("asset")]
    if not units:
        return None, "Reference analysis: none"

    merged: Dict[str, Any] = {"assets": [], "global_subjects": [], "notes": ""}
    notes: List[str] = []
    failures = 0
    for unit in units:
        asset = unit.get("asset") or {}
        label = _clean(asset.get("asset_id")) or "asset"
        unit_images = unit.get("images") or []
        user_note = _clean(unit.get("user_note"))
        user = (
            f"Analyze ONE reference asset: {label}.\n"
            f"Every image in this request belongs to {label} and to nothing else. "
            f"There are {len(unit_images)} image(s) for it.\n"
            + (
                f"The user describes the references as follows. Treat any statement about "
                f"{label} as ground truth and follow it over your own impression:\n"
                f"\"{user_note}\"\n"
                if user_note else ""
            )
            + "If those images show the SAME person in different poses, angles, crops or "
            "outfits (a character sheet), that is exactly ONE subject: describe the person "
            "overall and never split them per pose or per outfit.\n"
            "Keep reusable subject identity separate from incidental composition. "
            "Return JSON only.\n\n" + json.dumps({"assets": [asset]}, ensure_ascii=False, indent=2)
        )
        try:
            raw = _backend_chat(
                cfg,
                REFERENCE_ANALYZER_SYSTEM,
                user,
                images=unit_images or None,
                response_format=REFERENCE_ANALYZER_SCHEMA,
                # The dossier is an intermediate note, not the deliverable.
                max_output_tokens=min(cfg.max_output_tokens, 1024),
            )
            obj = _safe_json(raw)
        except Exception as exc:
            obj = None
            notes.append(f"{label}: {exc}")
        if not obj:
            failures += 1
            continue
        obj, _ = _dedupe_dossier(obj)
        for entry in obj.get("assets") or []:
            if isinstance(entry, dict):
                # Trust our own label over whatever the model echoed back.
                entry["asset_id"] = label
                merged["assets"].append(entry)
        for glob in obj.get("global_subjects") or []:
            if isinstance(glob, dict):
                glob["source_asset"] = label
                merged["global_subjects"].append(glob)
        if _clean(obj.get("notes")):
            notes.append(f"{label}: {_clean(obj.get('notes'))}")

    if not merged["assets"] and not merged["global_subjects"]:
        raise RuntimeError("Reference analyzer returned no usable output.")

    merged["notes"] = " | ".join(notes)[:1200]
    # Only collapse duplicates WITHIN an asset. Two different images legitimately
    # describe two different people, and merging across them loses one of them.
    total_before = len(merged["global_subjects"])
    merged["global_subjects"] = merged["global_subjects"][:MAX_GLOBAL_SUBJECTS]
    removed = sum(
        1
        for a in merged["assets"]
        for _ in range(max(0, len(a.get("subject_candidates") or []) - MAX_ASSET_SUBJECTS))
    ) + max(0, total_before - len(merged["global_subjects"]))

    note = f"Reference analysis: {len(units)} asset(s) analyzed separately"
    if removed:
        note += f" ({removed} duplicate subject(s) merged)"
    if failures:
        note += f" ({failures} asset(s) failed)"
    return merged, note


def _generate_with_provider(
    cfg: BackendConfig,
    system: str,
    user: str,
    images: List[str],
    schema: Dict[str, Any],
    required_keys: Tuple[str, ...] = (),
    min_words: int = 0,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if cfg.is_deterministic:
        return None, "Provider: Built-in deterministic"

    def generate(user_text: str) -> Optional[Dict[str, Any]]:
        raw = _backend_chat(cfg, system, user_text, images=images or None, response_format=schema)
        return _safe_json(raw)

    obj = generate(user)
    if not obj:
        raise RuntimeError(
            f"{cfg.provider} returned non-JSON content; deterministic formatter used."
        )

    note = f"Provider: {cfg.provider} ({cfg.model_label})"
    missing = [k for k in required_keys if not _clean(obj.get(k))]
    # The guide sets 350-500 words for generation tasks; local models routinely
    # stop near 150. Treat a very short body as a field worth regenerating.
    if min_words and "detailed_description" in required_keys and "detailed_description" not in missing:
        if len(_clean(obj.get("detailed_description")).split()) < min_words:
            missing.append("detailed_description")
    if missing:
        # Almost always caused by the model spending its whole budget listing
        # subjects. Ask again, naming the empty fields and forbidding the list.
        repair = (
            user
            + "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED. These required fields were empty or too "
            "short: " + ", ".join(missing)
            + ".\nWrite every field in full this time. Keep subject_definitions to at most "
            "80 words and at most 3 subjects, then spend the remaining length on "
            + ", ".join(missing)
            + ". Do not repeat any description you have already written."
        )
        if min_words and "detailed_description" in missing:
            repair += (
                f"\ndetailed_description must be {min_words}-500 English words of shot-by-shot "
                "playback detail: composition, subject appearance and position, environment and "
                "lighting, action and state changes, camera movement, sound and dialogue, and "
                "where each reference label takes effect. It is a timeline, not a summary."
            )
        retry = generate(repair)
        if retry:
            for key in missing:
                fresh = _clean(retry.get(key))
                if not fresh:
                    continue
                current = _clean(obj.get(key))
                # A retry triggered by shortness can come back shorter still;
                # never trade a fuller section for a thinner one.
                if not current or len(fresh.split()) > len(current.split()):
                    obj[key] = retry[key]
            still = [k for k in required_keys if not _clean(obj.get(k))]
            note += f" | retried for empty section(s): {', '.join(missing)}"
            if still:
                note += f" | STILL EMPTY: {', '.join(still)}"
        else:
            note += f" | empty section(s), retry failed: {', '.join(missing)}"
    return obj, note


# ---------------------------------------------------------------------------
# Renderers + deterministic fallbacks.
# ---------------------------------------------------------------------------


def _preflight(cfg: BackendConfig) -> str:
    """Fail fast on an unreachable backend or a missing key."""
    if cfg.is_deterministic:
        return ""
    if cfg.is_local:
        health = _ollama_health(cfg.ollama_url, cfg.ollama_model)
        return f"Ollama connected (v{health.get('version', 'unknown')}), model ready: {cfg.ollama_model}"
    if not cfg.api_key:
        names = " or ".join(h3_providers.ENV_KEYS.get(cfg.provider, ("an API key",)))
        raise RuntimeError(
            f"{cfg.provider} needs an API key. Type one into api_key, or set ${names} "
            "in the environment and restart ComfyUI."
        )
    return h3_providers.preflight(cfg.provider, cfg.api_key, cfg.api_model)


def _video_task(first_frame: Any, last_frame: Any) -> str:
    if first_frame is not None and last_frame is not None:
        return "FL2VA"
    if first_frame is not None:
        return "I2VA"
    if last_frame is not None:
        return "L2VA"
    return "T2VA"


def _last_shot_index(text: str) -> int:
    """Highest [Shot N] in the description.

    The FL2VA/L2VA alignment instruction must name the actual final shot, which
    is only known once the description exists.
    """
    numbers = [int(n) for n in re.findall(r"\[Shot\s*(\d+)\]", _clean(text))]
    return max(numbers) if numbers else 1


def _render_video(obj: Dict[str, Any], task: str, duration: float, last_shot: int = 0) -> str:
    body = _clean(obj.get("integrated_multimodal_description"))
    if body:
        body = _fix_shot_times(body, duration)
    if not last_shot:
        last_shot = _last_shot_index(body)
    if task == "I2VA":
        prefix = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    elif task == "FL2VA":
        prefix = (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {last_shot}) aligns with the {_fmt_two(duration)}-second mark of the target video.\n\n"
        )
    elif task == "L2VA":
        prefix = (
            "How the reference pictures align with the target video — <Picture 1> (from "
            f"[Shot {last_shot}]) aligns with the {_fmt_two(duration)}-second mark of the target video.\n\n"
        )
    else:
        prefix = ""
    return prefix + _normalize_h3_labels("\n".join(
        [
            f"integrated_multimodal_description: {body or 'A complete cinematic audiovisual timeline unfolds across the target duration.'}",
            "",
            f"overall_soundscape: {_clean(obj.get('overall_soundscape')) or 'N/A'}",
            "",
            f"non_diegetic_music: {_clean(obj.get('non_diegetic_music')) or 'N/A'}",
        ]
    ), duration)


def _fallback_video(idea: str, extra: str, task: str, duration: float, first_desc: str = "", last_desc: str = "") -> str:
    idea = _clean(idea) or "A cinematic scene unfolds with clear subject action and a coherent audiovisual timeline."
    extra = _clean(extra)
    base = f"[Shot 1] Cinematic, live-action, a coherent composition establishes the scene and main subject. {idea}"
    if task == "I2VA":
        if first_desc:
            base += f" The opening state from <Picture 1> is preserved: {first_desc}"
        base += " The action then develops forward from that exact opening state."
    elif task == "FL2VA":
        if first_desc:
            base += f" The shot begins from <Picture 1>, preserving its visible subject, composition and scene anchors: {first_desc}"
        base += " The action evolves continuously through observable intermediate states."
        if last_desc:
            base += f" The final motion settles into the composition and state established by Picture 2: {last_desc}"
    elif task == "L2VA":
        base += " The action develops from a plausible preceding state and progressively converges on the final referenced composition."
        if last_desc:
            base += f" <Picture 1> is reached at the end with this visible state: {last_desc}"
    if extra:
        base += f" Additional creative constraint: {extra}"
    return _render_video(
        {
            "integrated_multimodal_description": base,
            "overall_soundscape": "Natural ambient room tone and physical action sounds support the scene.",
            "non_diegetic_music": "N/A",
        },
        task,
        duration,
        1,
    )


_CONTINUE_RE = re.compile(
    r"\b(continue|continues|continuation|continuing|extend|extends|extending|"
    r"resume|resumes|pick up (?:from|where)|carry on)\b", re.I
)
_EDIT_RE = re.compile(
    r"\b(edit|edits|edited|editing|re-?cut|re-?cuts|remix|modif(?:y|ies|ied)|"
    r"alter|replace .* in the video|change the video)\b", re.I
)
_KEYFRAME_RE = re.compile(
    r"\b(first frame|last frame|final frame|end frame|opening frame|closing frame|"
    r"key ?frame|starts? (?:from|on|with) (?:this|the) (?:image|picture|frame)|"
    r"ends? (?:on|with) (?:this|the) (?:image|picture|frame))\b", re.I
)
_AUDIO_REUSE_RE = re.compile(
    r"\b(reuse|re-?use|copy the audio|copy the sound|same audio|same soundtrack|"
    r"keep (?:the )?(?:audio|sound|music|soundtrack)|original audio|original soundtrack|"
    r"1:1)\b", re.I
)

# Emission order for combined task types, matching the guide's own examples
# ("[video continuation + keyframe completion]", "[video editing + audio reuse]").
_TASK_ORDER = [
    "video editing",
    "video continuation",
    "keyframe completion",
    "reference generation",
    "audio reuse",
    "audio reference",
]


def _infer_reference_task(
    intent: str,
    has_image: bool,
    has_video: bool,
    has_audio: bool,
    extra: str,
    explicit_audio: bool = True,
) -> str:
    """Derive the summary's bracketed task type.

    Combines types with " + " the way the guide requires, instead of returning a
    single label. Two guide rules drive the details:
      - "The mere presence of video or audio does not automatically create a
        corresponding task type." A video supplying only camera movement, cuts or
        rhythm is reference generation, not video editing.
      - "An ordinary reference video does not create <Audio N> merely because the
        file contains sound." Audio extracted from a reference video therefore
        does not by itself make this an audio task; ``explicit_audio`` says
        whether the user actually connected an AUDIO input.
    """
    chosen = _clean(intent)
    if chosen.lower() != "auto":
        return chosen.lower()

    text = _clean(extra)
    types: List[str] = []

    if has_video:
        if _CONTINUE_RE.search(text):
            types.append("video continuation")
        elif _EDIT_RE.search(text):
            types.append("video editing")
        else:
            # Only camera/rhythm/style guidance was implied, so this is generation.
            types.append("reference generation")

    if has_image:
        types.append("keyframe completion" if _KEYFRAME_RE.search(text) else "reference generation")

    wants_audio_reuse = bool(_AUDIO_REUSE_RE.search(text))
    if explicit_audio or (has_audio and wants_audio_reuse):
        types.append("audio reuse" if wants_audio_reuse else "audio reference")

    if not types:
        types = ["reference generation"]

    # De-duplicate (video and image can both imply reference generation), then
    # emit in the canonical order.
    unique = {t for t in types}
    return " + ".join(t for t in _TASK_ORDER if t in unique)


def _render_full_ref(obj: Dict[str, Any], duration: float = 0.0) -> str:
    order = [
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    ]
    out: List[str] = []
    for i, key in enumerate(order):
        value = _clean(obj.get(key))
        if key == "subject_definitions" and value:
            value = _dedupe_subject_definitions(value)
        elif key == "detailed_description" and value and duration:
            # Description body only — retention_analysis cites shots without times.
            value = _fix_shot_times(value, duration)
        out.append(f"{key}:")
        out.append(value or "N/A")
        if i < len(order) - 1:
            out.append("")
    # Renumber across the whole prompt so labels agree between sections.
    return _normalize_h3_labels("\n".join(out), duration)


def _fallback_full_ref(idea: str, intent: str, image_count: int, video: bool, audio: bool, extra: str) -> str:
    defs: List[str] = []
    for i in range(image_count):
        defs.append(f"<Picture {i+1}> is reference image {i+1} supplied for target-video composition and visual identity.")
    if video:
        defs.append("<Video 1> is the supplied reference video and provides the source visual structure and temporal evidence for the target video.")
    if audio:
        defs.append("<Audio 1> is the supplied reference audio asset and provides an audio reference for the target video.")
    if not defs:
        defs.append("No reference assets were supplied.")
    detail = f"[Shot 1] Cinematic, the target scene follows the user's core idea: {_clean(idea) or 'a coherent audiovisual sequence'}."
    if video:
        detail += " The supplied <Video 1> informs the source structure, subject continuity and motion language without inventing unseen details."
    if image_count:
        detail += " The supplied picture references establish visible identity, composition, environment and style where applicable."
    if audio:
        detail += " <Audio 1> informs the specified audio relationship without inventing unverified dialogue or lyrics."
    if extra:
        detail += f" Additional instruction: {_clean(extra)}"
    return _render_full_ref(
        {
            "subject_definitions": "\n".join(defs),
            "summary": f"[{intent}] The target video develops the user's idea using the supplied reference relationships.",
            "retention_analysis": "\n".join([
                *(f"<Picture {i+1}> ([Shot 1]): fully_preserved - its visible reference role is retained." for i in range(image_count)),
                *(["<Video 1> (source structure): weak_reference - the target uses the supplied visual evidence to guide structure and motion."] if video else []),
                *(["<Audio 1>: reference - the supplied audio guides the target audio relationship without copying unknown content."] if audio else []),
            ]) or "No retention analysis supplied.",
            "detailed_description": detail,
            "overall_soundscape": "Natural scene ambience and physical action sounds support the target sequence.",
            "non_diegetic_music": "N/A",
        }
    )


# ---------------------------------------------------------------------------
# Shared Ollama widget controls.
# ---------------------------------------------------------------------------


_DEFAULT_MODEL = "qwen3-vl:8b"


def _model_choices() -> List[str]:
    """Offer the models actually installed in Ollama, preferring Qwen3-VL."""
    installed = _ollama_models("http://127.0.0.1:11434")
    if not installed:
        return [_DEFAULT_MODEL]
    preferred = [m for m in installed if "vl" in m.lower() or "vision" in m.lower()]
    ordered = preferred + [m for m in installed if m not in preferred]
    if _DEFAULT_MODEL in ordered:
        ordered.remove(_DEFAULT_MODEL)
        ordered.insert(0, _DEFAULT_MODEL)
    return ordered


class _OllamaMixin:
    @staticmethod
    def _ollama_inputs() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        models = _model_choices()
        return {
            "provider": (
                PROVIDERS,
                {
                    "default": OLLAMA,
                    "tooltip": (
                        "Ollama (Local) runs on your machine and needs no key. OpenAI, Anthropic, "
                        "OpenRouter and Google Gemini are hosted APIs and need api_key + api_model. "
                        "Built-in deterministic skips the model entirely."
                    ),
                },
            ),
            "api_key": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "placeholder": "leave blank to use an environment variable",
                    "tooltip": (
                        "Key for the hosted providers. Leave blank to read OPENAI_API_KEY, "
                        "ANTHROPIC_API_KEY, OPENROUTER_API_KEY or GEMINI_API_KEY from the "
                        "environment — safer, because a key typed here is saved into the "
                        "workflow JSON and travels with it if you share the workflow."
                    ),
                },
            ),
            "api_model": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "placeholder": "blank = provider default",
                    "tooltip": (
                        "Model for the hosted providers; ignored by Ollama. Blank uses the "
                        "default: OpenAI gpt-4o, Anthropic claude-opus-5, "
                        "OpenRouter anthropic/claude-sonnet-5, Gemini gemini-2.0-flash. "
                        "Must be a vision model when you connect reference images."
                    ),
                },
            ),
            "ollama_url": (
                "STRING",
                {
                    "default": "http://127.0.0.1:11434",
                    "multiline": False,
                    "placeholder": "http://127.0.0.1:11434",
                    "tooltip": "Ollama server base URL.",
                },
            ),
            "ollama_model": (
                models,
                {
                    "default": models[0],
                    "tooltip": "Vision-capable local model, listed from your running Ollama server. Use the Refresh Ollama Models button after pulling a new model.",
                },
            ),
            "temperature": (
                "FLOAT",
                {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.2,
                    "step": 0.05,
                    "round": 2,
                    "tooltip": "Lower values improve H3 schema adherence.",
                },
            ),
            "keep_alive": (
                ["0", "5m", "10m", "30m", "1h"],
                {"default": "10m", "tooltip": "Keep the local Ollama model loaded between requests."},
            ),
            "request_timeout": (
                "INT",
                {
                    "default": 600,
                    "min": 30,
                    "max": 3600,
                    "step": 10,
                    "tooltip": "Maximum Ollama request time in seconds.",
                },
            ),
            "max_output_tokens": (
                "INT",
                {
                    "default": 4096,
                    "min": 256,
                    "max": 12000,
                    "step": 256,
                    "tooltip": "Maximum generated tokens. H3 full-reference descriptions benefit from 4096+.",
                },
            ),
            "num_ctx": (
                [4096, 8192, 16384, 32768],
                {
                    "default": 8192,
                    "tooltip": (
                        "Ollama context window. CRITICAL for speed: left unset, Ollama sizes the "
                        "context from the model maximum (262144 for Qwen3-VL), which needs ~25 GB of "
                        "KV cache and pushes most layers onto the CPU. 8192 keeps an 8B model fully "
                        "on a 12 GB GPU. Raise it only when you connect many reference images."
                    ),
                },
            ),
            "enable_reference_analysis": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": (
                        "Run the separate reference-dossier pass before writing the prompt. "
                        "Turn off to halve generation time when references are simple."
                    ),
                },
            ),
            "whisper_model": (
                ["tiny", "base", "small"],
                {
                    "default": "small",
                    "tooltip": "Local faster-whisper model used for dialogue/lyrics transcription. Small is the recommended quality/speed balance.",
                },
            ),
            "whisper_device": (
                ["auto", "cuda", "cpu"],
                {"default": "auto", "tooltip": "Device for faster-whisper.",},
            ),
            "enable_audio_transcription": (
                "BOOLEAN",
                {"default": True, "tooltip": "Run faster-whisper when audio is connected (or embedded in the reference video).",},
            ),
        }


# ---------------------------------------------------------------------------
# Node 1 — automatic T2VA / I2VA / FL2VA / L2VA.
# ---------------------------------------------------------------------------


class H3VideoPromptCreator(_OllamaMixin):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "idea": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the video idea in plain language…",
                    "tooltip": "The only creative input you need. The node builds the complete H3 timeline, camera language and audio fields from this idea and any connected frames.",
                },
            ),
            "duration": (
                "FLOAT",
                {"default": 6.0, "min": 0.1, "max": 120.0, "step": 0.1, "round": 0.01, "tooltip": "Target video duration. Used for final-frame alignment when a last frame is connected."},
            ),
        }
        optional = {
            "first_frame": (
                "IMAGE",
                {"tooltip": "Optional. Connect this to make the node automatically use I2VA (first-frame reference) or FL2VA if a last frame is also connected."},
            ),
            "last_frame": (
                "IMAGE",
                {"tooltip": "Optional. Connect this to make the node automatically use L2VA (last-frame reference) or FL2VA if a first frame is also connected."},
            ),
            "extra_instructions": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional: camera, dialogue, exact text, sound, style, constraints…",
                    "tooltip": "Everything beyond the main idea is optional. Leave blank and the model will fill in the missing cinematic/audio details itself.",
                },
            ),
        }
        optional.update(cls._ollama_inputs())
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "reference_analysis", "generation_notes")
    FUNCTION = "execute"
    CATEGORY = "H3 / Prompt Creator"
    DESCRIPTION = "Automatic H3 T2VA/I2VA/FL2VA/L2VA prompt creator. Connect a first frame, last frame, both, or neither."
    SEARCH_ALIASES = ["H3", "MiniMax H3", "T2VA", "I2VA", "FL2VA", "L2VA", "Prompt Creator", "Ollama"]

    def execute(
        self,
        idea: str,
        duration: float,
        first_frame=None,
        last_frame=None,
        extra_instructions: str = "",
        provider: str = OLLAMA,
        api_key: str = "",
        api_model: str = "",
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen3-vl:8b",
        temperature: float = 0.25,
        keep_alive: str = "10m",
        request_timeout: int = 600,
        max_output_tokens: int = 4096,
        num_ctx: int = 8192,
        enable_reference_analysis: bool = True,
        whisper_model: str = "small",
        whisper_device: str = "auto",
        enable_audio_transcription: bool = True,
    ):
        dur = _duration(duration)
        task = _video_task(first_frame, last_frame)
        cfg = BackendConfig(
            provider, ollama_url, ollama_model, api_key, api_model,
            temperature, keep_alive, int(request_timeout), max_output_tokens, num_ctx,
        )
        preflight_note = _preflight(cfg)
        images: List[str] = []
        first_images = _image_batch_to_b64(first_frame, max_images=1) if first_frame is not None else []
        last_images = _image_batch_to_b64(last_frame, max_images=1) if last_frame is not None else []
        images.extend(first_images)
        images.extend(last_images)
        ctx = _fit_num_ctx(num_ctx, len(images), max_output_tokens)

        # One unit per asset so each image is analyzed against its own label only.
        analysis_units: List[Dict[str, Any]] = []
        if first_frame is not None and first_images:
            analysis_units.append({
                "asset": {"asset_id": "<Picture 1>", "asset_type": "IMAGE", "role_hint": "first-frame anchor", "instruction": "Separate reusable subject identity from incidental background/pose/camera details."},
                "images": first_images,
            })
        if last_frame is not None and last_images:
            analysis_units.append({
                "asset": {"asset_id": "<Picture 2>" if first_frame is not None else "<Picture 1>", "asset_type": "IMAGE", "role_hint": "last-frame anchor", "instruction": "Separate reusable subject identity from incidental background/pose/camera details."},
                "images": last_images,
            })
        dossier = None
        if not cfg.is_deterministic and images and enable_reference_analysis:
            try:
                cfg.num_ctx = ctx
                dossier, analysis_note = _analyze_references(cfg, analysis_units)
            except Exception as exc:
                analysis_note = f"Reference analysis failed: {exc}"
        else:
            analysis_note = "Reference analysis skipped"

        payload = {
            "mode": task,
            "duration_seconds": dur,
            "timing_constraint": (
                f"The target video is exactly {_fmt_two(dur)} seconds long. Every shot timestamp "
                f"must be below {_fmt_two(dur)} seconds. Shot 1 has no timestamp. "
                f"Use at most {max(1, int(dur // 3) + 1)} shots."
            ),
            "idea": _clean(idea),
            "extra_instructions": _clean(extra_instructions),
            "first_frame_connected": first_frame is not None,
            "last_frame_connected": last_frame is not None,
            "reference_dossier": _reference_dossier_text(dossier),
        }
        user = (
            "Create the complete final H3 prompt. Do not explain reasoning. Return JSON only.\n"
            "The input may be only a simple idea; infer the missing scene, camera, action, speaker/audio and timing details.\n"
            "Use the reference dossier as the semantic separation layer. Do NOT inject a character-reference image's background, pose, framing or lighting into the target unless the task explicitly requires it as a frame anchor.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        schema = _json_schema(
            {
                "integrated_multimodal_description": {"type": "string"},
                "overall_soundscape": {"type": "string"},
                "non_diegetic_music": {"type": "string"},
            },
            ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"],
        )
        notes = [f"Auto mode: {task}", preflight_note, analysis_note]
        try:
            obj, provider_note = _generate_with_provider(
                cfg, VIDEO_SYSTEM, user, images, schema,
                required_keys=("integrated_multimodal_description",),
            )
            notes.append(provider_note)
            if images:
                notes.append(f"Frame evidence: {len(images)} image(s) sent to Ollama")
            if obj:
                return _render_video(obj, task, dur), _reference_dossier_text(dossier), _notes(notes)
        except Exception as exc:
            notes.append(str(exc))
            notes.append("Deterministic H3 fallback used.")

        prompt = _fallback_video(
            idea,
            extra_instructions,
            task,
            dur,
            first_desc="Reference image supplied; visible opening state is treated as ground truth." if first_frame is not None else "",
            last_desc="Reference image supplied; visible final state is treated as ground truth." if last_frame is not None else "",
        )
        return prompt, _reference_dossier_text(dossier), _notes(notes)


# ---------------------------------------------------------------------------
# Node 2 — full-reference, with IMAGE + VIDEO + AUDIO evidence.
# ---------------------------------------------------------------------------


class H3FullReferenceVideoPromptCreator(_OllamaMixin):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "idea": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "placeholder": "What should the target video be?",
                    "tooltip": "Simple target idea. The node uses it together with any connected reference images, video and audio to write the complete six-section H3 full-reference prompt.",
                },
            ),
            "target_duration": (
                "FLOAT",
                {"default": 6.0, "min": 0.1, "max": 120.0, "step": 0.1, "round": 0.01, "tooltip": "Target duration used to shape the generated timeline. If reference video duration is available, Ollama also receives it as evidence."},
            ),
        }
        optional = {
            "reference_video": (
                "VIDEO",
                {"tooltip": "Optional full reference video. Sampled frames are sent to Ollama; the source is represented as <Video 1> in the H3 prompt."},
            ),
            "reference_audio": (
                "AUDIO",
                {"tooltip": "Optional reference audio. Local faster-whisper transcribes speech/lyrics and librosa extracts audio/music features; exact user notes can override ambiguity."},
            ),
            "reference_image_1": ("IMAGE", {"tooltip": "Optional picture reference 1. Automatically represented as <Picture 1> unless Ollama determines another H3 reference role."}),
            "reference_image_2": ("IMAGE", {"tooltip": "Optional picture reference 2."}),
            "reference_image_3": ("IMAGE", {"tooltip": "Optional picture reference 3."}),
            "reference_image_4": ("IMAGE", {"tooltip": "Optional picture reference 4."}),
            "reference_image_5": ("IMAGE", {"tooltip": "Optional picture reference 5."}),
            "reference_image_6": ("IMAGE", {"tooltip": "Optional picture reference 6."}),
            "reference_intent": (
                ["Auto", "reference generation", "keyframe completion", "video editing", "video continuation", "audio reuse", "audio reference"],
                {"default": "Auto", "tooltip": "Optional hint. Auto is recommended: the node infers the task type from the connected references and notes."},
            ),
            "reference_notes": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional: who/what each reference represents, preserve/change instructions, exact dialogue, audio role…",
                    "tooltip": "Optional clarification. Leave blank when the references and simple idea are self-explanatory.",
                },
            ),
            "audio_transcript_or_notes": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional: exact dialogue/lyrics or audio reuse notes",
                    "tooltip": "Ollama itself is not assumed to hear arbitrary AUDIO tensors. Put exact lyrics/dialogue here when they must be preserved verbatim.",
                },
            ),
        }
        optional.update(cls._ollama_inputs())
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "reference_analysis", "audio_analysis", "generation_notes")
    FUNCTION = "execute"
    CATEGORY = "H3 / Prompt Creator"
    DESCRIPTION = "Multimodal H3 full-reference prompt creator with IMAGE, VIDEO and AUDIO inputs, Qwen3-VL reasoning, faster-whisper transcription and librosa/FFmpeg audio analysis."
    SEARCH_ALIASES = ["H3", "MiniMax H3", "Full Reference", "Reference to Video", "Video Reference", "Audio Reference", "Ollama"]

    def execute(
        self,
        idea: str,
        target_duration: float,
        reference_video=None,
        reference_audio=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_intent: str = "Auto",
        reference_notes: str = "",
        audio_transcript_or_notes: str = "",
        provider: str = OLLAMA,
        api_key: str = "",
        api_model: str = "",
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen3-vl:8b",
        temperature: float = 0.25,
        keep_alive: str = "10m",
        request_timeout: int = 600,
        max_output_tokens: int = 4096,
        num_ctx: int = 8192,
        enable_reference_analysis: bool = True,
        whisper_model: str = "small",
        whisper_device: str = "auto",
        enable_audio_transcription: bool = True,
    ):
        images: List[str] = []
        image_count = 0
        cfg = BackendConfig(
            provider, ollama_url, ollama_model, api_key, api_model,
            temperature, keep_alive, int(request_timeout), max_output_tokens, num_ctx,
        )
        preflight_note = _preflight(cfg)
        # Keep each image tied to its own label so the analyzer cannot describe
        # image 2's subject under <Picture 1>.
        per_image: List[List[str]] = []
        for img in [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5, reference_image_6]:
            if img is not None:
                encoded = _image_batch_to_b64(img, max_images=1)
                if not encoded:
                    continue
                image_count += 1
                per_image.append(encoded)
                images.extend(encoded)

        # Each sampled frame costs vision tokens; 4 is enough to read structure
        # and keeps the request inside a GPU-resident context window.
        video_frames, video_meta = _sample_video_frames(reference_video, max_frames=4) if reference_video is not None else ([], {})
        images.extend(video_frames)
        has_video = reference_video is not None
        extracted_video_audio = None
        if has_video:
            extracted_video_audio = extract_audio_from_video_source(_extract_video_source(reference_video))
        analysis_audio = reference_audio if reference_audio is not None else extracted_video_audio
        has_audio = analysis_audio is not None
        has_image = image_count > 0

        # An AUDIO input the user actually connected is a real audio task; a track
        # merely extracted from a reference video is not (guide 2.5).
        intent = _infer_reference_task(
            reference_intent, has_image, has_video, has_audio,
            f"{reference_notes}\n{audio_transcript_or_notes}",
            explicit_audio=reference_audio is not None,
        )
        audio_meta = _audio_metadata(analysis_audio)
        spectrogram = _audio_spectrogram_b64(analysis_audio) if analysis_audio is not None else None
        transcription = None
        if enable_audio_transcription and analysis_audio is not None:
            try:
                transcription = transcribe_audio(analysis_audio, model_size=whisper_model, device=whisper_device)
            except Exception as exc:
                transcription = {"available": False, "reason": str(exc)}

        asset_manifest: List[Dict[str, Any]] = []
        for i in range(image_count):
            # Use the H3 label as the id so the model never writes an internal
            # name such as "image_1" into the finished prompt.
            asset_manifest.append({"asset_id": f"<Picture {i+1}>", "asset_type": "IMAGE", "role_hint": "classify as reusable subject vs. concrete Picture anchor", "h3_candidate_label": f"Picture {i+1}"})
        if has_video:
            asset_manifest.append({"asset_id": "<Video 1>", "asset_type": "VIDEO", "role_hint": "separate reusable subjects from source-video motion/edit/continuation structure", "h3_candidate_label": "Video 1", "metadata": video_meta})
        if has_audio:
            asset_manifest.append({
                "asset_id": "<Audio 1>",
                "asset_type": "AUDIO",
                "role_hint": "classify audio as reuse vs reference; preserve exact transcript when supplied/detected",
                "h3_candidate_label": "Audio 1",
                "analysis": {"features": audio_meta, "transcription": transcription or {"available": False}},
            })

        ctx = _fit_num_ctx(num_ctx, len(images), max_output_tokens)
        dossier = None
        analysis_note = "Reference analysis skipped"
        cfg.num_ctx = ctx
        if not cfg.is_deterministic and enable_reference_analysis:
            try:
                units: List[Dict[str, Any]] = [
                    {"asset": asset_manifest[i], "images": per_image[i], "user_note": reference_notes}
                    for i in range(image_count)
                ]
                if has_video:
                    # All sampled frames belong to the one video asset.
                    units.append({
                        "asset": next(a for a in asset_manifest if a["asset_id"] == "<Video 1>"),
                        "images": video_frames,
                    })
                if has_audio:
                    units.append({
                        "asset": next(a for a in asset_manifest if a["asset_id"] == "<Audio 1>"),
                        "images": [],
                    })
                dossier, analysis_note = _analyze_references(cfg, units)
            except Exception as exc:
                analysis_note = f"Reference analysis failed: {exc}"

        target_dur = _duration(target_duration)
        payload = {
            "target_duration_seconds": target_dur,
            "timing_constraint": (
                f"The target video is exactly {_fmt_two(target_dur)} seconds long. Every shot "
                f"timestamp must be below {_fmt_two(target_dur)} seconds. Shot 1 has no timestamp. "
                f"Use at most {max(1, int(target_dur // 3) + 1)} shots."
            ),
            "idea": _clean(idea),
            "detailed_description_length": (
                "350-500 English words for this generation task. This is the main body of the "
                "output: describe every shot in playback order with composition, appearance, "
                "environment, lighting, action, camera movement, sound and dialogue. Do not "
                "compress it into a summary."
            ),
            "inferred_or_selected_task_type": intent,
            "reference_assets": asset_manifest,
            "reference_dossier": _reference_dossier_text(dossier),
            # Flat, unambiguous mapping. The generator previously swapped two
            # people between pictures; this is the binding it must obey.
            "which_asset_holds_which_subject": _asset_bindings(dossier) or "no analysis available",
            "reference_notes": _clean(reference_notes),
            "audio_transcript_or_notes": _clean(audio_transcript_or_notes),
            "audio_transcription": transcription or {"available": False, "reason": "No audio transcription run."},
            "video_metadata": video_meta,
            "audio_metadata": audio_meta,
            "audio_pipeline": {
                "speech_engine": f"faster-whisper:{whisper_model}",
                "music_feature_engine": "librosa",
                "embedded_video_audio_extraction": bool(extracted_video_audio),
            },
            # The guide forbids inventing a label for an asset that was not
            # supplied, and an ordinary reference video does not create an
            # <Audio N> just because the file has sound.
            "labels_you_may_use": (
                [f"<Picture {i + 1}> or <Subject n> from image {i + 1}" for i in range(image_count)]
                + (["<Video 1>"] if has_video else [])
                + (["<Audio 1>"] if has_audio else [])
            ) or ["no reference labels - this is a text-only target"],
            "labels_you_must_not_use": (
                ([] if has_audio else ["<Audio N> - no audio asset is connected"])
                + ([] if has_video else ["<Video N> - no video asset is connected"])
                + ([] if image_count else ["<Picture N> - no image asset is connected"])
            ),
            "visual_evidence_counts": {
                "picture_references": image_count,
                "sampled_video_frames": len(video_frames),
                "audio_features_available": bool(audio_meta.get("feature_analysis") or audio_meta.get("analysis_backend")),
                "audio_transcript_available": bool(transcription and transcription.get("text")),
            },
        }
        image_order = [f"image {i + 1} of {len(images)} = <Picture {i + 1}>" for i in range(image_count)]
        if video_frames:
            image_order.append(
                f"images {image_count + 1}-{image_count + len(video_frames)} = sampled frames of <Video 1>"
            )
        user = (
            "Create the complete final H3 full-reference prompt. Do not explain reasoning. Return JSON only.\n"
            + (
                "The attached images are supplied in this exact order — never mix them up: "
                + "; ".join(image_order)
                + ".\nThe reference dossier below was produced by analysing each asset separately; "
                "its per-asset descriptions are authoritative. If your own reading of an image "
                "disagrees with the dossier, follow the dossier.\n"
                "BEFORE writing subject_definitions, read 'which_asset_holds_which_subject'. "
                "Each <Subject N> must cite the picture that the dossier says its person came "
                "from. Attaching a person to the wrong picture makes the whole prompt wrong.\n"
                "Also obey reference_notes: when the user states which person is in which "
                "picture, that statement overrides everything else.\n"
                if image_order else ""
            )
            + "Use the supplied assets as evidence, not as vague inspiration. Never invent a reference asset that is not connected.\n"
            "CRITICAL: a character/object reference image is not automatically a <Picture N>. Only make it a standalone <Picture N> when it functions as a concrete frame/keyframe/composition anchor. Otherwise extract the reusable content as <Subject N> and explicitly exclude incidental background, pose, camera and lighting from the subject definition.\n"
            "The user's idea may be simple; infer all missing cinematic details needed for a complete H3 prompt.\n"
            "For reference video, distinguish source-video structure/continuation/editing from reusable subjects.\n"
            "For reference audio, do not invent exact speech or lyrics unless supplied in the notes or clearly available from an explicit transcript.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        schema = _json_schema(
            {
                "subject_definitions": {"type": "string"},
                "summary": {"type": "string"},
                "retention_analysis": {"type": "string"},
                "detailed_description": {"type": "string"},
                "overall_soundscape": {"type": "string"},
                "non_diegetic_music": {"type": "string"},
            },
            [
                "subject_definitions",
                "summary",
                "retention_analysis",
                "detailed_description",
                "overall_soundscape",
                "non_diegetic_music",
            ],
        )
        notes = [f"Task type: {intent}", preflight_note, analysis_note]
        try:
            obj, provider_note = _generate_with_provider(
                cfg, FULL_REF_SYSTEM, user, images, schema,
                required_keys=("summary", "retention_analysis", "detailed_description"),
                min_words=300,
            )
            notes.append(provider_note)
            if image_count:
                notes.append(f"Picture evidence: {image_count}")
            if video_frames:
                notes.append(f"Video evidence: {len(video_frames)} sampled frame(s)")
            if has_audio:
                notes.append("Audio evidence: faster-whisper transcript + librosa features + spectrogram")
                if transcription and transcription.get("text"):
                    notes.append(f"Transcription: {transcription.get('language') or 'unknown'} / {len(transcription.get('segments') or [])} segment(s)")
            if obj:
                rendered = _render_full_ref(obj, target_dur)
                rendered, label_warnings = _strip_unavailable_labels(
                    rendered, has_image, has_video, has_audio
                )
                notes.extend(label_warnings)
                return rendered, _reference_dossier_text(dossier), _audio_analysis_text(transcription, audio_meta), _notes(notes)
        except Exception as exc:
            notes.append(str(exc))
            notes.append("Deterministic H3 full-reference fallback used.")

        prompt = _fallback_full_ref(idea, intent, image_count, has_video, has_audio, reference_notes or audio_transcript_or_notes)
        return prompt, _reference_dossier_text(dossier), _audio_analysis_text(transcription, audio_meta), _notes(notes)


# ---------------------------------------------------------------------------
# ComfyUI registration.
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "H3VideoPromptCreator": H3VideoPromptCreator,
    "H3FullReferenceVideoPromptCreator": H3FullReferenceVideoPromptCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VideoPromptCreator": "H3 Video Prompt Creator",
    "H3FullReferenceVideoPromptCreator": "H3 Full-Reference Video Prompt Creator",
}

