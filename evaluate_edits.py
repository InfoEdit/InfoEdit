import os
import re
import json
import time
import hashlib
import random
import argparse
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from PIL import Image
try:                      # Vertex-only; unused on the openai-proxy branch
    from google import genai
except ImportError:       # pragma: no cover
    genai = None
try:
    import wandb
except ImportError:   # optional: only needed when W&B logging is enabled
    wandb = None

from llm_client import chat_text

DEFAULT_MODEL_PATH = "gemini-3.1-flash-lite-preview"

NUM_VARIANTS = 1

OPERATIONS = ("text_expand", "add", "swap_inter", "aspect_ratio")

# Per-operation metadata keys that pass through from the input JSONL produced
# by generate_editing_prompts.py into the evaluation result JSONL — purely for
# downstream stratified analysis / case study; not consumed by the prompts.
# Keys absent from a given record are silently skipped.
PASSTHROUGH_KEYS = {
    "add": ("add_count",),
    "swap_inter": ("swap_pairs",),
    "text_expand": ("text_expand_word_count", "text_expand_word_bucket"),
    "aspect_ratio": (
        "source_orientation", "source_aspect_ratio",
        "source_width", "source_height",
        "target_orientation", "target_orientations",
    ),
}


def _passthrough_metadata(operation: str, item: dict) -> dict:
    keys = PASSTHROUGH_KEYS.get(operation, ())
    return {k: item[k] for k in keys if k in item}


# Per-operation field used to stratify stats by difficulty. text_expand uses
# the bucket label (e.g. "11-15") rather than the raw word count so groups
# stay coarse enough to be statistically meaningful. aspect_ratio has no
# difficulty knob and is excluded from stratified reporting.
DIFFICULTY_KEY = {
    "add": "add_count",
    "swap_inter": "swap_pairs",
    "text_expand": "text_expand_word_bucket",
    "aspect_ratio": None,
}


def _difficulty_sort_key(val):
    """Sort numerically when possible; bucket labels like '11-15' sort by lo."""
    if isinstance(val, bool):
        return (1, str(val))
    if isinstance(val, (int, float)):
        return (0, float(val))
    s = str(val)
    m = re.match(r"\s*(\d+)", s)
    if m:
        return (0, float(m.group(1)))
    return (1, s)


def per_op_path(base: str, operation: str) -> str:
    stem, ext = os.path.splitext(base)
    return f"{stem}.{operation}{ext}" if ext else f"{base}.{operation}"


def edited_dir_for(base: str, operation: str) -> str:
    return os.path.join(base, operation)


VISUAL_AMBIGUITY_CAVEAT = """
[Visual-Ambiguity Caveat (CRITICAL — applies to EVERY text-related judgement in this prompt, including BOTH criterion 1 and criterion 2)]:
The evaluator (you) can itself mis-read characters when OCR-ing rendered text in images. To prevent false-positive 'text changed / rewritten / altered' verdicts, treat any text difference that can be explained by OCR-style ambiguity between visually similar glyphs as the SAME text. Bias toward judging text as PRESERVED — when in doubt, assume the text is UNCHANGED.

Common OCR-confusable pairs (non-exhaustive — apply the same principle to ANY visually similar glyph pair, not only those listed):
- Letters: l / I / i / 1 / | / j (any vertical-stroke shapes), O / 0 / o / Q / D, c / e / o, S / 5 / s, B / 8, G / 6 / b / C, Z / 2, q / 9 / g, U / V / Y / v, n / h, m / rn, w / vv, d / cl, b / h, e / a, a / o, s / e, t / l / f
- Digits: 0 / 8 / 9 / 6, 1 / 7, 3 / 8 / 5, 5 / 6, 2 / Z
- Punctuation: straight vs curly quotes, hyphen / en-dash / em-dash (- – —), period vs comma, three dots vs ellipsis (... …), colon vs semicolon
- Diacritics: e/é/è, a/á/à, n/ñ, c/ç, o/ó (accented and unaccented forms are interchangeable for OCR purposes)
- Case-only differences for letters whose upper/lower forms share the same shape: K/k, X/x, W/w, Z/z, S/s, C/c, O/o, P/p, V/v, U/u, M/m

Application rule (this OVERRIDES any 'text rewritten / changed / altered' criterion stated below — read carefully):
- A SINGLE-CHARACTER difference between an original token and the corresponding edited token must be treated as OCR ambiguity by default — EVEN WHEN the resulting string happens to spell a different real word in any language. ALL of the following are to be judged UNCHANGED:
  · "Status" vs "Statue" (final s↔e)         · "Stable" vs "Steble" (a↔e)
  · "Belt"   vs "Beit"   (l↔i)               · "Nobel"  vs "Nabel"  (o↔a)
  · "Awards" vs "Awarde" (final s↔e)         · "[SYS]"  vs "[8YS]"  (S↔8)
  · "2022"   vs "2822"   (0↔8)               · "200"    vs "208"    (0↔8)
  · "2020"   vs "2920"   (0↔9)
- Only flag text as TRULY altered when EITHER:
  (a) MULTIPLE characters differ in ways no single OCR slip can explain — e.g. "1933" → "SS3S" (three different chars all garbled), "Hello" → "World"; OR
  (b) the SEMANTIC MEANING has unambiguously and substantially changed beyond what OCR can account for — e.g. an entire word swapped for an unrelated one ("approve" → "reject"), an entire sentence inserted or deleted, a date moved by years (not by a single visually-similar digit), a brand name replaced by a different brand, a numerical value shifted by an order of magnitude.
- 'ENTIRE WORD changed' means a SUBSTANTIAL fraction of the word's characters differ (≥2 chars, or ≥half the word). It does NOT mean 'one character changed and the result happens to spell a different word'.
"""


PRESERVATION_CORE = (
    "2. Content Preservation: Every original element{exempt_clause} "
    "must remain intact and content-unchanged. Repositioning, resizing, "
    "proportional rescaling, and font-size adjustments ({reflow_examples}) "
    "are ALLOWED — these are normal reflow behaviors, not failures. "
    "What counts as failure: deletion of an original element, "
    "alteration of an element's text / icon / number / shading "
    "(subject to the Visual-Ambiguity Caveat declared at the top of this prompt), "
    "occlusion that hides content, cropping that pushes content off-canvas, "
    "or shrinking so aggressive that the text becomes illegible.{tail_note}"
)

PRESERVATION_SLOTS = {
    "text_expand": {
        "exempt_clause": " other than the target whose body text is being extended",
        "reflow_examples": "e.g. shrinking text to fit a tightened cell, and re-wrapping of non-target elements",
        "tail_note": " Scrutinize neighbors immediately adjacent to the target most aggressively — they are the most common victims of occlusion, displacement, or being pushed off-canvas when text is extended.",
    },
    "add": {
        "exempt_clause": "",
        "reflow_examples": "e.g. shrinking text to fit a smaller cell after reflow",
        "tail_note": "",
    },
    "swap_inter": {
        "exempt_clause": " other than the two named blocks (which exchange locations)",
        "reflow_examples": "e.g. shrinking text on surrounding blocks to fit after the swap",
        "tail_note": "",
    },
    "aspect_ratio": {
        "exempt_clause": "",
        "reflow_examples": "e.g. blocks re-stacked, re-flowed, or re-wrapped to fit the new canvas shape",
        "tail_note": " This criterion is judged strictly regardless of which track (LENIENT/STRICT) was selected for criterion 1.",
    },
}


def _preservation(operation: str) -> str:
    return PRESERVATION_CORE.format(**PRESERVATION_SLOTS[operation])


EVAL_TEXT_EXPAND_TEMPLATE = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to EXTEND the body text of ONE specific element by adding one or more additional sentences on the same topic (moderate extension — enough to overflow the original slot but still legible after reflow), and the edited image, determine whether the extension has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The element whose body text gets rewritten in the edited image must be exactly the one named in the instruction — same title, same anchor location, same numbering — and the target's title, label, and number must NOT change. AND the rewritten body text must be noticeably longer than the original — adding several additional short sentences on the same topic — but NOT turned into a dense paragraph. The extended text must be fully and legibly rendered inside the canvas — no truncation, no overflow beyond the canvas, no overlap with neighboring elements, no cropped lines, no text rendered outside its container. Targeting the wrong element, changing multiple elements' texts, trivial length changes, dense paragraph-length rewrites, off-topic rewrites, or visible overflow/clipping all count as failure.
""" + _preservation("text_expand") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. Pay special attention to whether the named element's body text has been moderately extended on-topic without truncation or overflow, and whether all other elements remain intact and fully visible — especially the neighbors of the target.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) verify the target element identity and that its rewritten body text is moderately longer (several short sentences added, on the same topic, not a dense paragraph) and fully rendered without truncation, overflow, or overlap; 2) verify all other content is preserved, with neighbors of the target especially checked for occlusion, displacement, or being pushed off-canvas>",
    "edit_compliance": "<success/failure>",
    "content_preservation": "<success/failure>",
    "overall_judgement": "<success/failure>"
}}
"""


EVAL_ADD_TEMPLATE = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to INSERT one or more new elements at a specified location, and the edited image, determine whether the insertion has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The new element(s) must be inserted at the exact position specified in the instruction — for example, if the instruction says "insert between A and B", the new element must appear precisely between A and B, not above A, not below B, not in any other location. AND the specified element(s) must be fully and correctly added as described. Any deviation in placement, or missing / incomplete / incorrect additions, count as failure.
""" + _preservation("add") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. Pay special attention to whether the new element is placed at the exact position specified in the instruction with the specified content.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) identify what the instruction asks to add and where exactly, then verify the new element exists in the edited image at the specified position with the specified content; 2) verify all original content is preserved>",
    "edit_compliance": "<success/failure>",
    "content_preservation": "<success/failure>",
    "overall_judgement": "<success/failure>"
}}
"""


EVAL_SWAP_INTER_TEMPLATE = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to SWAP two ENTIRE blocks/modules on the canvas, and the edited image, determine whether the swap has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The two named blocks/modules must have exchanged their locations on the canvas as whole units — Block A must now sit at Block B's original location, and vice versa. AND each block must carry ALL of its internal contents — titles, icons, text, numbers, shading — unchanged into its new location. Any partial move, single-block shift, location other than the swap, or any block that loses / gains / alters its internal content during the swap counts as failure.
""" + _preservation("swap_inter") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. Pay special attention to whether exactly the two named blocks have exchanged locations while carrying their internal contents intact.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) identify the two blocks/modules named in the instruction and verify each block is now at the other's original location with its internal content unchanged; 2) verify untouched blocks are preserved>",
    "edit_compliance": "<success/failure>",
    "content_preservation": "<success/failure>",
    "overall_judgement": "<success/failure>"
}}
"""


EVAL_ASPECT_RATIO_TEMPLATE = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to RE-RENDER the infographic in a new canvas aspect ratio (e.g. landscape ↔ portrait), and the edited image, determine whether the re-layout has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

[Source-Ratio-Dependent Track Selection]:
Before applying the criteria, estimate the ORIGINAL image's aspect ratio and pick the evaluation track:
- LENIENT track — original is approximately 1:1 (square): since forcing a square composition into a landscape / portrait frame inherently requires some geometric accommodation, mechanical uniform scaling, stretching, letterbox padding, or centered cropping are ACCEPTABLE ways to reach the target ratio. A full block-level re-layout is NOT required. Content Preservation (criterion 2) is still judged strictly.
- STRICT track — original is NOT 1:1 (already landscape or portrait): both criteria below must be met as written. Mechanical stretching, squeezing, cropping, or padding in place of a genuine re-layout is a failure.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The edited image's canvas must clearly match the TARGET aspect ratio specified in the instruction (e.g. 9:16 portrait, 16:9 landscape, 1:1 square) — a noticeable deviation from that shape, or leaving the canvas at the original aspect ratio, is a failure. AND every original element must remain within the edited canvas and fully readable — no text pushed off-canvas, no block missing, no content clipped by the frame edge.
   - STRICT track: in addition, blocks must be genuinely re-arranged — stacked, re-wrapped, resized, or reordered — to suit the new canvas shape; simply cropping or padding the original image to the new shape (without re-laying-out the blocks), or mechanically squeezed / stretched arrangements that cause overlaps, cropped text, or empty voids, are a failure.
   - LENIENT track: uniform scaling, stretching, letterbox padding, or centered cropping of the original 1:1 composition is fine even if it distorts geometry (e.g., circles become ovals, photos look widened). Only score failure here if content is actually lost or clipped off the frame.
""" + _preservation("aspect_ratio") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. First decide which track applies based on the original's aspect ratio, then apply criteria (a) Edit Compliance — canvas matches the target ratio AND every block fits / is readable in the new canvas under the selected track, (b) all original content preserved.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) estimate the ORIGINAL image's aspect ratio and declare whether the LENIENT (source ≈ 1:1) or STRICT (source non-1:1) track applies; 2) verify the edited image's canvas matches the target aspect ratio AND that every original block fits and remains readable in the new canvas — STRICT requires genuine re-arrangement, LENIENT accepts stretching / padding / cropping; 3) verify every original block's content is present and unmodified (strict regardless of track)>",
    "edit_compliance": "<success/failure>",
    "content_preservation": "<success/failure>",
    "overall_judgement": "<success/failure>"
}}
"""


EVAL_TEXT_EXPAND_TEMPLATE_DETAILED = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to EXTEND the body text of ONE specific element by adding one or more additional sentences on the same topic (moderate extension — enough to overflow the original slot but still legible after reflow), and the edited image, determine whether the extension has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The element whose body text gets rewritten in the edited image must be exactly the one named in the instruction — same title, same anchor location, same numbering — and the target's title, label, and number must NOT change. AND the rewritten body text must be noticeably longer than the original — adding several additional short sentences on the same topic — but NOT turned into a dense paragraph. The extended text must be fully and legibly rendered inside the canvas — no truncation, no overflow beyond the canvas, no overlap with neighboring elements, no cropped lines, no text rendered outside its container. Targeting the wrong element, changing multiple elements' texts, trivial length changes, dense paragraph-length rewrites, off-topic rewrites, or visible overflow/clipping all count as failure.
""" + _preservation("text_expand") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. Pay special attention to whether the named element's body text has been moderately extended on-topic without truncation or overflow, and whether all other elements remain intact and fully visible — especially the neighbors of the target.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) verify the target element identity and that its rewritten body text is moderately longer (several short sentences added, on the same topic, not a dense paragraph) and fully rendered without truncation, overflow, or overlap; 2) verify all other content is preserved, with neighbors of the target especially checked for occlusion, displacement, or being pushed off-canvas>",
    "edit_compliance": "<success/failure>",
    "edit_compliance_details": {{
        "correct_position": "<success/failure>",
        "content_fully_rendered": "<success/failure>"
    }},
    "content_preservation": "<success/failure>",
    "content_preservation_details": {{
        "elements_preserved": "<success/failure>",
        "content_unchanged": "<success/failure>",
        "visual_style_preserved": "<success/failure>",
        "no_occlusion": "<success/failure>",
        "structural_integrity": "<success/failure>"
    }},
    "overall_judgement": "<success/failure>"
}}

[Sub-Field Definitions]:
edit_compliance_details apply ONLY to the target element being extended (named in the instruction):
  - correct_position: success iff the rewritten body text is on EXACTLY the target element named in the instruction (correct identity match — not a different element, not multiple elements at once).
  - content_fully_rendered: success iff the extended body text is fully and legibly visible inside the canvas — no truncation, no overflow beyond the frame, no clipped lines, no text rendered outside its container.
content_preservation_details apply ONLY to all OTHER original elements (everything except the target being extended):
  - elements_preserved: success iff every other original element appears FULLY and COMPLETELY inside the edited canvas — none has been deleted, none has been pushed entirely outside the canvas frame (so it is no longer visible), and none has been partially cropped at the canvas edge. Any of those three failure modes (deletion, pushed-off, edge-cropped) makes this field failure.
  - content_unchanged: success iff no other element's text / numbers / labels were rewritten. The Visual-Ambiguity Caveat at the top of this prompt OVERRIDES this check — a single-character OCR-similar difference does NOT count as rewriting, even if the result spells a different real word; flag failure only when ≥2 characters differ or the semantic meaning has unambiguously changed.
  - visual_style_preserved: success iff no other element's icon / logo / color / typography was restyled (the content may be unchanged but the visual style was changed).
  - no_occlusion: success iff no other element is covered or hidden by another element.
  - structural_integrity: success iff there are no stranded connectors, no orphaned labels, no broken or skipped numbering sequences, and no empty visual voids.

[Consistency Rule]:
- If ANY sub-field under edit_compliance_details is failure, edit_compliance must be failure. edit_compliance may additionally be failure due to aspects of criterion 1 not broken out into sub-fields (e.g., rewritten text is the wrong length or off-topic).
- If ANY sub-field under content_preservation_details is failure, content_preservation must be failure.
- overall_judgement = edit_compliance AND content_preservation.
"""


EVAL_ADD_TEMPLATE_DETAILED = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to INSERT one or more new elements at a specified location, and the edited image, determine whether the insertion has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The new element(s) must be inserted at the exact position specified in the instruction — for example, if the instruction says "insert between A and B", the new element must appear precisely between A and B, not above A, not below B, not in any other location. AND the specified element(s) must be fully and correctly added as described. Any deviation in placement, or missing / incomplete / incorrect additions, count as failure.
""" + _preservation("add") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. Pay special attention to whether the new element is placed at the exact position specified in the instruction with the specified content.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) identify what the instruction asks to add and where exactly, then verify the new element exists in the edited image at the specified position with the specified content; 2) verify all original content is preserved>",
    "edit_compliance": "<success/failure>",
    "edit_compliance_details": {{
        "correct_position": "<success/failure>",
        "content_fully_rendered": "<success/failure>"
    }},
    "content_preservation": "<success/failure>",
    "content_preservation_details": {{
        "elements_preserved": "<success/failure>",
        "content_unchanged": "<success/failure>",
        "visual_style_preserved": "<success/failure>",
        "no_occlusion": "<success/failure>",
        "structural_integrity": "<success/failure>"
    }},
    "overall_judgement": "<success/failure>"
}}

[Sub-Field Definitions]:
edit_compliance_details apply ONLY to the new element being inserted:
  - correct_position: success iff the new element is inserted at the EXACT relative position specified in the instruction (e.g., "between A and B" must literally place the new element between A and B, not above A, not below B).
  - content_fully_rendered: success iff the new element's content (icons, labels, numbers, etc.) is complete as described AND fully visible inside the canvas (not clipped, not pushed off-frame, not partially rendered). When checking that the rendered text matches what the instruction asked to add, apply the Visual-Ambiguity Caveat at the top of this prompt — a single-character OCR-similar mismatch (e.g. "Belt" rendered as "Beit") does NOT count as a content failure here.
content_preservation_details apply ONLY to all OTHER original elements (everything except the newly inserted one):
  - elements_preserved: success iff every original element appears FULLY and COMPLETELY inside the edited canvas — none has been deleted, none has been pushed entirely outside the canvas frame (so it is no longer visible), and none has been partially cropped at the canvas edge. Any of those three failure modes (deletion, pushed-off, edge-cropped) makes this field failure.
  - content_unchanged: success iff no original element's text / numbers / labels were rewritten. The Visual-Ambiguity Caveat at the top of this prompt OVERRIDES this check — a single-character OCR-similar difference does NOT count as rewriting, even if the result spells a different real word; flag failure only when ≥2 characters differ or the semantic meaning has unambiguously changed.
  - visual_style_preserved: success iff no original element's icon / logo / color / typography was restyled.
  - no_occlusion: success iff no original element is covered or hidden by another element.
  - structural_integrity: success iff there are no stranded connectors, no orphaned labels, no broken or skipped numbering sequences, and no empty visual voids.

[Consistency Rule]:
- If ANY sub-field under edit_compliance_details is failure, edit_compliance must be failure. edit_compliance may additionally be failure due to aspects of criterion 1 not broken out into sub-fields (e.g., the new element does not match sibling style, or violates the fractional-numbering rule when applicable).
- If ANY sub-field under content_preservation_details is failure, content_preservation must be failure.
- overall_judgement = edit_compliance AND content_preservation.
"""


EVAL_SWAP_INTER_TEMPLATE_DETAILED = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to SWAP two ENTIRE blocks/modules on the canvas, and the edited image, determine whether the swap has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The two named blocks/modules must have exchanged their locations on the canvas as whole units — Block A must now sit at Block B's original location, and vice versa. AND each block must carry ALL of its internal contents — titles, icons, text, numbers, shading — unchanged into its new location. Any partial move, single-block shift, location other than the swap, or any block that loses / gains / alters its internal content during the swap counts as failure.
""" + _preservation("swap_inter") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. Pay special attention to whether exactly the two named blocks have exchanged locations while carrying their internal contents intact.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) identify the two blocks/modules named in the instruction and verify each block is now at the other's original location with its internal content unchanged; 2) verify untouched blocks are preserved>",
    "edit_compliance": "<success/failure>",
    "edit_compliance_details": {{
        "correct_position": "<success/failure>",
        "content_intact": "<success/failure>"
    }},
    "content_preservation": "<success/failure>",
    "content_preservation_details": {{
        "elements_preserved": "<success/failure>",
        "content_unchanged": "<success/failure>",
        "visual_style_preserved": "<success/failure>",
        "no_occlusion": "<success/failure>",
        "structural_integrity": "<success/failure>"
    }},
    "overall_judgement": "<success/failure>"
}}

[Sub-Field Definitions]:
edit_compliance_details apply ONLY to the two blocks named in the instruction (the swap targets):
  - correct_position: success iff the two named blocks have actually exchanged their canvas locations as whole units (Block A now sits at Block B's original location, AND Block B now sits at Block A's original location).
  - content_intact: success iff each swapped block carried ALL of its internal contents (titles, icons, text, numbers, shading, visual style) UNCHANGED into its new location. The Visual-Ambiguity Caveat at the top of this prompt OVERRIDES the text-unchanged portion of this check — single-character OCR-similar differences inside the swapped block's text do NOT count as content alteration.
content_preservation_details apply ONLY to all OTHER original elements (everything except the two swapped blocks):
  - elements_preserved: success iff every other original element appears FULLY and COMPLETELY inside the edited canvas — none has been deleted, none has been pushed entirely outside the canvas frame (so it is no longer visible), and none has been partially cropped at the canvas edge. Any of those three failure modes (deletion, pushed-off, edge-cropped) makes this field failure.
  - content_unchanged: success iff no other element's text / numbers / labels were rewritten. The Visual-Ambiguity Caveat at the top of this prompt OVERRIDES this check — a single-character OCR-similar difference does NOT count as rewriting, even if the result spells a different real word; flag failure only when ≥2 characters differ or the semantic meaning has unambiguously changed.
  - visual_style_preserved: success iff no other element's icon / logo / color / typography was restyled.
  - no_occlusion: success iff no other element is covered or hidden by another element.
  - structural_integrity: success iff there are no stranded connectors, no orphaned labels, no broken or skipped numbering sequences, and no empty visual voids.

[Consistency Rule]:
- If ANY sub-field under edit_compliance_details is failure, edit_compliance must be failure.
- If ANY sub-field under content_preservation_details is failure, content_preservation must be failure.
- overall_judgement = edit_compliance AND content_preservation.
"""


EVAL_ASPECT_RATIO_TEMPLATE_DETAILED = """
Role: You are a strict image editing quality evaluator.

Objective: Given an original image, an editing instruction that asks to RE-RENDER the infographic in a new canvas aspect ratio (e.g. landscape ↔ portrait), and the edited image, determine whether the re-layout has been successfully implemented. Your judgement must be binary: either success or failure, with no middle ground.

[Source-Ratio-Dependent Track Selection]:
Before applying the criteria, estimate the ORIGINAL image's aspect ratio and pick the evaluation track:
- LENIENT track — original is approximately 1:1 (square): since forcing a square composition into a landscape / portrait frame inherently requires some geometric accommodation, mechanical uniform scaling, stretching, letterbox padding, or centered cropping are ACCEPTABLE ways to reach the target ratio. A full block-level re-layout is NOT required. Content Preservation (criterion 2) is still judged strictly.
- STRICT track — original is NOT 1:1 (already landscape or portrait): both criteria below must be met as written. Mechanical stretching, squeezing, cropping, or padding in place of a genuine re-layout is a failure.

""" + VISUAL_AMBIGUITY_CAVEAT + """
[Evaluation Criteria (BOTH must be met for success)]:
1. Edit Compliance: The edited image's canvas must clearly match the TARGET aspect ratio specified in the instruction (e.g. 9:16 portrait, 16:9 landscape, 1:1 square) — a noticeable deviation from that shape, or leaving the canvas at the original aspect ratio, is a failure. AND every original element must remain within the edited canvas and fully readable — no text pushed off-canvas, no block missing, no content clipped by the frame edge.
   - STRICT track: in addition, blocks must be genuinely re-arranged — stacked, re-wrapped, resized, or reordered — to suit the new canvas shape; simply cropping or padding the original image to the new shape (without re-laying-out the blocks), or mechanically squeezed / stretched arrangements that cause overlaps, cropped text, or empty voids, are a failure.
   - LENIENT track: uniform scaling, stretching, letterbox padding, or centered cropping of the original 1:1 composition is fine even if it distorts geometry (e.g., circles become ovals, photos look widened). Only score failure here if content is actually lost or clipped off the frame.
""" + _preservation("aspect_ratio") + """

[Editing Instruction]:
{instruction}

Please compare the original image (first image) with the edited image (second image) carefully. First decide which track applies based on the original's aspect ratio, then apply criteria (a) Edit Compliance — canvas matches the target ratio AND every block fits / is readable in the new canvas under the selected track, (b) all original content preserved.

Output in the following JSON format strictly (no other text):
{{
    "reasoning": "<step-by-step analysis: 1) estimate the ORIGINAL image's aspect ratio and declare whether the LENIENT (source ≈ 1:1) or STRICT (source non-1:1) track applies; 2) verify the edited image's canvas matches the target aspect ratio AND that every original block fits and remains readable in the new canvas — STRICT requires genuine re-arrangement, LENIENT accepts stretching / padding / cropping; 3) verify every original block's content is present and unmodified (strict regardless of track)>",
    "edit_compliance": "<success/failure>",
    "content_preservation": "<success/failure>",
    "content_preservation_details": {{
        "elements_preserved": "<success/failure>",
        "content_unchanged": "<success/failure>",
        "visual_style_preserved": "<success/failure>",
        "no_occlusion": "<success/failure>",
        "structural_integrity": "<success/failure>"
    }},
    "overall_judgement": "<success/failure>"
}}

[Sub-Field Definitions]:
edit_compliance has NO sub-fields for aspect_ratio (the criterion is conditional on the LENIENT/STRICT track and resists clean decomposition); judge it as a single binary using criterion 1 above.
content_preservation_details apply to ALL original blocks/elements (every block in the original infographic — there is no "target" element to exclude in aspect_ratio, since the operation re-renders the whole canvas):
  - elements_preserved: success iff every original block appears FULLY and COMPLETELY inside the edited canvas — none has been deleted, none has been pushed entirely outside the canvas frame (so it is no longer visible), and none has been partially cropped at the canvas edge. This is the most common failure mode for aspect_ratio: if even one block has any of those three problems (deletion, pushed-off, edge-cropped), this field is failure.
  - content_unchanged: success iff no original block's text / numbers / labels were rewritten. The Visual-Ambiguity Caveat at the top of this prompt OVERRIDES this check — a single-character OCR-similar difference does NOT count as rewriting, even if the result spells a different real word; flag failure only when ≥2 characters differ or the semantic meaning has unambiguously changed.
  - visual_style_preserved: success iff no original block's icon / logo / color palette / typography was restyled (the content may be unchanged but the visual style was changed). Mechanical stretching of geometry (e.g., circles becoming ovals on the LENIENT track) does NOT count as a style change here.
  - no_occlusion: success iff no original block is covered or hidden by another block. Especially scrutinize the STRICT track for mechanical-squeeze artifacts where re-layout produced overlapping blocks.
  - structural_integrity: success iff there are no stranded connectors, no orphaned labels, no broken or skipped numbering sequences, and no empty visual voids introduced by the re-layout. Re-layout commonly breaks arrow / line connections between blocks — check for these.

[Consistency Rule]:
- edit_compliance is judged directly from criterion 1 (no sub-fields to roll up).
- If ANY sub-field under content_preservation_details is failure, content_preservation must be failure.
- overall_judgement = edit_compliance AND content_preservation.
"""


def get_eval_template(operation: str, detailed: bool = False) -> str:
    if detailed:
        if operation == "text_expand":
            return EVAL_TEXT_EXPAND_TEMPLATE_DETAILED
        if operation == "add":
            return EVAL_ADD_TEMPLATE_DETAILED
        if operation == "swap_inter":
            return EVAL_SWAP_INTER_TEMPLATE_DETAILED
        if operation == "aspect_ratio":
            return EVAL_ASPECT_RATIO_TEMPLATE_DETAILED
    if operation == "text_expand":
        return EVAL_TEXT_EXPAND_TEMPLATE
    if operation == "add":
        return EVAL_ADD_TEMPLATE
    if operation == "swap_inter":
        return EVAL_SWAP_INTER_TEMPLATE
    if operation == "aspect_ratio":
        return EVAL_ASPECT_RATIO_TEMPLATE
    raise ValueError(f"Unknown operation: {operation}")


def load_prompts(file_path, prompt_index=0):
    """Load editing prompts. Accepts new schema (``generated_edit_prompts`` list)
    and legacy schema (``generated_edit_prompt`` string). For the list schema,
    picks ``prompt_index`` (default 0 = first prompt) and normalizes it onto
    ``generated_edit_prompt`` so downstream code stays unchanged.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("image_path"):
                continue

            prompts_list = item.get("generated_edit_prompts")
            if isinstance(prompts_list, list) and prompts_list:
                idx = prompt_index if prompt_index < len(prompts_list) else 0
                item["generated_edit_prompt"] = str(prompts_list[idx]).strip()
                # For aspect_ratio: also slice the per-variant target_orientation
                # so the result JSONL records the specific target this run used.
                target_orientations = item.get("target_orientations")
                if isinstance(target_orientations, list) and target_orientations:
                    t_idx = idx if idx < len(target_orientations) else 0
                    item["target_orientation"] = target_orientations[t_idx]
            elif item.get("generated_edit_prompt"):
                item["generated_edit_prompt"] = str(item["generated_edit_prompt"]).strip()
            else:
                continue

            if not item["generated_edit_prompt"]:
                continue
            data.append(item)
    return data


def find_edited_images(output_dir, item_id):
    edited = []
    for var_idx in range(1, NUM_VARIANTS + 1):
        path = os.path.join(output_dir, f"{item_id}_edited_{var_idx}.png")
        if os.path.exists(path):
            edited.append((var_idx, path))
    return edited


def make_comparison_image(original_path, edited_path, label_height=30):
    """Concatenate original and edited images side-by-side with labels."""
    from PIL import ImageDraw, ImageFont

    original = Image.open(original_path).convert("RGB")
    edited = Image.open(edited_path).convert("RGB")

    target_h = max(original.height, edited.height)
    def _resize(img):
        if img.height == target_h:
            return img
        new_w = int(img.width * target_h / img.height)
        return img.resize((new_w, target_h), Image.LANCZOS)
    original = _resize(original)
    edited = _resize(edited)

    gap = 10
    total_w = original.width + edited.width + gap
    total_h = target_h + label_height
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    canvas.paste(original, (0, label_height))
    canvas.paste(edited, (original.width + gap, label_height))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((original.width // 2 - 30, 5), "Original", fill=(0, 0, 0), font=font)
    draw.text((original.width + gap + edited.width // 2 - 20, 5), "Edited", fill=(0, 0, 0), font=font)
    return canvas


def parse_response(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "reasoning": response_text,
            "edit_compliance": "unknown",
            "content_preservation": "unknown",
            "overall_judgement": "unknown",
        }


def load_existing_keys(output_file: str) -> set:
    """Return set of (id, variant) that have already been evaluated."""
    if not os.path.exists(output_file):
        return set()
    keys = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                keys.add((str(item.get("id")), int(item.get("variant", -1))))
            except (json.JSONDecodeError, ValueError):
                continue
    return keys


def _coerce_id(item_id):
    try:
        return int(item_id)
    except (TypeError, ValueError):
        return item_id


def _write_skip_record(fout, item_id, variant, original_path, edited_path, instruction, reason, metadata=None):
    result = {
        "id": _coerce_id(item_id),
        "variant": variant,
        "original_image": original_path or "",
        "edited_image": edited_path or "",
        "instruction": instruction or "",
    }
    if metadata:
        result.update(metadata)
    result["evaluation"] = {
        "reasoning": reason,
        "edit_compliance": "skip",
        "content_preservation": "skip",
        "overall_judgement": "skip",
    }
    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    fout.flush()


def evaluate_shard(clients, model_path, data_list, edited_dir, shard_output_file, done_keys, operation, worker_id=0, detailed=False):
    max_retries = 10
    success_count = 0
    fail_count = 0
    skip_count = 0

    desc = f"Worker-{worker_id}" if worker_id else "Evaluating Edits"
    with open(shard_output_file, "a", encoding="utf-8") as fout:
        for item in tqdm(data_list, desc=desc, position=worker_id):
            item_id = str(item.get("id"))
            original_path = item.get("image_path")
            instruction = item.get("generated_edit_prompt")
            metadata = _passthrough_metadata(operation, item)

            if (item_id, -1) in done_keys:
                continue

            if not original_path or not os.path.exists(original_path):
                tqdm.write(f"[Warning] Original image not found for ID {item_id}: {original_path}")
                _write_skip_record(fout, item_id, -1, original_path, "", instruction,
                                   f"Original image not found: {original_path}", metadata=metadata)
                skip_count += 1
                continue

            try:
                original_image = Image.open(original_path)
                if original_image.mode != 'RGB':
                    original_image = original_image.convert('RGB')
            except Exception as e:
                tqdm.write(f"[Error] Failed to open original image for ID {item_id}: {e}")
                _write_skip_record(fout, item_id, -1, original_path, "", instruction,
                                   f"Failed to open original image: {e}", metadata=metadata)
                skip_count += 1
                continue

            edited_images = find_edited_images(edited_dir, item_id)
            if not edited_images:
                tqdm.write(f"[Warning] No edited images found for ID {item_id}")
                _write_skip_record(fout, item_id, -1, original_path, "", instruction,
                                   "No edited images found", metadata=metadata)
                skip_count += 1
                continue

            for var_idx, edited_path in edited_images:
                if (item_id, var_idx) in done_keys:
                    continue

                try:
                    edited_image = Image.open(edited_path)
                    if edited_image.mode != 'RGB':
                        edited_image = edited_image.convert('RGB')
                except Exception as e:
                    tqdm.write(f"[Error] Failed to open edited image {edited_path}: {e}")
                    _write_skip_record(fout, item_id, var_idx, original_path, edited_path, instruction,
                                       f"Failed to open edited image: {e}", metadata=metadata)
                    skip_count += 1
                    continue

                prompt = get_eval_template(operation, detailed=detailed).format(instruction=instruction)

                response_text = ""
                api_success = False
                for attempt in range(max_retries):
                    try:
                        response_text = chat_text(
                            model_path,
                            prompt,
                            images=[original_image, edited_image],
                        )

                        if not response_text.strip():
                            tqdm.write(f"[Warning] ID {item_id} V{var_idx}: Empty response, retrying ({attempt + 1}/{max_retries})...")
                            time.sleep(2)
                            continue

                        api_success = True
                        break
                    except Exception as e:
                        tqdm.write(f"[Error] ID {item_id} V{var_idx} (Attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(5)

                if not api_success:
                    tqdm.write(f"[Skip] ID {item_id} V{var_idx}: failed after {max_retries} retries")
                    _write_skip_record(fout, item_id, var_idx, original_path, edited_path, instruction,
                                       f"API failed after {max_retries} retries", metadata=metadata)
                    skip_count += 1
                    time.sleep(2)
                    continue

                evaluation = parse_response(response_text)

                result = {
                    "id": _coerce_id(item_id),
                    "variant": var_idx,
                    "original_image": original_path,
                    "edited_image": edited_path,
                    "instruction": instruction,
                    **metadata,
                    "evaluation": evaluation,
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                if evaluation.get("overall_judgement") == "success":
                    success_count += 1
                else:
                    fail_count += 1

                time.sleep(2)

    return success_count, fail_count, skip_count


def _worker_entry(worker_id, api_keys, model_path, data_shard, edited_dir, shard_output_file, done_keys, operation, result_queue, detailed=False):
    try:
        clients = [None]  # unused: llm_client holds the proxy client
        success, fail, skip = evaluate_shard(
            clients=clients,
            model_path=model_path,
            data_list=data_shard,
            edited_dir=edited_dir,
            shard_output_file=shard_output_file,
            done_keys=done_keys,
            operation=operation,
            worker_id=worker_id,
            detailed=detailed,
        )
        result_queue.put((worker_id, success, fail, skip))
    except Exception as e:
        print(f"[Worker-{worker_id}] Fatal error: {e}")
        result_queue.put((worker_id, 0, 0, len(data_shard)))


def run_parallel(num_workers, api_keys, model_path, data, edited_dir, output_file, done_keys, operation, detailed=False):
    shards = [[] for _ in range(num_workers)]
    for i, item in enumerate(data):
        shards[i % num_workers].append(item)

    shard_files = [f"{output_file}.part{wid}" for wid in range(num_workers)]
    for sf in shard_files:
        if os.path.exists(sf):
            os.remove(sf)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for wid in range(num_workers):
        if not shards[wid]:
            continue
        p = ctx.Process(
            target=_worker_entry,
            args=(wid, api_keys, model_path, shards[wid], edited_dir, shard_files[wid], done_keys, operation, result_queue, detailed),
        )
        p.start()
        processes.append(p)

    total_success = 0
    total_fail = 0
    total_skip = 0
    for _ in processes:
        wid, s, f, sk = result_queue.get()
        total_success += s
        total_fail += f
        total_skip += sk

    for p in processes:
        p.join()

    with open(output_file, "a", encoding="utf-8") as fout:
        for sf in shard_files:
            if not os.path.exists(sf):
                continue
            with open(sf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(sf)

    return total_success, total_fail, total_skip


def _batch_state_name(state) -> str:
    """Return a stable string state name regardless of SDK enum vs raw string."""
    return getattr(state, "name", None) or str(state)


def _guess_mime(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _gcs_safe_name(local_path: str) -> str:
    """Build a collision-resistant GCS object name for a local file path."""
    h = hashlib.sha1(os.path.abspath(local_path).encode("utf-8")).hexdigest()[:10]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(local_path))
    return f"{h}_{base}"


def process_evaluate_batch(
    project_id,
    location,
    bucket_uri,
    model_path,
    data_list,
    edited_dir,
    output_file,
    done_keys,
    operation,
    poll_interval=30,
    detailed=False,
):
    """Submit all evaluation requests as a single Vertex AI Gemini batch prediction job.

    Each request ships the eval prompt plus two images (original + edited), both
    uploaded to GCS under ``{bucket_uri}/batch_eval_images_{ts}/``. Responses are
    matched back to their source (item_id, variant) by the
    ``(prompt_text, orig_uri, edit_uri)`` tuple, which is unique per variant.

    Auth uses Application Default Credentials — run `gcloud auth application-default login`
    beforehand. No API key is used in this path.
    """
    import fsspec  # lazy import — only needed in batch mode (requires gcsfs)
    from google.genai.types import CreateBatchJobConfig

    bucket_uri = bucket_uri.rstrip("/")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    client = genai.Client(vertexai=True, project=project_id, location=location)
    fs = fsspec.filesystem("gcs")

    images_gcs_dir = f"{bucket_uri}/batch_eval_images_{timestamp}"
    input_jsonl_uri = f"{bucket_uri}/batch_eval_input_{timestamp}.jsonl"
    output_prefix = f"{bucket_uri}/batch_eval_output_{timestamp}"

    items_by_key = {}  # (prompt, orig_uri, edit_uri) -> (item_id, var_idx, orig_path, edit_path, instruction)
    uploaded = {}      # local_path -> gcs_uri (cache — original appears N times for N variants)
    skip_records = []  # (item_id, variant, orig_path, edit_path, instruction, reason)

    def upload_image(local_path):
        if local_path in uploaded:
            return uploaded[local_path]
        if local_path.startswith("gs://"):
            uploaded[local_path] = local_path
            return local_path
        gcs_uri = f"{images_gcs_dir}/{_gcs_safe_name(local_path)}"
        with open(local_path, "rb") as src, fs.open(gcs_uri, "wb") as dst:
            dst.write(src.read())
        uploaded[local_path] = gcs_uri
        return gcs_uri

    def _has_pending(item):
        iid = str(item.get("id"))
        if (iid, -1) in done_keys:
            return False
        return any((iid, v) not in done_keys for v in range(1, NUM_VARIANTS + 1))

    pending_data = [it for it in data_list if _has_pending(it)]
    already_done = len(data_list) - len(pending_data)
    print(f"[Batch] Preparing requests for {len(pending_data)} items "
          f"({already_done} already evaluated, skipped)...")
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    with fs.open(input_jsonl_uri, "w", encoding="utf-8") as fout:
        for item in tqdm(pending_data, desc="Uploading & staging"):
            item_id = str(item.get("id"))
            original_path = item.get("image_path")
            instruction = item.get("generated_edit_prompt")
            metadata = _passthrough_metadata(operation, item)

            if (item_id, -1) in done_keys:
                continue

            if not original_path or not os.path.exists(original_path):
                skip_records.append((item_id, -1, original_path, "", instruction,
                                     f"Original image not found: {original_path}", metadata))
                continue

            edited_images = find_edited_images(edited_dir, item_id)
            if not edited_images:
                skip_records.append((item_id, -1, original_path, "", instruction,
                                     "No edited images found", metadata))
                continue

            pending_variants = [(v, p) for v, p in edited_images if (item_id, v) not in done_keys]
            if not pending_variants:
                continue

            try:
                orig_uri = upload_image(original_path)
            except Exception as e:
                skip_records.append((item_id, -1, original_path, "", instruction,
                                     f"Failed to upload original: {e}", metadata))
                continue
            orig_mime = _guess_mime(original_path)

            for var_idx, edited_path in pending_variants:
                try:
                    edit_uri = upload_image(edited_path)
                except Exception as e:
                    skip_records.append((item_id, var_idx, original_path, edited_path, instruction,
                                         f"Failed to upload edited: {e}", metadata))
                    continue
                edit_mime = _guess_mime(edited_path)

                prompt = get_eval_template(operation, detailed=detailed).format(instruction=instruction)
                key = (prompt, orig_uri, edit_uri)
                items_by_key[key] = (item_id, var_idx, original_path, edited_path, instruction, metadata)

                payload = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {"file_data": {"file_uri": orig_uri, "mime_type": orig_mime}},
                                    {"file_data": {"file_uri": edit_uri, "mime_type": edit_mime}},
                                ],
                            }
                        ],
                    }
                }
                fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1

    # Flush skip records to the output file immediately.
    skip_count = 0
    if skip_records:
        with open(output_file, "a", encoding="utf-8") as f_skip:
            for rec in skip_records:
                _write_skip_record(f_skip, *rec)
                skip_count += 1

    if written == 0:
        print("[Batch] No eval requests to submit (everything skipped or already done).")
        return 0, 0, skip_count

    print(f"[Batch] Submitting batch job (model={model_path}, dest={output_prefix})")
    batch_job = client.batches.create(
        model=model_path,
        src=input_jsonl_uri,
        config=CreateBatchJobConfig(dest=output_prefix),
    )
    print(f"[Batch] Job name: {batch_job.name}")

    running_states = {"JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"}
    while _batch_state_name(batch_job.state) in running_states:
        print(f"[Batch] state={_batch_state_name(batch_job.state)}; sleeping {poll_interval}s...")
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=batch_job.name)

    final_state = _batch_state_name(batch_job.state)
    if final_state != "JOB_STATE_SUCCEEDED":
        print(f"[Batch] Job {batch_job.name} failed: state={final_state} error={batch_job.error}")
        # Record every staged request as a skip so downstream stats are consistent.
        with open(output_file, "a", encoding="utf-8") as f_skip:
            for key, (item_id, var_idx, orig_path, edit_path, instruction, metadata) in items_by_key.items():
                _write_skip_record(f_skip, item_id, var_idx, orig_path, edit_path, instruction,
                                   f"Batch job {batch_job.name} failed: {final_state}", metadata=metadata)
                skip_count += 1
        return 0, 0, skip_count

    dest_uri = batch_job.dest.gcs_uri
    pred_paths = fs.glob(f"{dest_uri}/*/predictions.jsonl")
    if not pred_paths:
        print(f"[Batch] No predictions.jsonl found under {dest_uri}")
        with open(output_file, "a", encoding="utf-8") as f_skip:
            for key, (item_id, var_idx, orig_path, edit_path, instruction, metadata) in items_by_key.items():
                _write_skip_record(f_skip, item_id, var_idx, orig_path, edit_path, instruction,
                                   "No predictions.jsonl produced", metadata=metadata)
                skip_count += 1
        return 0, 0, skip_count

    success = 0
    fail = 0
    seen_keys = set()
    with open(output_file, "a", encoding="utf-8") as fout_res:
        for pred_path in pred_paths:
            with fs.open(f"gs://{pred_path}", "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[Batch] Skipping malformed line: {e}")
                        continue

                    # Rebuild the request key (prompt, orig_uri, edit_uri).
                    prompt_text = None
                    file_uris = []
                    try:
                        parts = row["request"]["contents"][0]["parts"]
                        for p in parts:
                            if p.get("text"):
                                prompt_text = p["text"]
                            fd = p.get("file_data") or p.get("fileData")
                            if fd:
                                uri = fd.get("file_uri") or fd.get("fileUri")
                                if uri:
                                    file_uris.append(uri)
                    except (KeyError, IndexError, TypeError):
                        pass

                    if not prompt_text or len(file_uris) < 2:
                        print("[Batch] Skipping row without a recognizable request")
                        continue

                    key = (prompt_text, file_uris[0], file_uris[1])
                    matched = items_by_key.get(key)
                    if not matched:
                        print("[Batch] Could not match response back to any input item")
                        continue
                    item_id, var_idx, orig_path, edit_path, instruction, metadata = matched
                    seen_keys.add(key)

                    status = row.get("status", "") or ""
                    response = row.get("response") or {}
                    text_buf = ""
                    for cand in response.get("candidates", []) or []:
                        for part in (cand.get("content") or {}).get("parts", []) or []:
                            if part.get("text"):
                                text_buf += part["text"]

                    if not text_buf.strip():
                        _write_skip_record(fout_res, item_id, var_idx, orig_path, edit_path, instruction,
                                           f"Empty response (status={status})", metadata=metadata)
                        skip_count += 1
                        continue

                    evaluation = parse_response(text_buf)
                    result = {
                        "id": _coerce_id(item_id),
                        "variant": var_idx,
                        "original_image": orig_path,
                        "edited_image": edit_path,
                        "instruction": instruction,
                        **metadata,
                        "evaluation": evaluation,
                    }
                    fout_res.write(json.dumps(result, ensure_ascii=False) + "\n")

                    if evaluation.get("overall_judgement") == "success":
                        success += 1
                    else:
                        fail += 1

        # Items that were staged but never came back in predictions.
        missing = set(items_by_key) - seen_keys
        for key in missing:
            item_id, var_idx, orig_path, edit_path, instruction, metadata = items_by_key[key]
            _write_skip_record(fout_res, item_id, var_idx, orig_path, edit_path, instruction,
                               "No row in predictions.jsonl", metadata=metadata)
            skip_count += 1

    if missing:
        missing_ids = sorted({items_by_key[k][0] for k in missing})
        print(f"[Batch] {len(missing)} items had no row in predictions.jsonl: {missing_ids[:10]}...")

    return success, fail, skip_count


def sort_output_file(output_file: str) -> None:
    """Sort evaluation_results.jsonl in place by (id asc, variant asc)."""
    if not os.path.exists(output_file):
        return
    items = []
    tail = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                tail.append(stripped)
                continue
            try:
                id_key = int(str(obj.get("id")))
            except (TypeError, ValueError):
                id_key = float("inf")
            try:
                var_key = int(obj.get("variant", -1))
            except (TypeError, ValueError):
                var_key = -1
            items.append(((id_key, var_key), obj))
    items.sort(key=lambda x: x[0])
    with open(output_file, "w", encoding="utf-8") as f:
        for _, obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for line in tail:
            f.write(line + "\n")


SUB_METRICS = ("edit_compliance", "content_preservation")

# Sub-fields under *_details objects produced by detailed-mode templates.
# Standard-mode records don't have *_details; their detail counters stay 0/0
# and are silently skipped from reporting.
EDIT_COMPLIANCE_DETAIL_FIELDS = {
    "text_expand": ("correct_position", "content_fully_rendered"),
    "add": ("correct_position", "content_fully_rendered"),
    "swap_inter": ("correct_position", "content_intact"),
    "aspect_ratio": (),  # detailed variant exists, but edit_compliance is intentionally flat (track-conditional, resists clean decomposition)
}

CONTENT_PRESERVATION_DETAIL_FIELDS = (
    "elements_preserved",
    "content_unchanged",
    "visual_style_preserved",
    "no_occlusion",
    "structural_integrity",
)


def compute_stats_from_file(output_file, operation=None):
    success = 0
    fail = 0
    skip = 0
    sub_success = {m: 0 for m in SUB_METRICS}
    sub_total = {m: 0 for m in SUB_METRICS}

    ec_fields = EDIT_COMPLIANCE_DETAIL_FIELDS.get(operation, ()) if operation else ()
    cp_fields = CONTENT_PRESERVATION_DETAIL_FIELDS
    detail_success = {
        "edit_compliance": {f: 0 for f in ec_fields},
        "content_preservation": {f: 0 for f in cp_fields},
    }
    detail_total = {
        "edit_compliance": {f: 0 for f in ec_fields},
        "content_preservation": {f: 0 for f in cp_fields},
    }

    if not os.path.exists(output_file):
        return success, fail, skip, sub_success, sub_total, detail_success, detail_total
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            evaluation = row.get("evaluation", {}) or {}
            judgement = evaluation.get("overall_judgement", "")
            if judgement == "success":
                success += 1
            elif judgement == "skip":
                skip += 1
            else:
                fail += 1
            if judgement == "skip":
                continue
            for m in SUB_METRICS:
                v = evaluation.get(m, "")
                if v in ("success", "failure"):
                    sub_total[m] += 1
                    if v == "success":
                        sub_success[m] += 1

            # Detail sub-fields (only present in detailed-mode records).
            ec_details = evaluation.get("edit_compliance_details") or {}
            for fld in ec_fields:
                v = ec_details.get(fld, "")
                if v in ("success", "failure"):
                    detail_total["edit_compliance"][fld] += 1
                    if v == "success":
                        detail_success["edit_compliance"][fld] += 1
            cp_details = evaluation.get("content_preservation_details") or {}
            for fld in cp_fields:
                v = cp_details.get(fld, "")
                if v in ("success", "failure"):
                    detail_total["content_preservation"][fld] += 1
                    if v == "success":
                        detail_success["content_preservation"][fld] += 1
    return success, fail, skip, sub_success, sub_total, detail_success, detail_total


def compute_stats_by_difficulty(output_file, operation):
    """Stratify the same stats as compute_stats_from_file by per-record difficulty.

    Returns {difficulty_value: bucket_dict} where bucket_dict has keys
    success / fail / skip / sub_success / sub_total / detail_success / detail_total.
    Returns {} for operations without a difficulty knob (aspect_ratio) or when
    the output file is missing / empty.
    """
    diff_key = DIFFICULTY_KEY.get(operation)
    if not diff_key:
        return {}

    ec_fields = EDIT_COMPLIANCE_DETAIL_FIELDS.get(operation, ())
    cp_fields = CONTENT_PRESERVATION_DETAIL_FIELDS

    def _new_bucket():
        return {
            "success": 0, "fail": 0, "skip": 0,
            "sub_success": {m: 0 for m in SUB_METRICS},
            "sub_total":   {m: 0 for m in SUB_METRICS},
            "detail_success": {
                "edit_compliance":      {f: 0 for f in ec_fields},
                "content_preservation": {f: 0 for f in cp_fields},
            },
            "detail_total": {
                "edit_compliance":      {f: 0 for f in ec_fields},
                "content_preservation": {f: 0 for f in cp_fields},
            },
        }

    buckets = {}
    if not os.path.exists(output_file):
        return buckets
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if diff_key not in row:
                continue
            diff_val = row[diff_key]
            bucket = buckets.setdefault(diff_val, _new_bucket())
            evaluation = row.get("evaluation", {}) or {}
            judgement = evaluation.get("overall_judgement", "")
            if judgement == "success":
                bucket["success"] += 1
            elif judgement == "skip":
                bucket["skip"] += 1
            else:
                bucket["fail"] += 1
            if judgement == "skip":
                continue
            for m in SUB_METRICS:
                v = evaluation.get(m, "")
                if v in ("success", "failure"):
                    bucket["sub_total"][m] += 1
                    if v == "success":
                        bucket["sub_success"][m] += 1
            ec_details = evaluation.get("edit_compliance_details") or {}
            for fld in ec_fields:
                v = ec_details.get(fld, "")
                if v in ("success", "failure"):
                    bucket["detail_total"]["edit_compliance"][fld] += 1
                    if v == "success":
                        bucket["detail_success"]["edit_compliance"][fld] += 1
            cp_details = evaluation.get("content_preservation_details") or {}
            for fld in cp_fields:
                v = cp_details.get(fld, "")
                if v in ("success", "failure"):
                    bucket["detail_total"]["content_preservation"][fld] += 1
                    if v == "success":
                        bucket["detail_success"]["content_preservation"][fld] += 1
    return buckets


def format_difficulty_breakdown_lines(operation, stats_by_diff):
    """Render a human-readable per-difficulty breakdown shared by stdout and the .txt file."""
    if not stats_by_diff:
        return []
    diff_key = DIFFICULTY_KEY.get(operation)
    lines = [f"[{operation} by difficulty ({diff_key})]"]
    for dv in sorted(stats_by_diff.keys(), key=_difficulty_sort_key):
        b = stats_by_diff[dv]
        s, f_, sk = b["success"], b["fail"], b["skip"]
        denom = s + f_ + sk
        rate = f"{s / denom:.4f}" if denom > 0 else "n/a"
        lines.append(f"  {diff_key}={dv}: success={s}, fail={f_}, skip={sk}, rate={rate} (n={denom})")
        sub_parts = []
        for m in SUB_METRICS:
            ss, st = b["sub_success"][m], b["sub_total"][m]
            r = f"{ss / st:.4f}" if st > 0 else "n/a"
            sub_parts.append(f"{m}_rate={r} ({ss}/{st})")
        lines.append(f"    {' | '.join(sub_parts)}")
        for category in ("edit_compliance", "content_preservation"):
            populated = [(fld, b["detail_success"][category][fld], b["detail_total"][category][fld])
                         for fld in b["detail_total"][category]
                         if b["detail_total"][category][fld] > 0]
            if not populated:
                continue
            parts = [f"{fld}={(s_v / t_v):.4f} ({s_v}/{t_v})" for fld, s_v, t_v in populated]
            lines.append(f"    {category}_details: {' | '.join(parts)}")
    return lines


def upload_to_wandb(args, per_op_plan, total_tasks, success, fail, skip):
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_path": args.model_path,
            "input_file": args.input_file,
            "edited_dir": args.edited_dir,
            "output_file": args.output_file,
            "operations": list(per_op_plan.keys()),
            "input_files": [p["input_file"] for p in per_op_plan.values()],
            "edited_dirs": [p["edited_dir"] for p in per_op_plan.values()],
            "output_files": [p["output_file"] for p in per_op_plan.values()],
            "limit": args.limit,
            "num_workers": args.num_workers,
            "total_tasks": total_tasks,
        },
    )

    denom = success + fail + skip
    log_payload = {
        "total_variants": denom,
        "success": success,
        "failed": fail,
        "skipped": skip,
        "success_rate": success / denom if denom > 0 else 0,
    }
    difficulty_table = wandb.Table(columns=[
        "operation", "difficulty_key", "difficulty_value", "n",
        "success", "failed", "skipped", "success_rate",
        "edit_compliance_rate", "content_preservation_rate",
    ])
    for op, plan in per_op_plan.items():
        s, f, sk, sub_s, sub_t, det_s, det_t = compute_stats_from_file(plan["output_file"], operation=op)
        d = s + f + sk
        log_payload[f"{op}/success"] = s
        log_payload[f"{op}/failed"] = f
        log_payload[f"{op}/skipped"] = sk
        log_payload[f"{op}/success_rate"] = s / d if d > 0 else 0
        for m in SUB_METRICS:
            log_payload[f"{op}/{m}_rate"] = sub_s[m] / sub_t[m] if sub_t[m] > 0 else 0
        # Detail sub-field rates (only logged when there are detailed-mode rows).
        for category in ("edit_compliance", "content_preservation"):
            for fld, total in det_t[category].items():
                if total > 0:
                    log_payload[f"{op}/{category}_details/{fld}_rate"] = det_s[category][fld] / total

        # Per-difficulty stratified scalars + table rows.
        diff_stats = compute_stats_by_difficulty(plan["output_file"], op)
        diff_key = DIFFICULTY_KEY.get(op)
        for dv in sorted(diff_stats.keys(), key=_difficulty_sort_key):
            b = diff_stats[dv]
            ds, df_, dsk = b["success"], b["fail"], b["skip"]
            dn = ds + df_ + dsk
            prefix = f"{op}/by_difficulty/{dv}"
            log_payload[f"{prefix}/total"] = dn
            log_payload[f"{prefix}/success"] = ds
            log_payload[f"{prefix}/failed"] = df_
            log_payload[f"{prefix}/skipped"] = dsk
            log_payload[f"{prefix}/success_rate"] = ds / dn if dn > 0 else 0
            sub_rates = {}
            for m in SUB_METRICS:
                rate = b["sub_success"][m] / b["sub_total"][m] if b["sub_total"][m] > 0 else 0
                log_payload[f"{prefix}/{m}_rate"] = rate
                sub_rates[m] = rate
            for category in ("edit_compliance", "content_preservation"):
                for fld, total in b["detail_total"][category].items():
                    if total > 0:
                        log_payload[f"{prefix}/{category}_details/{fld}_rate"] = (
                            b["detail_success"][category][fld] / total
                        )
            difficulty_table.add_data(
                op, diff_key, str(dv), dn, ds, df_, dsk,
                ds / dn if dn > 0 else 0,
                sub_rates["edit_compliance"], sub_rates["content_preservation"],
            )
    run.log(log_payload)
    run.log({"difficulty_breakdown": difficulty_table})

    # Rebuild a W&B Table from each per-op output file so images can be previewed.
    wandb_table = wandb.Table(columns=[
        "operation", "id", "variant", "original_image", "edited_image",
        "instruction", "reasoning",
        "edit_compliance", "content_preservation", "overall_judgement",
    ])
    placeholder = Image.new("RGB", (1, 1), (255, 255, 255))
    for op, plan in per_op_plan.items():
        out_path = plan["output_file"]
        if not os.path.exists(out_path):
            continue
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = row.get("evaluation", {})
                orig_path = row.get("original_image", "")
                edit_path = row.get("edited_image", "")
                try:
                    orig_img = wandb.Image(orig_path) if orig_path and os.path.exists(orig_path) else wandb.Image(placeholder)
                    edit_img = wandb.Image(edit_path) if edit_path and os.path.exists(edit_path) else wandb.Image(placeholder)
                    wandb_table.add_data(
                        op,
                        str(row.get("id", "")),
                        int(row.get("variant", -1)),
                        orig_img,
                        edit_img,
                        row.get("instruction", ""),
                        ev.get("reasoning", ""),
                        ev.get("edit_compliance", ""),
                        ev.get("content_preservation", ""),
                        ev.get("overall_judgement", ""),
                    )
                except Exception as e:
                    print(f"[W&B] Skipping row due to error: {e}")
    run.log({"evaluation_results": wandb_table})

    artifact = wandb.Artifact(
        name="edit-evaluations",
        type="evaluation",
        description=f"Edit evaluation results ({success} success, {fail} fail, {skip} skip)",
    )
    for plan in per_op_plan.values():
        if os.path.exists(plan["output_file"]):
            artifact.add_file(os.path.abspath(plan["output_file"]))
    run.log_artifact(artifact)

    run.finish()
    print(f"W&B run uploaded: {run.url}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate edited infographics using Gemini")
    parser.add_argument("--api_keys", type=str, nargs="+", default=None,
                        help="One or more Google API Keys (randomly selected per request). "
                             "Required for online mode; ignored when --use_batch is set.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Gemini multimodal model ID")
    parser.add_argument("--input_file", type=str, default="editing_prompts.jsonl",
                        help="Base path of the editing-prompt file. The per-operation file is "
                             "derived as <stem>.<operation><ext> (e.g. editing_prompts.add.jsonl).")
    parser.add_argument("--edited_dir", type=str, default="edited_infographics",
                        help="Base directory holding edited images. Per-operation images are read "
                             "from <edited_dir>/<operation>/ (e.g. edited_infographics/add/).")
    parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl",
                        help="Base path for evaluation results. Per-operation results are written to "
                             "<stem>.<operation><ext> (e.g. evaluation_results.add.jsonl).")
    parser.add_argument("--operation", type=str, default="add",
                        choices=["text_expand", "add", "swap_inter", "aspect_ratio", "all"],
                        help="Which editing operation's edits to evaluate. 'all' iterates through every "
                             "operation, using per-op input files, edited dirs, and output files.")
    parser.add_argument("--prompt_index", type=int, default=0,
                        help="When the input file has a list under 'generated_edit_prompts', "
                             "use the prompt at this index (default 0 = first).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N examples")
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-evaluate even if (id, variant) already present in output_file")
    parser.add_argument("--dry_run", action="store_true", help="Print evaluation prompts without calling the API")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel subprocess workers")
    parser.add_argument("--use_batch", action="store_true",
                        help="Use Vertex AI Gemini Batch Prediction (async, GCS-staged). "
                             "Requires --gcp_project and --batch_bucket_uri; uses ADC auth (no api_key). "
                             "Ignores --api_keys / --num_workers.")
    parser.add_argument("--gcp_project", type=str, default=None,
                        help="GCP project ID for Vertex AI batch prediction (required with --use_batch)")
    parser.add_argument("--gcp_location", type=str, default="us-central1",
                        help="GCP location for Vertex AI batch prediction")
    parser.add_argument("--batch_bucket_uri", type=str, default=None,
                        help="Cloud Storage bucket URI (e.g. gs://my-bucket) used to stage batch input/output. "
                             "Bucket should be in --gcp_location (typically us-central1).")
    parser.add_argument("--batch_poll_interval", type=int, default=30,
                        help="Seconds between batch job status polls")
    parser.add_argument("--wandb_project", type=str, default="edit-evaluation", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--detailed", action="store_true",
                        help="Use detailed evaluation prompts that ask the LLM to additionally output "
                             "diagnostic sub-fields under edit_compliance_details and content_preservation_details. "
                             "Only text_expand / add / swap_inter have detailed variants; aspect_ratio falls back "
                             "to the standard prompt. Output filenames are suffixed with .detailed to avoid mixing.")
    return parser.parse_args()


def main():
    args = parse_args()

    if getattr(args, "use_batch", False):
        raise SystemExit(
            "--use_batch is a Vertex AI feature and is not available on the "
            "openai-proxy branch. Drop --use_batch and raise --num_workers "
            "instead; requests then go to the proxy concurrently."
        )

    operations = list(OPERATIONS) if args.operation == "all" else [args.operation]

    per_op_plan = {}  # op -> {"input_file", "edited_dir", "output_file", "data", "done_keys"}
    for op in operations:
        input_file = per_op_path(args.input_file, op)
        edited_dir = edited_dir_for(args.edited_dir, op)
        output_file = per_op_path(args.output_file, op)
        if args.detailed:
            stem, ext = os.path.splitext(output_file)
            output_file = f"{stem}.detailed{ext}"

        if not os.path.exists(input_file):
            print(f"[{op}] Input file not found: {input_file} — skipping this operation")
            continue
        if not os.path.isdir(edited_dir):
            print(f"[{op}] Edited dir not found: {edited_dir} — skipping this operation")
            continue

        data = load_prompts(input_file, prompt_index=args.prompt_index)
        for item in data:
            item["operation"] = op

        if args.limit is not None and args.limit > 0:
            data = data[:args.limit]
            print(f"[{op}] Limited to first {args.limit} tasks")

        done_keys = set() if args.no_skip else load_existing_keys(output_file)
        if done_keys:
            print(f"[{op}] Will skip {len(done_keys)} (id, variant) pairs already in {output_file}")

        per_op_plan[op] = {
            "input_file": input_file,
            "edited_dir": edited_dir,
            "output_file": output_file,
            "data": data,
            "done_keys": done_keys,
        }

    total_tasks = sum(len(p["data"]) for p in per_op_plan.values())
    if total_tasks == 0:
        print("Nothing to do.")
        return

    print("-" * 50)
    print(f"Mode:         {'batch (Vertex AI)' if args.use_batch else 'online'}{' [DETAILED]' if args.detailed else ''}")
    print(f"Model:        {args.model_path}")
    print(f"Operations:   {', '.join(per_op_plan.keys())}")
    print(f"Total Tasks:  {total_tasks}")
    for op, plan in per_op_plan.items():
        print(f"  - [{op}] {len(plan['data'])} tasks | edited={plan['edited_dir']}/ | out={plan['output_file']}")
    print(f"Prompt Index: {args.prompt_index}")
    if args.use_batch:
        print(f"GCP Project:   {args.gcp_project}")
        print(f"GCP Location:  {args.gcp_location}")
        print(f"Bucket URI:    {args.batch_bucket_uri}")
        print(f"Poll Interval: {args.batch_poll_interval}s")
    else:
        print(f"Num Workers:  {args.num_workers}")
        print(f"Proxy:        {os.environ.get('OPENAI_BASE_URL', '(OPENAI_BASE_URL unset)')}")
    print("-" * 50)

    if args.dry_run:
        for op, plan in per_op_plan.items():
            print(f"\n[Dry Run] operation={op} — first 3 evaluation prompts:\n")
            for item in plan["data"][:3]:
                print(f"--- ID {item.get('id')} | {item.get('image_path')} ---")
                print(get_eval_template(op, detailed=args.detailed).format(instruction=item.get("generated_edit_prompt", "")))
                print()
        return

    if args.use_batch:
        if not args.gcp_project:
            print("Error: --use_batch requires --gcp_project")
            return
        if not args.batch_bucket_uri:
            print("Error: --use_batch requires --batch_bucket_uri (e.g. gs://my-bucket)")
            return

    total_success = 0
    total_fail = 0
    total_skip = 0
    for op, plan in per_op_plan.items():
        data = plan["data"]
        if not data:
            continue
        output_file = plan["output_file"]
        edited_dir = plan["edited_dir"]
        done_keys = plan["done_keys"]
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
        print(f"\n=== Evaluating operation: {op} ({len(data)} tasks) → {output_file} ===")

        if args.use_batch:
            s, f, sk = process_evaluate_batch(
                project_id=args.gcp_project,
                location=args.gcp_location,
                bucket_uri=args.batch_bucket_uri,
                model_path=args.model_path,
                data_list=data,
                edited_dir=edited_dir,
                output_file=output_file,
                done_keys=done_keys,
                operation=op,
                poll_interval=args.batch_poll_interval,
                detailed=args.detailed,
            )
        elif args.num_workers and args.num_workers > 1:
            s, f, sk = run_parallel(
                num_workers=args.num_workers,
                api_keys=args.api_keys,
                model_path=args.model_path,
                data=data,
                edited_dir=edited_dir,
                output_file=output_file,
                done_keys=done_keys,
                operation=op,
                detailed=args.detailed,
            )
        else:
            clients = [None]  # unused: llm_client holds the proxy client
            s, f, sk = evaluate_shard(
                clients=clients,
                model_path=args.model_path,
                data_list=data,
                edited_dir=edited_dir,
                shard_output_file=output_file,
                done_keys=done_keys,
                operation=op,
                worker_id=0,
                detailed=args.detailed,
            )
        total_success += s
        total_fail += f
        total_skip += sk

        sort_output_file(output_file)

    print(f"\n[Runtime counters] success={total_success}, fail={total_fail}, skip={total_skip}")

    grand_success = 0
    grand_fail = 0
    grand_skip = 0
    grand_sub_success = {m: 0 for m in SUB_METRICS}
    grand_sub_total = {m: 0 for m in SUB_METRICS}
    print("\n[Final stats per operation]")
    for op, plan in per_op_plan.items():
        s, f, sk, sub_s, sub_t, det_s, det_t = compute_stats_from_file(plan["output_file"], operation=op)
        denom = s + f + sk
        rate = f"{s / denom:.4f}" if denom > 0 else "n/a"

        # Build the lines once, then both print (with terminal indentation)
        # and write to a sister .txt file (clean, unindented).
        txt_lines = [f"[{op}] success={s}, fail={f}, skip={sk}, rate={rate}"]
        sub_parts = []
        for m in SUB_METRICS:
            r = f"{sub_s[m] / sub_t[m]:.4f}" if sub_t[m] > 0 else "n/a"
            sub_parts.append(f"{m}_rate={r} ({sub_s[m]}/{sub_t[m]})")
        txt_lines.append(" | ".join(sub_parts))
        for category in ("edit_compliance", "content_preservation"):
            populated = [(fld, det_s[category][fld], det_t[category][fld])
                         for fld in det_t[category] if det_t[category][fld] > 0]
            if not populated:
                continue
            label = f"{category}_details"
            parts = [f"{fld}={(s_v / t_v):.4f} ({s_v}/{t_v})" for fld, s_v, t_v in populated]
            txt_lines.append(f"{label}: {' | '.join(parts)}")

        # Per-difficulty stratified breakdown (no-op for aspect_ratio).
        diff_stats = compute_stats_by_difficulty(plan["output_file"], op)
        diff_lines = format_difficulty_breakdown_lines(op, diff_stats)

        # Terminal: keep the original indented layout.
        print(f"  - {txt_lines[0]} ({plan['output_file']})")
        for line in txt_lines[1:]:
            print(f"        {line}")
        for line in diff_lines:
            print(f"        {line}")

        # Sister .txt file (same stem as the per-op jsonl).
        txt_path = os.path.splitext(plan["output_file"])[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as ftxt:
            ftxt.write(f"output_file: {plan['output_file']}\n")
            for line in txt_lines:
                ftxt.write(line + "\n")
            if diff_lines:
                ftxt.write("\n")
                for line in diff_lines:
                    ftxt.write(line + "\n")

        grand_success += s
        grand_fail += f
        grand_skip += sk
        for m in SUB_METRICS:
            grand_sub_success[m] += sub_s[m]
            grand_sub_total[m] += sub_t[m]
    grand_denom = grand_success + grand_fail + grand_skip
    print(f"\n[Overall] success={grand_success}/{grand_denom}, fail={grand_fail}, skip={grand_skip}")
    if grand_denom > 0:
        print(f"[Overall] success rate: {grand_success / grand_denom:.4f}")
    for m in SUB_METRICS:
        if grand_sub_total[m] > 0:
            print(f"[Overall] {m}_rate: {grand_sub_success[m] / grand_sub_total[m]:.4f} ({grand_sub_success[m]}/{grand_sub_total[m]})")
        else:
            print(f"[Overall] {m}_rate: n/a (0/0)")

    if not args.no_wandb and wandb is None:

        raise SystemExit("W&B logging requested but wandb is not installed — pip install wandb, or pass --no_wandb.")

    if not args.no_wandb:
        upload_to_wandb(args, per_op_plan, total_tasks, grand_success, grand_fail, grand_skip)


if __name__ == "__main__":
    main()
