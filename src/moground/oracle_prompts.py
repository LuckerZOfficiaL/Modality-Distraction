"""Prompt templates for the Sonnet-as-oracle passes.

Every task is executed by a Claude Code subagent reading images via the Read tool
and writing JSONL output to a result file. Prompts here are the instructional
payload given to each agent.
"""
from __future__ import annotations

import json
from textwrap import dedent

# ----- generation (step 2) -----

GENERATION_SYSTEM = dedent(
    """
    You are an oracle building multiple-choice questions for a vision-language model
    study on modality preference. For each image you see, you must produce TWO
    4-option multiple-choice questions, each grounded in exactly one modality.
    """
).strip()


GENERATION_INSTRUCTIONS = dedent(
    """
    The original_caption you are given is a DENSE descriptive paragraph (often
    several sentences) that already enumerates many visible attributes — colors,
    counts, clothing, materials, spatial layout, background objects. Design both
    questions with that in mind: a generic perceptual question (e.g. "what color
    is the shirt") will usually fail the modality-isolation test on dense
    captions because the caption already states the answer.

    For the image you are given, together with its original caption, produce the
    following JSON object:

    {
      "vision": {
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": <0|1|2|3>,
        "rationale": "brief reason answer is only in the image"
      },
      "text": {
        "augmented_caption": "<original caption> + one short sentence adding a fact not visible in the image",
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": <0|1|2|3>,
        "rationale": "brief reason answer is only in the augmented caption"
      }
    }

    Rules for the VISION-grounded MCQ:
    - The answer must be obtainable ONLY from the image. The dense caption must NOT
      already state or strongly imply the answer.
    - Productive targets on dense captions: exact text written on signs / labels /
      packaging, brand names visible but not transcribed, fine-grained counts of
      items the caption describes only generically, micro-expressions, exact shade
      vs. caption's coarse color word, items present in the background that the
      caption omits, exact spatial relations not stated, gestures, gaze direction,
      reflections, lighting / time-of-day cues, weather.
    - Before finalizing, RE-READ the original caption: if your candidate question's
      answer is even loosely derivable from the caption text, pick a different
      detail.
    - Distractors must be plausible under the caption but wrong given the image.

    Rules for the TEXT-grounded MCQ:
    - Pick a category ORTHOGONAL to what the dense caption already covers. The
      caption likely already enumerates physical attributes; do NOT invent a fact
      about a physical attribute (size, color, count, shape, material) — choose
      from non-physical categories: year / date, city or country, person's name,
      institution or brand owning the scene, the photographer's intent, what
      happened immediately before or after, a quoted price, a sensor reading,
      a backstory detail, a relationship between people, an upcoming event.
    - Insert that fact into the original caption WHEREVER it reads most
      naturally. It does NOT need to be a standalone sentence appended at
      the end — and SHOULD NOT default to that, because length / position
      become tells. Acceptable forms (pick whichever fits the prose best):
        * a single word inserted as an apposition: "there is a man" → "there is a man, John,"
        * a parenthetical: "the bridge" → "the bridge (built in 1937)"
        * a short inline modifier: "the painting" → "the 1503 painting"
        * a relative clause: "the woman" → "the woman, who works for Reuters,"
        * a separate sentence (only if no inline form fits)
      The original caption's existing content must be preserved (minimal
      grammatical adjustments to integrate the fact are fine; do not remove
      or rewrite existing facts). Vary placement (start / middle / end) and
      vary form across seeds — do NOT systematically append at the tail.
    - The question's answer must be exactly that injected fact, obtainable from
      the augmented caption alone and NOT inferable from the image.
    - Distractors must be plausible alternatives of the same category as the
      correct fact.

    MODALITY-AGNOSTIC PHRASING (MANDATORY for BOTH questions):
    - The question MUST NOT contain any of these words or close synonyms:
      "image", "picture", "photo", "photograph", "caption", "text", "description",
      "shown", "depicted", "visible", "pictured", "displayed", "according to",
      "based on", "in the [image|caption|text|photo|picture]".
    - Phrase the question as if asking about the underlying situation, subject, scene,
      or event itself, not about a medium. Read alone (without the image and without
      the caption), a human reader must NOT be able to tell which modality holds the
      answer.
      GOOD: "What color is the dog's collar?", "In what year did this take place?",
            "How many people are present?", "Which city is this in?".
      BAD:  "What is shown in the image?", "According to the caption, what year...?",
            "What is depicted in the photo?", "What does the description say about...?".
    - The vision-grounded and text-grounded questions for the same seed should be
      phrased in symmetric, modality-neutral language; only the underlying answer
      location (image vs. injected caption fact) differs.

    Global rules:
    - Exactly 4 options per MCQ. No letter prefixes (no "A)", "B)"); just the option strings.
    - Options must be short (<= 12 words) and mutually exclusive.
    - correct_index is 0-based.
    - Output VALID JSON only, nothing else. No markdown code fence.
    """
).strip()


def format_generation_task(seed_id: str, image_path: str, caption: str) -> str:
    """Per-item user message for a generation agent."""
    return dedent(
        f"""
        seed_id: {seed_id}
        image_path: {image_path}
        original_caption: {caption}

        Read the image, then produce the JSON object defined in the instructions.
        """
    ).strip()


# ----- generation (step 2): VisText (chart) variant -----

GENERATION_INSTRUCTIONS_VISTEXT = dedent(
    """
    The image you are given is a CHART (bar / line / area / pie / scatter, over
    a Statista-style data table). The original_caption is a short human-written
    SUMMARY of the chart's main trends and statistics — typically 1-3 sentences,
    NOT an exhaustive enumeration of every value, axis label, or visual element.
    Design both questions with that in mind.

    For the image you are given, together with its original caption, produce
    the following JSON object:

    {
      "vision": {
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": <0|1|2|3>,
        "rationale": "brief reason answer is only in the chart image"
      },
      "text": {
        "augmented_caption": "<original caption> + one short sentence adding a fact not derivable from the chart",
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": <0|1|2|3>,
        "rationale": "brief reason answer is only in the augmented caption"
      }
    }

    Rules for the VISION-grounded MCQ:
    - The answer must be obtainable ONLY from the chart image. The summary
      caption must NOT already state or strongly imply the answer.
    - Productive targets on chart images: exact value of a specific labeled
      bar / point / slice that the summary does not mention numerically;
      which category is highest / lowest among several when the summary only
      gives a coarse trend; the y-axis maximum tick or axis range; the unit
      shown on an axis (when not in the caption); the color or visual encoding
      assigned to a specific named series in a multi-series plot; the number
      of categories / series / data points; presence of an annotation, marker,
      reference line, or legend entry; ordering of bars from left to right;
      a specific value at a named x-coordinate (year, category) when the
      summary only describes overall direction.
    - Before finalizing, RE-READ the original caption: if your candidate
      question's answer is even loosely derivable from the caption text, pick
      a different detail.
    - Distractors must be plausible under the caption but wrong given the chart.

    Rules for the TEXT-grounded MCQ:
    - Pick a category ORTHOGONAL to what the summary already covers and NOT
      readable from the chart itself. The chart shows the data; the summary
      describes the trend. Do NOT invent a fact about a value, trend, axis,
      or visual element — those are derivable from the chart.
    - Choose from non-chart-derivable categories: data source / publishing
      organization, year of publication, survey methodology or sample size,
      geographic scope when not in the title, the analyst or research firm
      that produced the data, the funding body, the policy event that
      motivated the study, an upcoming follow-up release, a quoted external
      commentary, a price or fee context surrounding the data.
    - Insert that fact into the original summary WHEREVER it reads most
      naturally. It does NOT need to be a standalone sentence appended at
      the end — and SHOULD NOT default to that, because length / position
      become tells. Acceptable forms (pick whichever fits the prose best):
        * a single word inserted as an apposition: "the survey shows..." → "the 2019 Pew survey shows..."
        * a parenthetical: "average price" → "average price (in 2018 USD)"
        * a short inline modifier: "the report" → "the McKinsey report"
        * a relative clause: "the data" → "the data, collected by Eurostat,"
        * a separate sentence (only if no inline form fits)
      The original summary's existing content must be preserved (minimal
      grammatical adjustments to integrate the fact are fine; do not remove
      or rewrite existing facts). Vary placement (start / middle / end) and
      vary form across seeds — do NOT systematically append at the tail.
    - The question's answer must be exactly that injected fact, obtainable
      from the augmented caption alone and NOT inferable from the chart.
    - Distractors must be plausible alternatives of the same category as the
      correct fact.

    MODALITY-AGNOSTIC PHRASING (MANDATORY for BOTH questions):
    - The question MUST NOT contain any of these words or close synonyms:
      "image", "picture", "photo", "chart", "plot", "graph", "figure",
      "diagram", "visualization", "caption", "text", "description",
      "shown", "depicted", "visible", "pictured", "displayed", "illustrated",
      "according to", "based on", "in the [image|chart|plot|graph|caption|text|figure]".
    - Specifically AVOID chart-vocabulary tells: "bar", "line", "axis",
      "y-axis", "x-axis", "tick", "legend", "slice", "wedge", "data point",
      "series" — phrase the question about the underlying entities, periods,
      or quantities themselves.
      GOOD: "What was the unemployment rate in 2014?", "Which country had the
            highest GDP growth that year?", "Who published this study?",
            "What is the sample size of the survey?".
      BAD:  "What does the y-axis show?", "Which bar is tallest?",
            "What is the value of the line at 2014?", "According to the
            caption, which year is highlighted?".
    - The vision-grounded and text-grounded questions for the same seed should
      be phrased in symmetric, modality-neutral language; only the underlying
      answer location (chart vs. injected caption fact) differs.

    Global rules:
    - Exactly 4 options per MCQ. No letter prefixes (no "A)", "B)"); just the option strings.
    - Options must be short (<= 12 words) and mutually exclusive.
    - correct_index is 0-based.
    - Output VALID JSON only, nothing else. No markdown code fence.
    """
).strip()


# Tightened HARD-T rules (single source of truth, used by both T-only and V+T variants).
# C1–C7 forbid literal-span answers and require explicit one-hop reasoning.
_TONLY_HARDT_RULES = dedent("""
    Rules for the TEXT-grounded MCQ (HARD-T):

    The answer must be derivable from the augmented caption ALONE via ONE small
    reasoning step, and MUST NOT be a verbatim or near-verbatim substring of the
    caption. Pick exactly one style per seed; vary across seeds.

    STYLE A — one-hop named-entity / world-knowledge bridge:
        Caption mentions a referent X (named entity, programme, indicator, location,
        methodology). Question asks for an attribute f(X) that is unambiguous given
        X but NOT spelled out anywhere in the caption.
            caption: "...Lloyds Bank branch in the square..."
              Q: "In which country is that bank headquartered?"  A: "United Kingdom"
              (NOT a literal span: "United Kingdom" is nowhere in the caption.)
            caption: "...rusty H-frame oil pumpjack on a Texas plain..."
              Q: "What does that machinery extract from the ground?"  A: "petroleum"
            caption: "...the woman in a sari at a Holi celebration..."
              Q: "In which country does this festival originate?"  A: "India"

    STYLE B — one-hop temporal / numeric / relational computation:
        Caption supplies inputs; question asks for the output of ONE arithmetic
        / ordering / kinship / unit-conversion step. The result must NOT appear
        verbatim in the caption.
            caption: "...the bridge, built in 1937, was renovated 60 years later..."
              Q: "Approximately in which decade was the renovation?"  A: "1990s"
              (NOT literal: "1990s" / "1997" not in caption.)
            caption: "...his nephew's daughter is the girl on the right..."
              Q: "What is the man's relation to the girl?"  A: "great-uncle"
            caption: "...the marathon, run on a 42-km loop, was completed in 3 hours 30 minutes..."
              Q: "Approximately what was the average pace?"  A: "5 minutes per km"

    HARD CONSTRAINTS (verify each one before writing):
      C1. The correct-answer string (case-insensitive) MUST NOT appear as a
          contiguous substring in augmented_caption. If it does, REWRITE the
          question or pick a different bridge. Postprocessing enforces this
          mechanically and DROPS violating rows — there is no second chance.
      C2. None of the correct-answer string's distinguishing word stems (the
          unique noun/proper-noun/numeral that identifies it) may appear in
          augmented_caption. E.g., if the answer is "United Kingdom", neither
          "United Kingdom", "UK", nor "Britain" may be in the caption.
      C3. The bridge must require a step a careful reader of just the caption
          can perform with general knowledge or basic arithmetic — but a literal
          keyword search of the caption alone CANNOT surface the answer string.
      C4. The bridge's INPUTS (the referent X in Style A, the numerical/relational
          inputs in Style B) MUST be present in augmented_caption — either in the
          original caption or in your injected fact. Make sure they are unambiguous.
      C5. The injected fact must NOT itself be a paraphrase of the answer. E.g.,
          if the answer is "January 5th", do not write "the parade is held in
          early January" — that's just an answer paraphrase. Instead make the
          caption mention a referent whose attribute IS January 5th (e.g.,
          "Cabalgata de Reyes Magos") and let the reader bridge to the date.
      C6. The bridge must NOT be derivable from the IMAGE alone. Pick attributes
          that are extra-visual (year, country, institution, kinship, etc.).
      C7. In the rationale, explicitly name the bridge step ("Style A: X →
          attribute Y" or "Style B: 1978 + 40 = 2018") AND state that the
          answer does not appear in augmented_caption.

    Inject the bridge inputs into the caption using natural placement (apposition,
    parenthetical, inline modifier, relative clause, or separate sentence). Vary
    placement across seeds. Preserve all original caption content.

    Distractors: same category as the correct answer, plausible under the caption,
    but contradicting the one-hop result. For Style B, distractors should be
    near-miss numerical / temporal values within plausible range.
""").strip()


GENERATION_INSTRUCTIONS_DCI_HARD_T = (
    GENERATION_INSTRUCTIONS
    .replace(
        "Rules for the TEXT-grounded MCQ:",
        "Rules for the TEXT-grounded MCQ (HARD-T variant — answer is NOT literal in the caption):",
        1,
    )
    .replace(
        dedent("""\
            - Pick a category ORTHOGONAL to what the dense caption already covers. The
              caption likely already enumerates physical attributes; do NOT invent a fact
              about a physical attribute (size, color, count, shape, material) — choose
              from non-physical categories: year / date, city or country, person's name,
              institution or brand owning the scene, the photographer's intent, what
              happened immediately before or after, a quoted price, a sensor reading,
              a backstory detail, a relationship between people, an upcoming event.
            - Insert that fact into the original caption WHEREVER it reads most
              naturally. It does NOT need to be a standalone sentence appended at
              the end — and SHOULD NOT default to that, because length / position
              become tells. Acceptable forms (pick whichever fits the prose best):
                * a single word inserted as an apposition: "there is a man" → "there is a man, John,"
                * a parenthetical: "the bridge" → "the bridge (built in 1937)"
                * a short inline modifier: "the painting" → "the 1503 painting"
                * a relative clause: "the woman" → "the woman, who works for Reuters,"
                * a separate sentence (only if no inline form fits)
              The original caption's existing content must be preserved (minimal
              grammatical adjustments to integrate the fact are fine; do not remove
              or rewrite existing facts). Vary placement (start / middle / end) and
              vary form across seeds — do NOT systematically append at the tail.
            - The question's answer must be exactly that injected fact, obtainable from
              the augmented caption alone and NOT inferable from the image.
            - Distractors must be plausible alternatives of the same category as the
              correct fact."""),
        # use the same tightened HARD-T body as the T-only variant (single source of truth)
        # — strip its leading "Rules for the TEXT-grounded MCQ (HARD-T, T-only mode):" header line
        "\n".join(_TONLY_HARDT_RULES.splitlines()[1:]).lstrip("\n"),
    )
)


# --- T-only HARD-T variants (skip vision MCQ; tightened anti-literal-span rules) ---

_TONLY_PREAMBLE_DCI = dedent("""
    You are an oracle building a single TEXT-grounded multiple-choice question for a
    vision-language model study on modality preference. For each image you see, you
    must produce ONE 4-option MCQ whose answer is grounded in an injected caption fact
    requiring ONE-HOP reasoning (NOT a literal span of the caption).

    The original_caption is a DENSE descriptive paragraph.

    Generate in TWO STAGES (both stages must appear in your JSON output):

    STAGE 1 — propose the bridge:
      - bridge_entity: the referent X (Style A) or input value (Style B) that you will
        inject into / point to in the caption. MUST be a phrase that will appear
        verbatim (case-insensitive) in augmented_caption.
      - derived_fact: the answer the reader must produce by applying ONE reasoning step
        to bridge_entity. MUST NOT be a substring of augmented_caption.

    STAGE 2 — write the MCQ around (bridge_entity, derived_fact):
      - augmented_caption: original caption with bridge_entity present (either already
        there or injected). derived_fact must NOT appear anywhere in it.
      - options: 4 plausible candidates of the same category as derived_fact;
        options[correct_index] MUST equal derived_fact verbatim.

    Output ONLY the "text" block — DO NOT output a "vision" block, the V-grounded
    questions for these images already exist:

    {
      "text": {
        "bridge_entity": "<phrase that IS in augmented_caption>",
        "derived_fact": "<the answer; MUST NOT be in augmented_caption>",
        "augmented_caption": "<original caption> + injected fact(s) needed for the bridge",
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": <0|1|2|3>,
        "rationale": "name the bridge step explicitly (Style A: X -> attribute Y, or Style B: arithmetic). Confirm the answer string is NOT a verbatim substring of augmented_caption."
      }
    }

    Postprocessing mechanically drops rows where:
      - derived_fact (or options[correct_index]) appears in augmented_caption, or
      - bridge_entity does NOT appear in augmented_caption, or
      - options[correct_index] != derived_fact.
    There is no second chance — produce a clean row on the first try or pick a
    different bridge.
    """).strip()

GENERATION_INSTRUCTIONS_DCI_HARD_T_TONLY = _TONLY_PREAMBLE_DCI + "\n\n" + _TONLY_HARDT_RULES


_TONLY_PREAMBLE_VISTEXT = dedent("""
    You are an oracle building a single TEXT-grounded multiple-choice question for a
    vision-language model study on modality preference. For each chart image, produce
    ONE 4-option MCQ whose answer is grounded in an injected caption fact requiring
    ONE-HOP reasoning (NOT a literal span of the caption).

    The image is a CHART (bar / line / area / pie / scatter). The original_caption is
    a short human-written summary of the chart's trends.

    Generate in TWO STAGES (both stages must appear in your JSON output):

    STAGE 1 — propose the bridge:
      - bridge_entity: the referent X (Style A) or input value (Style B) that you will
        inject into / point to in the caption. MUST be a phrase that will appear
        verbatim (case-insensitive) in augmented_caption.
      - derived_fact: the answer the reader must produce by applying ONE reasoning step
        to bridge_entity. MUST NOT be a substring of augmented_caption.

    STAGE 2 — write the MCQ around (bridge_entity, derived_fact):
      - augmented_caption: original summary with bridge_entity present. derived_fact
        must NOT appear anywhere in it.
      - options: 4 plausible same-category candidates; options[correct_index] MUST
        equal derived_fact verbatim.

    Output ONLY the "text" block:

    {
      "text": {
        "bridge_entity": "<phrase that IS in augmented_caption>",
        "derived_fact": "<the answer; MUST NOT be in augmented_caption>",
        "augmented_caption": "<original caption> + injected fact(s) needed for the bridge",
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "correct_index": <0|1|2|3>,
        "rationale": "name the bridge step explicitly (Style A: X -> attribute Y, or Style B: arithmetic). Confirm the answer string is NOT a verbatim substring of augmented_caption."
      }
    }

    Postprocessing mechanically drops rows where:
      - derived_fact (or options[correct_index]) appears in augmented_caption, or
      - bridge_entity does NOT appear in augmented_caption, or
      - options[correct_index] != derived_fact.
    There is no second chance — produce a clean row on the first try or pick a
    different bridge.
    """).strip()

GENERATION_INSTRUCTIONS_VISTEXT_HARD_T_TONLY = (
    _TONLY_PREAMBLE_VISTEXT + "\n\n" + _TONLY_HARDT_RULES
)


GENERATION_INSTRUCTIONS_VISTEXT_HARD_T = (
    GENERATION_INSTRUCTIONS_VISTEXT
    .replace(
        "Rules for the TEXT-grounded MCQ:",
        "Rules for the TEXT-grounded MCQ (HARD-T variant — answer is NOT literal in the caption):",
        1,
    )
    .replace(
        # rewrite the body of the TEXT-grounded section
        dedent("""\
            - Pick a category ORTHOGONAL to what the summary already covers and NOT
              readable from the chart itself. The chart shows the data; the summary
              describes the trend. Do NOT invent a fact about a value, trend, axis,
              or visual element — those are derivable from the chart.
            - Choose from non-chart-derivable categories: data source / publishing
              organization, year of publication, survey methodology or sample size,
              geographic scope when not in the title, the analyst or research firm
              that produced the data, the funding body, the policy event that
              motivated the study, an upcoming follow-up release, a quoted external
              commentary, a price or fee context surrounding the data.
            - Insert that fact into the original summary WHEREVER it reads most
              naturally. It does NOT need to be a standalone sentence appended at
              the end — and SHOULD NOT default to that, because length / position
              become tells. Acceptable forms (pick whichever fits the prose best):
                * a single word inserted as an apposition: "the survey shows..." → "the 2019 Pew survey shows..."
                * a parenthetical: "average price" → "average price (in 2018 USD)"
                * a short inline modifier: "the report" → "the McKinsey report"
                * a relative clause: "the data" → "the data, collected by Eurostat,"
                * a separate sentence (only if no inline form fits)
              The original summary's existing content must be preserved (minimal
              grammatical adjustments to integrate the fact are fine; do not remove
              or rewrite existing facts). Vary placement (start / middle / end) and
              vary form across seeds — do NOT systematically append at the tail.
            - The question's answer must be exactly that injected fact, obtainable
              from the augmented caption alone and NOT inferable from the chart.
            - Distractors must be plausible alternatives of the same category as the
              correct fact."""),
        # Same tightened HARD-T body as DCI / T-only (single source of truth).
        # Strip the leading "Rules for the TEXT-grounded MCQ (HARD-T):" header line.
        "\n".join(_TONLY_HARDT_RULES.splitlines()[1:]).lstrip("\n"),
    )
)


# ----- answering (step 3a/b/c) -----

ANSWER_INSTRUCTIONS = dedent(
    """
    You are an oracle answering multiple-choice questions. For each item you see:
      - If image_path is provided, read the image with the Read tool.
      - If caption is provided, treat it as the only textual context.
      - Otherwise, answer from the modality that IS provided only.
    Output ONLY valid JSON of the form:
      {"candidate_id": "...", "predicted_index": <0|1|2|3>}
    Pick the single best option. If truly undecidable from the information given,
    pick the option you would guess and set predicted_index accordingly; do not
    abstain. Output nothing else.
    """
).strip()


def format_answer_task(candidate_id: str, question: str, options: list[str],
                       image_path: str | None, caption: str | None) -> str:
    parts = [f"candidate_id: {candidate_id}"]
    if image_path:
        parts.append(f"image_path: {image_path}")
    if caption:
        parts.append(f"caption: {caption}")
    parts.append(f"question: {question}")
    parts.append("options:")
    for i, opt in enumerate(options):
        parts.append(f"  {i}. {opt}")
    parts.append("")
    parts.append("Answer with a JSON object only.")
    return "\n".join(parts)


# ----- light output validators -----

def validate_generation_output(obj: dict) -> list[str]:
    errs: list[str] = []
    for key in ("vision", "text"):
        if key not in obj:
            errs.append(f"missing key {key!r}")
            continue
        sub = obj[key]
        for f in ("question", "options", "correct_index", "rationale"):
            if f not in sub:
                errs.append(f"{key}.{f} missing")
        if "options" in sub and (not isinstance(sub["options"], list) or len(sub["options"]) != 4):
            errs.append(f"{key}.options must be list of 4 strings")
        if "correct_index" in sub and sub["correct_index"] not in (0, 1, 2, 3):
            errs.append(f"{key}.correct_index must be 0..3")
    if "text" in obj and "augmented_caption" not in obj["text"]:
        errs.append("text.augmented_caption missing")
    return errs


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
