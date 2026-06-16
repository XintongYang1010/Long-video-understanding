"""Prompts for two-user EgoLife QA generation and review."""

from __future__ import annotations

import collections
import json
from typing import Any


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question_type": "short self-described type, e.g. memory_gap/follow_up/role_handoff/verification; strict mode may use commonality or difference",
    "question": "natural first-person question (ask from a AR glass user perspective) without timestamps or words like video/footage/recording/frame/camera",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "content_category": "social_interaction/task_coordination/theory_of_mind/temporal_reasoning/environmental_interaction",
    "category": "same value as content_category, kept for backward compatibility",
    "required_users": ["at least two user names"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "visual fact contributed by this user's own video",
            "timeframe": "specific start-end time range or approximate moment in this user's video",
            "frames_used": ["video-level evidence or approximate moment label"],
        }
    ],
    "referred_timestamps": [
        {
            "user": "name",
            "timestamp_seconds": 0.0,
            "moment": "brief visual moment used as evidence",
        }
    ],
    "single_user_answerability": {
        "Jake": "insufficient because Jake alone only provides ...",
        "Alice": "insufficient because Alice alone only provides ...",
    },
    "combined_answerability": "sufficient because combining the required users' videos supports exactly one option",
    "added_agent_utility": "offscreen_followup/simultaneous_elsewhere/handoff_chain/visual_disambiguation/object_state_change/social_reaction/role_or_task_split",
    "reasoning_pattern": "anchor_to_missing_state/before_after_outcome/simultaneity/handoff/role_attribution/object_state_change/visual_detail_resolution/social_response/verification",
    "question_style": "memory_gap/simultaneity/follow_up/handoff/disambiguation/role_split/object_state/social_response/verification",
    "generator_rationale": "why this is a natural speaker-anchor plus missing-detail question",
    "why_two_users_needed": "why each required user contributes necessary non-redundant visual evidence",
    "per_user_evidence_claims": [
        {"user": "name", "claim": "claim grounded in that user's own video"}
    ],
    "review": {
        "generator_self_check": "why this cannot be answered by one user alone and is not just asking what both users saw",
        "status": "draft",
    },
}

GENERATION_MODES = ("strict_design", "relaxed_natural")


ANSWERABILITY_SCHEMA = {
    "choice": "A/B/C/D/E or insufficient",
    "answer_text": "selected option text or empty string",
    "confidence": 0.0,
    "evidence_used": "short explanation grounded only in the provided videos",
    "insufficient_reason": "explain what is missing if choice is insufficient",
}


JUDGE_CHECK_SCHEMA = {
    "status": "PASS/FAIL/UNCERTAIN",
    "reason": "short evidence-grounded explanation",
    "fix": "specific repair instruction if status is FAIL or UNCERTAIN; empty string if PASS",
}


JUDGE_SCHEMA = {
    "review_passed": True,
    "checks": {
        "first_person_naturalness": JUDGE_CHECK_SCHEMA,
        "agent_perspective": JUDGE_CHECK_SCHEMA,
        "source_scope": JUDGE_CHECK_SCHEMA,
        "question_type_semantics": JUDGE_CHECK_SCHEMA,
        "multi_video_necessity": JUDGE_CHECK_SCHEMA,
        "visual_grounding": JUDGE_CHECK_SCHEMA,
        "mcq_option_quality": JUDGE_CHECK_SCHEMA,
        "gaze_safety": JUDGE_CHECK_SCHEMA,
        "human_auditability": JUDGE_CHECK_SCHEMA,
    },
    "blocking_failures": ["names of failed checks that should block acceptance"],
    "why_generator_asked_this": "short justification of the generator's likely reason",
    "feedback_to_generator": "specific edit instructions if review_passed is false; empty string if passed",
}


CONTENT_CATEGORY_DEFINITIONS = {
    "social_interaction": "people reacting, responding, helping, joking, conversing, or taking turns",
    "task_coordination": "shared task progress, role split, handoff, setup, assembly, or collaboration",
    "theory_of_mind": "visible signs of intention, confusion, surprise, misunderstanding, or what someone failed to notice",
    "temporal_reasoning": "simultaneous events, before/after outcomes, or what continued after the speaker looked away",
    "environmental_interaction": "object state, object location, room state, layout, device state, or visible environmental change",
}

ADDED_AGENT_UTILITY_DEFINITIONS = {
    "offscreen_followup": "the speaker sees an anchor or leaves/turns away; another user supplies the related follow-up state",
    "simultaneous_elsewhere": "the speaker is occupied with one event while another user sees a related event elsewhere or nearby",
    "handoff_chain": "an object, device, or task moves across people or roles, and both perspectives are needed to understand the chain",
    "visual_disambiguation": "the speaker has a partial/ambiguous view, and another user resolves a concrete visual detail",
    "object_state_change": "another user supplies a concrete state, location, or change of an object/place tied to the speaker's anchor",
    "social_reaction": "another user supplies a visible reaction or response to something the speaker did, said, or missed",
    "role_or_task_split": "the users perform complementary roles in a task, and the answer depends on both role contexts",
}

REASONING_PATTERN_DEFINITIONS = {
    "anchor_to_missing_state": "start from the speaker's own anchor event and ask for a missing related state",
    "before_after_outcome": "ask what happened or continued after the speaker's action, attention shift, or departure",
    "simultaneity": "ask what was happening at the same time as the speaker's anchor action",
    "handoff": "ask where an object/task went next, who handled it next, or which step followed the speaker's action",
    "role_attribution": "ask who took which visible role while the speaker was doing another role",
    "object_state_change": "ask how an object/place changed while the speaker was focused elsewhere",
    "visual_detail_resolution": "ask which visible detail another perspective clarifies",
    "social_response": "ask who reacted/responded and how, grounded in visible or audible behavior if available",
    "verification": "ask which detail can be confirmed only by combining the speaker's anchor with another view",
}

QUESTION_STYLE_TEMPLATES = {
    "memory_gap": [
        "I remember [speaker anchor], but what was still happening with [target]?",
        "What did I miss about [target] after [speaker anchor]?",
        "I couldn't see [target] from where I was; what was going on with it?",
    ],
    "simultaneity": [
        "At the moment I was focused on [anchor], what happened to [same object/task]?",
        "As I moved into [speaker phase], which part of [same task/object] changed outside my view?",
    ],
    "follow_up": [
        "Once I [left/turned away/shifted attention], what happened next with [target]?",
        "What was still going on in [place] after I [speaker anchor]?",
        "Where did [target] end up once I stopped looking at it?",
    ],
    "handoff": [
        "Who handled [object/task] next after it left my view?",
        "Which step happened next after I [speaker anchor]?",
        "Where did [object from my action] go once I was no longer holding it?",
    ],
    "disambiguation": [
        "From my angle, I could only tell [partial clue]. What detail was I missing?",
        "Which detail about [object/person/action] was unclear from where I was?",
        "What could I not tell about [target] from my side of the room?",
    ],
    "role_split": [
        "How was the task divided while I was [speaker action]?",
        "Who took over [role/task] while I was [speaker action]?",
        "What part of the same task was [other person/group] handling as I worked on [speaker role]?",
    ],
    "object_state": [
        "What changed about [object/place] while I was [speaker action]?",
        "Where was [object] when my view no longer showed it clearly?",
        "What state was [object/place] in by the time I [speaker anchor]?",
    ],
    "social_response": [
        "Who reacted when I [speaker action], and how?",
        "What response did I miss while I was focused on [anchor]?",
        "How did [person/group] respond as I was [speaker action]?",
    ],
    "verification": [
        "Which detail about [target] could I not confirm from my own view?",
        "What detail about [target] needed another person's perspective to verify?",
        "From what I experienced, what remained uncertain about [target]?",
    ],
}

QUESTION_STYLE_REQUIREMENTS = {
    "memory_gap": (
        "Start with a missing-memory shape such as 'What did I miss...', "
        "'I couldn't see...', or 'From where I was...'. Do not start with 'After I' or 'While I'."
    ),
    "simultaneity": (
        "Start with 'At the moment...' or 'As I...' and make the simultaneous relation explicit. "
        "Avoid the generic 'While I was..., what was Alice doing...' shape."
    ),
    "follow_up": (
        "Start with 'Once I...', 'What was still...', or 'Where did... end up...' and ask for a follow-up state."
    ),
    "handoff": (
        "Start with 'Who handled...', 'Which step happened next...', or 'Where did [object] go...' "
        "and make the handed-off object or task explicit."
    ),
    "disambiguation": (
        "Start with 'From my angle...', 'Which detail...', or 'What could I not tell...' "
        "and ask for a detail that the speaker's view leaves ambiguous."
    ),
    "role_split": (
        "Start with 'How was...', 'Who took over...', or 'What part...' and ask about complementary roles."
    ),
    "object_state": (
        "Start with 'What changed...', 'Where was...', or 'What state was...' and ask about a state or location change."
    ),
    "social_response": (
        "Start with 'Who reacted...', 'What response did I miss...', or 'How did...' "
        "and ask for a visible response tied to the speaker's action."
    ),
    "verification": (
        "Start with 'Which detail...', 'What detail...', or 'From what I experienced...' "
        "and ask what needed the added user's perspective to verify."
    ),
}


def _definition(mapping: dict[str, str], key: str | None) -> str:
    if key and key in mapping:
        return mapping[key]
    return "use the assigned label if present; otherwise choose the best matching pattern from the current videos"


def _design_cell_text(question_type: str, design_cell: dict[str, Any] | None) -> str:
    design_cell = dict(design_cell or {})
    content_category = design_cell.get("content_category")
    added_agent_utility = design_cell.get("added_agent_utility")
    reasoning_pattern = design_cell.get("reasoning_pattern")
    question_style = design_cell.get("question_style")
    style_templates = QUESTION_STYLE_TEMPLATES.get(str(question_style), [])
    style_requirement = QUESTION_STYLE_REQUIREMENTS.get(
        str(question_style),
        "Use the assigned style deliberately while preserving a real two-user dependency.",
    )
    template_lines = "\n".join(f"- {template}" for template in style_templates)
    if not template_lines:
        template_lines = "- Use a concrete first-person memory-gap style; do not default to the same sentence skeleton."

    return f"""Assigned diversity design cell:
- question_type: {question_type}
- content_category: {content_category or "choose one MA-EgoQA style content category"}
  Definition: {_definition(CONTENT_CATEGORY_DEFINITIONS, content_category)}
- added_agent_utility: {added_agent_utility or "choose a non-redundant added-agent utility pattern"}
  Definition: {_definition(ADDED_AGENT_UTILITY_DEFINITIONS, added_agent_utility)}
- reasoning_pattern: {reasoning_pattern or "choose the best reasoning pattern"}
  Definition: {_definition(REASONING_PATTERN_DEFINITIONS, reasoning_pattern)}
- question_style: {question_style or "choose a varied first-person style"}
  Required wording shape: {style_requirement}
  Style examples to imitate structurally, not copy:
{template_lines}

Use the assigned design cell to diversify the question. The output category should match the assigned content_category when one is provided.
Return both content_category and category with the same value.
Still obey the current strict gate: every single required user's video alone must be insufficient, and the combined required users' videos must make exactly one option correct.
Dependency guard:
- Do not create a question where the correct option is simply a standalone detail visible in the non-speaker user's video, such as who was near a window, what Alice was holding, or what Alice was doing with a device.
- The speaker-side anchor must select the exact phase, object instance, handoff, outcome, or ambiguity that the other user resolves. If the non-speaker user's video alone can pick the answer without using the speaker anchor, rewrite.
- First satisfy the two-user dependency, then use the assigned wording shape. Do not fall back to the plain "After I..." or "While I..." skeleton unless the assigned style requirement itself allows it.
"""


def _accepted_context_text(accepted_context: list[dict[str, Any]] | None) -> str:
    if not accepted_context:
        return ""
    recent = list(accepted_context[-8:])
    openings = collections.Counter(
        str(row.get("question", "")).split()[0]
        for row in accepted_context
        if str(row.get("question", "")).strip()
    )
    styles = collections.Counter(str(row.get("question_style", "")) for row in accepted_context if row.get("question_style"))
    categories = collections.Counter(
        str(row.get("content_category") or row.get("category") or "")
        for row in accepted_context
        if row.get("content_category") or row.get("category")
    )
    question_types = collections.Counter(
        str(row.get("question_type", "")) for row in accepted_context if row.get("question_type")
    )
    utilities = collections.Counter(
        str(row.get("added_agent_utility", ""))
        for row in accepted_context
        if row.get("added_agent_utility")
    )
    reasoning = collections.Counter(
        str(row.get("reasoning_pattern", ""))
        for row in accepted_context
        if row.get("reasoning_pattern")
    )
    questions_lower = [str(row.get("question", "")).lower() for row in accepted_context]
    phrase_counts = collections.Counter(
        {
            "i_was_focused": sum("i was focused" in question for question in questions_lower),
            "i_was_holding": sum("i was holding" in question for question in questions_lower),
            "did_not_notice": sum(
                ("didn't notice" in question or "did not notice" in question)
                for question in questions_lower
            ),
            "who_took": sum("who took" in question for question in questions_lower),
            "took_over": sum("took over" in question for question in questions_lower),
            "who_started": sum("who started" in question for question in questions_lower),
        }
    )
    dominant_lines = []
    for label, counter in (
        ("opening", openings),
        ("question_type", question_types),
        ("content_category", categories),
        ("added_agent_utility", utilities),
        ("reasoning_pattern", reasoning),
        ("question_style", styles),
    ):
        repeated = {key: count for key, count in counter.items() if key and count >= 2}
        if repeated:
            dominant_lines.append(f"- Repeated {label} values to avoid now: {dict(repeated)}")
    repeated_phrases = {key: count for key, count in phrase_counts.items() if count >= 2}
    if repeated_phrases:
        dominant_lines.append(f"- Repeated wording templates to avoid now: {repeated_phrases}")
    if (
        question_types.get("role_handoff", 0) >= 2
        or utilities.get("handoff_chain", 0) >= 2
        or reasoning.get("handoff", 0) >= 2
        or styles.get("handoff", 0) >= 2
    ):
        dominant_lines.append(
            "- Handoff is already dominant: do not make another role_handoff/handoff_chain/"
            "handoff question, and do not use 'took over' or 'who started'."
        )
    if phrase_counts["i_was_focused"] >= 2:
        dominant_lines.append(
            "- The opening 'I was focused...' is already dominant: start with a different "
            "everyday question form such as 'What changed...', 'Which detail...', "
            "'Did anyone...', 'Where was...', 'How did...', or 'What was still...'."
        )
    if phrase_counts["who_took"] >= 2:
        dominant_lines.append(
            "- 'Who took...' is already dominant: ask about a state, reaction, verification "
            "detail, simultaneous event, location, or visible outcome instead of another taker."
        )
    dominant_block = (
        "Dominant patterns to avoid in the next relaxed_natural item:\n"
        + "\n".join(dominant_lines)
        if dominant_lines
        else "Dominant patterns to avoid in the next relaxed_natural item:\n- No dominant repeated pattern yet."
    )
    recent_lines = []
    for row in recent:
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        recent_lines.append(
            "- "
            f"{row.get('question_type', '')} | "
            f"{row.get('content_category') or row.get('category') or ''} | "
            f"{row.get('added_agent_utility', '')} | "
            f"{row.get('question_style', '')}: "
            f"{question}"
        )
    if not recent_lines:
        recent_lines = ["- No accepted question text available."]
    return f"""Already accepted QA diversity context for this run:
- accepted_count: {len(accepted_context)}
- opening_counts: {dict(openings)}
- question_type_counts: {dict(question_types)}
- question_style_counts: {dict(styles)}
- content_category_counts: {dict(categories)}
- added_agent_utility_counts: {dict(utilities)}
- reasoning_pattern_counts: {dict(reasoning)}
- wording_template_counts: {dict(phrase_counts)}
- recent accepted questions:
{chr(10).join(recent_lines)}

{dominant_block}

Do not reuse an already accepted question verbatim or as the same fill-in template with only a timestamp changed.
If the assigned style is already represented, create a visibly different wording pattern, object anchor, place, and answer target.
For social_response questions, avoid the bare template "Who reacted when I was [action], and how?" unless the reaction target, action, and response are specific and not already used.
Hard diversity requirement for relaxed_natural runs:
- Treat the recent accepted questions as patterns to avoid, not examples to imitate.
- Do not repeat the same opening word, question_type, content_category, added_agent_utility, reasoning_pattern, and question_style combination unless the visible evidence leaves no other natural question.
- If any relation, wording template, or label family appears two or more times above, treat it as closed for this run and choose a different relation.
- If recent accepted questions ask "I was focused/holding..., didn't notice what happened to [object], who took it and where did it end up?", choose a different natural relation: a follow-up state, a social response, a verification detail, a simultaneous event, a place/state change, a visual disambiguation, an outcome check, or a missed response.
- If recent accepted questions ask "I was focused..., who took over..., who started...", do not ask another handoff/took-over/started question. Pick a non-handoff relation instead.
- Avoid reusing the same target object, answer target, and final action pattern from recent accepted questions.
- Privately compare at least three possible questions against the accepted context, then output only the JSON for the least similar evidence-grounded question.
"""


def _feedback_text(feedback: str | None) -> str:
    return (
        "\nPrevious judger/evaluator feedback to fix:\n"
        f"{feedback}\n"
        "You must incorporate this feedback into the new question, options, answer, and evidence. "
        "Do not repeat the rejected issue.\n"
        if feedback
        else ""
    )


def video_packet_brief(packet: dict[str, Any]) -> str:
    clips = []
    for clip in packet.get("clips", []):
        clips.append(
            {
                "user": clip.get("agent_name"),
                "day": clip.get("day"),
                "clip_clock": clip.get("clip_clock"),
                "video_url": clip.get("video_url"),
                "local_video": clip.get("local_video"),
                "gaze_summary": clip.get("gaze_summary"),
                "projection_status": clip.get("gaze_summary", {}).get("projection_status"),
            }
        )
    return json.dumps(
        {
            "evidence_id": packet.get("evidence_id"),
            "required_users": packet.get("required_users"),
            "requirement": packet.get("requirement"),
            "clips": clips,
            "source_urls": packet.get("source_urls"),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_video_generation_prompt(
    packet: dict[str, Any],
    question_type: str,
    feedback: str | None = None,
    design_cell: dict[str, Any] | None = None,
    accepted_context: list[dict[str, Any]] | None = None,
    generation_mode: str = "strict_design",
) -> str:
    if generation_mode == "relaxed_natural":
        accepted_context_block = _accepted_context_text(accepted_context)
        feedback_block = _feedback_text(feedback)
        return f"""You are an assistant tasked with generating one meaningful, natural MCQ from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other, or in different places. Look directly at the raw videos and use only visual evidence, video metadata, and provided gaze coordinates when available. Do not use captions, subtitles, transcripts, or pre-written observations.

Goal for this relaxed 32B experiment:
1. Generate exactly one five-option multiple-choice question that a real AR-glasses wearer might naturally ask later.
2. Do not force the question into a preset commonality/difference type, style, or wording template.
3. The question should be grounded in one speaker/base user's own experience and a missing related detail available from another required user's view.
4. The question should sound conversational and specific, not like a taxonomy exercise.
5. The question must not mention video, footage, recording, frame, dataset, camera, clip, caption, subtitle, or timestamp.
6. Options must be plausible, parallel, and have exactly one correct answer.

Perspective and identity rules:
- required_users are viewpoint owners / camera wearers. A viewpoint owner is not automatically the person visible in that view.
- For any two users, if user A's view shows user B, describe user B as visible from user A's perspective; do not attribute user B's visible action to user A.
- When the viewpoint owner and visible person differ, state both roles clearly instead of treating the camera wearer as the actor on screen.
- Keep speaker/base user, viewpoint_owner, and visible_person distinct in evidence claims and rationale.
- Do not name the speaker/base user in the question or answer when the question is asked from that user's first-person perspective.

Naturalness guidance:
- Prefer everyday memory or AR-assistant wording: "What did I miss...", "What was still happening...", "Which detail could I not confirm...", "How did they respond...", "What changed after I looked away...", "Where was it by then?", "Did anyone react?".
- Use "Who took over..." or "Who started..." only when that relation is genuinely the freshest natural relation for the current videos and not already repeated in the accepted context.
- Avoid rigid openings reused from prior accepted questions.
- Avoid overusing "I was focused on..." / "I was holding..." / "I didn't notice what happened to..." openings across a run.
- Do not default to an object-movement question such as "who took it and where did it end up" when another natural two-user dependency is visible.
- Do not default to a handoff question such as "who took over" or "who started" when recent accepted questions already use handoff language.
- Avoid generic questions like "what was the other person doing?" unless tied to a concrete object, place, action, role, reaction, or follow-up state from the speaker's own context.
- Avoid asking what both users saw, both noticed, or both were doing together.

Metadata instructions:
- Fill question_type, content_category, added_agent_utility, reasoning_pattern, and question_style as short self-descriptive labels after choosing the natural question. They are diagnostic labels, not constraints.
- Do not copy the previous row's labels automatically; choose labels that describe the new natural question.
- Return both content_category and category with the same value.
- Fill single_user_answerability and combined_answerability as the generator's rationale only; a human reviewer will replace the automatic answerability gate for this experiment.
- Fill per_user_evidence_claims with clear viewpoint language, for example: "The viewpoint owner's view shows another named person placing the device on the table."

{accepted_context_block}

{feedback_block}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""
    if generation_mode != "strict_design":
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    type_instruction = {
        "commonality": (
            "Create a commonality question only when the common state is established by combining "
            "a speaker-side anchor from one required user's video with a missing related detail "
            "visible only in another required user's video. Do not ask for an object, action, or "
            "room state that each single video independently reveals."
        ),
        "difference": (
            "Create a difference question: the answer should identify a meaningful difference, "
            "asymmetry, or complementary detail between the required users' egocentric videos."
        ),
    }[question_type]
    feedback_block = _feedback_text(feedback)
    design_cell_block = _design_cell_text(question_type, design_cell)
    accepted_context_block = _accepted_context_text(accepted_context)
    example_block = """One good example for natural multi-user QA design.
This example illustrates the desired reasoning pattern only. Do not copy its objects, activities, answers, names, or options into the new QA item.
Do not treat it as evidence for the current videos. Use only the current raw videos and packet metadata for the actual QA.

Good example: setup check followed by missing room state
Video situation:
- One person checks a device/timer/setup near a practice or presentation room, then walks toward the stairwell.
- Another person's view still shows the front of that room, where an exercise or dance tutorial continues on the big screen.
Good question:
- "After I checked the setup and walked toward the stairwell, what was still going on at the front of the room I had just left?"
Why good:
- It starts from what the speaker experienced: checking the setup and leaving.
- The other video answers the missing follow-up state after the speaker left.
- The answer requires combining the speaker's anchor event with another user's visual evidence.

Compact design rules:
- A good question starts from one user's own anchor event and asks for a missing related detail supplied by another user's video.
- The speaker/base user's perspective is the question perspective. Another user's video is an added source that fills the speaker's memory gap; do not ask what the video shows.
- The speaker's video must not already reveal the correct answer; the other user's video must add the missing visual detail.
- If either single user's video can select the correct option, discard the question and create a different one.
- Do not make a question just because clips share a timestamp.
- Do not ask what both users saw, noticed, or looked at.
- Do not ask what both users did, handled, had, shared, or were doing together.
- You may ask about another room, an offscreen area, or what continued after the speaker left, but only if the speaker-side anchor is needed to set the time/context and the other video supplies a concrete missing detail.
- Avoid generic wording like "what was the other person doing nearby"; if the question uses "other person", it must also name a concrete object, place, role, action, or follow-up state tied to the speaker's anchor.
- Do not ask a generic comparison of two views, rooms, or camera angles.
"""
    return f"""You are an assistant tasked with generating one meaningful, contextually grounded MCQ from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other, or in different places. Look directly at the videos and use only visual evidence, video metadata, and the provided 2D gaze coordinates when available. Do not use captions, subtitles, transcripts, or pre-written observations.

Your job:
1. Generate exactly one five-option multiple-choice question.
2. The question_type must be "{question_type}": {type_instruction}
3. The question must be asked from one user's first-person speaker/base perspective, starting from that user's own memory, action, attention, or location context.
4. The question must ask about a missing related detail that is supplied by another required user's video. The users may be in the same area, another room, or different parts of a shared task.
5. The question must use the assigned diversity design cell below; follow its Required wording shape instead of defaulting to the same "After I..., what was..." skeleton.
6. The question must require visual evidence from at least two required users. Timestamp overlap is not enough.
7. Any single required user's video alone must be insufficient; the combined required users' videos must make exactly one option correct.
8. Fill the evidence field with each needed user's visual fact and a specific timeframe.
9. Return every field in the JSON shape exactly. Do not omit content_category, category, single_user_answerability, combined_answerability, added_agent_utility, reasoning_pattern, question_style, generator_rationale, why_two_users_needed, per_user_evidence_claims, referred_timestamps, or review.
10. The answer field must exactly equal the text of options[correct], and correct must be one letter: A, B, C, D, or E.

Guidelines:
1) Ask in a natural, informal, everyday way, like someone looking back at their memories.
For example, "Where did I put my glasses when I was having lunch with Tasha and Alice?"
2) Use first-person or shared-memory wording from an AR-glasses user's perspective, such as "I", "me", "my", "we", or "our"...
3) Do not name a required user in the question or the answer when the question is asked from that person's perspective.
For example, If the question is asked from Jake's perspective, Jake's name should not appear in the question or the answer.
4) Do not use words such as video, footage, recording, frame, dataset, camera, clip, caption, subtitle, or timestamp in the question or options.
5) Keep the question specific, concrete, conversational, and visually grounded.
6) Options must be multi-word, plausible, parallel in length/style, and have exactly one correct answer.
7) False options may use Jake, Alice, Tasha, Lucia, Katrina, or Shure when helpful, please refer to guideline 3) for name requirement.
8) The gaze input is provided as <gaze_coordinate>, a 2D image coordinate (x, y) indicating the user's attended area. Ask questions about visible objects, regions, or actions near what the user attended to.
9) single_user_answerability must be an object with one entry for each required user, and each entry must explicitly say "insufficient because ...".
10) combined_answerability must explicitly say "sufficient because ..." and explain why the combined videos support the correct option.
11) Before returning, mentally run the single-user test. If Jake alone, Alice alone, or any other single required user can answer the question, rewrite it.
12) Avoid these rejected patterns: "What did we both...", "What did we all...", "What did everyone...", "What did I and Alice both...", "What were we doing together...", and "What was the other person doing nearby?".
13) For commonality questions, the commonality must be the relationship between the speaker's anchor and the second user's missing detail, not merely a shared object or shared action visible in both views.
14) Vary the question opening according to the assigned question_style. Prefer assigned-style openings such as "What...", "Who...", "Which...", "Where...", "How...", "Once I...", "At the moment I...", "From my angle...", or "I couldn't see...". Avoid starting with "After I" or "While I" unless the assigned style requirement explicitly points there.
15) Hard single-user trap: do not ask "who was standing near...", "what was Alice doing...", "what was Alice holding...", or "what color/person/object was visible..." when the other user's video alone can identify it. Tie the answer to the speaker's exact object/action/phase, such as the item I had just picked up, the case I had just opened, the device part I handed over, or the place I had just left.
16) Before returning, apply this stricter rewrite test: if the second user's video alone could answer by scanning its visible scene, change the question so the speaker's anchor is needed to know which object, person, phase, or follow-up state is being asked about.

{accepted_context_block}

{design_cell_block}

{example_block}

{feedback_block}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_judger_prompt(qa_item: dict[str, Any], packet: dict[str, Any]) -> str:
    return f"""You are a strict judger for EgoLife video-first two-user MCQ generation.

You will see the same raw egocentric videos used by the generator. Judge whether the generated question is acceptable.

Return every check in the JSON schema. Judge all checks, but focus most carefully on multi_video_necessity.

Brief checks:
1. first_person_naturalness: natural first-person memory/AR-assistant wording.
2. agent_perspective: no dataset-observer wording and no video/footage/recording/frame/camera/clip/timestamp in the question or options.
3. source_scope: answerable from provided raw videos and metadata only.
4. question_type_semantics: commonality means shared/jointly verified; difference means meaningful asymmetry or complementary detail.
6. visual_grounding: correct option and evidence claims are visually grounded in concrete moments.
7. mcq_option_quality: exactly five plausible options and exactly one correct answer.
8. gaze_safety: do not invent exact gaze-to-object claims when 2D gaze is unavailable.
9. human_auditability: enough user/video/time evidence exists for a human to inspect later.

Main check, 5. multi_video_necessity:
- Judge whether the QA has a situated cross-video dependency, not just two synchronized clips.
- PASS only if one required user's video provides a speaker-side anchor event and another required user's video provides a missing visual detail that is simultaneous, follow-up, or otherwise naturally related.
- PASS only if both videos are necessary: removing either user's video would make the question unanswerable or would leave more than one plausible option.
- PASS only if the connection would be a plausible memory or AR-assistant question from someone involved in the situation.
- PASS can include offscreen, different-room, or after-I-left questions when the speaker/base user's own event sets the time/context and another user's view supplies a concrete missing state, action, object, or response.
- FAIL if the question merely stitches together two clips because they share a time interval.
- FAIL if the activities are unrelated, such as one person discussing/checking a device while another person is washing dishes, unless the question identifies a concrete shared task or natural dependency.
- FAIL if the question asks what both users saw, both noticed, or both looked at; do not ask what both users saw or noticed because one user may not know the other user's perception.
- FAIL if the question is a generic comparison of two views, rooms, or camera angles rather than a speaker anchor plus missing visual detail.
- FAIL if the question generically asks what "the other person", "everyone else", or "others" were doing nearby, in the room, or at the same time without naming a concrete missing visual detail, place, object, role, reaction, or follow-up state tied to the speaker-side anchor.
- FAIL if the question is merely "what was [other user] doing/holding/handling" and that user's video alone can choose the answer. PASS that wording only when the speaker-side anchor selects a specific object instance, handoff, phase, outcome, ambiguity, or follow-up state that the other user's video alone would not identify.
- FAIL if the question is a near-duplicate of a known prior pattern in the prompt context, such as repeatedly asking "I was focused/holding..., didn't notice what happened to [object], who took it and where did it end up?" with only the object/person/place changed.
- FAIL if a single user's video already reveals the correct answer.
- UNCERTAIN if the videos do not clearly show the anchor, the missing visual detail, or the relation between them.
- In the reason, explicitly name the speaker-side anchor, the missing visual detail, and why the second video is or is not needed.

Contrastive example for multi_video_necessity:
- PASS: One video shows the speaker checking a setup and leaving toward a stairwell; another video still shows the front of that room where a tutorial continues. A good question asks what was still happening after the speaker left. The first video gives the anchor; the second supplies the missing follow-up detail.
- PASS: One video shows the speaker focused on assembling a device at a table; another view shows a related object, person, or room state the speaker cannot see. A good question asks from the speaker's memory gap, not from the dataset or video perspective.
- FAIL: One video shows someone discussing/checking a device setup while another shows dishwashing. If no shared task or natural dependency is visible, this is only timestamp alignment and should fail.

Use FAIL for a clear violation, UNCERTAIN when the videos do not provide enough evidence to verify the check, and PASS only when the dimension is satisfied.

Video set metadata:
{video_packet_brief(packet)}

Generated QA:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return one valid JSON object only, with this exact shape:
{json.dumps(JUDGE_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_answerability_prompt(qa_item: dict[str, Any], condition: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(["A", "B", "C", "D", "E"], qa_item.get("options", []))
    )
    return f"""Answer this EgoLife multiple-choice question using only the videos provided in this condition.

Condition:
{json.dumps(condition, ensure_ascii=False, indent=2)}

Question:
{qa_item.get("question")}

Options:
{options}

Rules:
- Choose A, B, C, D, or E only if the provided videos are sufficient.
- If the condition does not contain enough evidence, set choice to "insufficient".
- Do not guess from common sense or from the answer options.
- Do not use information from users/videos that are not provided in this condition.

Return one valid JSON object only with this exact shape:
{json.dumps(ANSWERABILITY_SCHEMA, ensure_ascii=False, indent=2)}
"""
