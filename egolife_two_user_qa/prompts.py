"""Prompts for two-user EgoLife QA generation and review."""

from __future__ import annotations

import json
from typing import Any


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question_type": "commonality or difference",
    "question": "natural first-person question (ask from a AR glass user perspective) without timestamps or words like video/footage/recording/frame/camera",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "category": "social_interaction/task_coordination/theory_of_mind/temporal_reasoning/environmental_interaction",
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
) -> str:
    type_instruction = {
        "commonality": (
            "Create a commonality question: the answer should identify something that is shared, "
            "jointly established, or mutually verified across the required users' egocentric videos."
        ),
        "difference": (
            "Create a difference question: the answer should identify a meaningful difference, "
            "asymmetry, or complementary detail between the required users' egocentric videos."
        ),
    }[question_type]
    feedback_block = (
        "\nPrevious judger/evaluator feedback to fix:\n"
        f"{feedback}\n"
        "You must incorporate this feedback into the new question, options, answer, and evidence. "
        "Do not repeat the rejected issue.\n"
        if feedback
        else ""
    )
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
- Do not make a question just because clips share a timestamp.
- Do not ask what both users saw, noticed, or looked at.
- Do not ask a generic comparison of two views, rooms, or camera angles.
"""
    return f"""You are an assistant tasked with generating one meaningful, contextually grounded MCQ from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other, or in different places. Look directly at the videos and use only visual evidence, video metadata, and the provided 2D gaze coordinates when available. Do not use captions, subtitles, transcripts, or pre-written observations.

Your job:
1. Generate exactly one five-option multiple-choice question.
2. The question_type must be "{question_type}": {type_instruction}
3. The question must require visual evidence from at least two required users.
4. Any single required user's video alone must be insufficient; the combined required users' videos must make exactly one option correct.
5. Fill the evidence field with each needed user's visual fact and a specific timeframe.
6. Return every field in the JSON shape exactly. Do not omit category, single_user_answerability, combined_answerability, generator_rationale, why_two_users_needed, per_user_evidence_claims, referred_timestamps, or review.
7. The answer field must exactly equal the text of options[correct].

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
- FAIL if the question merely stitches together two clips because they share a time interval.
- FAIL if the activities are unrelated, such as one person discussing/checking a device while another person is washing dishes, unless the question identifies a concrete shared task or natural dependency.
- FAIL if the question asks what both users saw, both noticed, or both looked at; do not ask what both users saw or noticed because one user may not know the other user's perception.
- FAIL if the question is a generic comparison of two views, rooms, or camera angles rather than a speaker anchor plus missing visual detail.
- FAIL if a single user's video already reveals the correct answer.
- UNCERTAIN if the videos do not clearly show the anchor, the missing visual detail, or the relation between them.
- In the reason, explicitly name the speaker-side anchor, the missing visual detail, and why the second video is or is not needed.

Contrastive example for multi_video_necessity:
- PASS: One video shows the speaker checking a setup and leaving toward a stairwell; another video still shows the front of that room where a tutorial continues. A good question asks what was still happening after the speaker left. The first video gives the anchor; the second supplies the missing follow-up detail.
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
