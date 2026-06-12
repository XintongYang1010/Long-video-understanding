"""Prompts for two-user EgoLife QA generation and review."""

from __future__ import annotations

import json
from typing import Any


GENERATION_SCHEMA = {
    "qa_id": "string",
    "question": "natural everyday question without timestamps or words like video/footage/recording",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "category": "social_interaction/task_coordination/theory_of_mind/temporal_reasoning/environmental_interaction",
    "required_users": ["at least two user names"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "fact contributed by this user's view",
            "frames_used": ["frame path or frame timestamp labels"],
        }
    ],
    "single_user_answerability": {
        "Jake": "insufficient because ...",
        "Alice": "insufficient because ...",
    },
    "combined_answerability": "sufficient because ...",
    "review": {
        "generator_self_check": "why this cannot be answered by one user alone",
        "status": "draft",
    },
}


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question_type": "commonality or difference",
    "question": "natural first-person question without timestamps or words like video/footage/recording/frame/camera",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "category": "social_interaction/task_coordination/theory_of_mind/temporal_reasoning/environmental_interaction",
    "required_users": ["at least two user names"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "visual fact contributed by this user's own video",
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
        "Jake": "insufficient because ...",
        "Alice": "insufficient because ...",
    },
    "combined_answerability": "sufficient because ...",
    "generator_rationale": "why this is a natural agent-perspective two-user question",
    "why_two_users_needed": "why at least two users' videos are necessary",
    "per_user_evidence_claims": [
        {"user": "name", "claim": "claim grounded in that user's own video"}
    ],
    "review": {
        "generator_self_check": "why this cannot be answered by one user alone",
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


def packet_brief(packet: dict[str, Any]) -> str:
    clips = []
    for clip in packet.get("clips", []):
        frame_lines = [
            f"{idx + 1}. {frame.get('path')} at {frame.get('timestamp_seconds')}s"
            for idx, frame in enumerate(clip.get("frames", []))
        ]
        clips.append(
            {
                "user": clip.get("agent_name"),
                "day": clip.get("day"),
                "clip_clock": clip.get("clip_clock"),
                "video_url": clip.get("video_url"),
                "gaze_summary": clip.get("gaze_summary"),
                "observation": clip.get("observation"),
                "frames": frame_lines,
            }
        )
    return json.dumps(
        {
            "evidence_id": packet.get("evidence_id"),
            "candidate_type": packet.get("candidate_type"),
            "required_users": packet.get("required_users"),
            "complementarity": packet.get("complementarity"),
            "clips": clips,
        },
        ensure_ascii=False,
        indent=2,
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
    feedback_block = f"\nPrevious judger/evaluator feedback to fix:\n{feedback}\n" if feedback else ""
    return f"""You are constructing one high-quality EgoLife multi-user video QA item.

You are given raw egocentric videos from multiple daily-life users. Look directly at the videos. Do not rely on captions or pre-written observations.

Goal:
- Create exactly one five-option multiple-choice question.
- The question_type must be "{question_type}".
- {type_instruction}
- At least two required users must be essential for answering.
- Any single required user's video alone must be insufficient to answer completely.
- The combined required users' videos must make exactly one option correct.

Instruction:
- MUST ask the question from FIRST person point of view.
- Do not use words such as video, footage, recording, frame, dataset, camera, clip, or timestamp.
- Avoid god-view wording or third person view like "what does the camera show" or "in the first person's view".
- Do not speculate about private thoughts, identities, intentions, or off-screen facts unless visually supported.
- Do not make 2D gaze-to-object claims unless projection_status is "projected"; unprojected EgoLife gaze is Aria CPF yaw/pitch/depth, not image pixels.
- Include referred_timestamps with approximate seconds within each user's clip when a specific moment supports the answer. Use an empty list if you cannot localize the moment.
{feedback_block}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_judger_prompt(qa_item: dict[str, Any], packet: dict[str, Any]) -> str:
    return f"""You are a strict judger for EgoLife video-first two-user MCQ generation.

You will see the same raw egocentric videos used by the generator. Judge whether the generated question is acceptable.

Hard pass criteria:
1. It is answerable from the provided videos and metadata, not from captions or external knowledge.
2. The wording is natural from an agent/user memory perspective.
3. It does not mention video, footage, recording, frame, dataset, camera, clip, or timestamp.
4. It matches question_type: commonality means a shared/jointly verified fact; difference means a meaningful asymmetry or complementary detail.
5. At least two required users contribute necessary evidence.
6. No single required user alone can answer completely.
7. The five options have exactly one correct answer.
8. Any gaze-to-object claim is allowed only when projection_status is "projected".

Evidence packet metadata:
{video_packet_brief(packet)}

Generated QA:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return one valid JSON object only:
{{
  "review_passed": true,
  "natural_agent_perspective_check": "PASS/FAIL with short explanation",
  "question_type_check": "PASS/FAIL with short explanation",
  "two_user_necessity_check": "PASS/FAIL with short explanation",
  "single_user_insufficiency_check": "PASS/FAIL with short explanation",
  "mcq_uniqueness_check": "PASS/FAIL with short explanation",
  "visual_grounding_check": "PASS/FAIL with short explanation",
  "gaze_claim_check": "PASS/FAIL with short explanation",
  "why_generator_asked_this": "short justification of the generator's likely reason",
  "feedback_to_generator": "specific edit instructions if review_passed is false; empty string if passed"
}}
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


def build_generation_prompt(packet: dict[str, Any]) -> str:
    return f"""You are constructing a high-quality multi-user egocentric video QA item.

Goal:
- Create exactly one multiple-choice question that requires evidence from at least two EgoLife users.
- A single user's frames must be insufficient to answer the question completely.
- The combined evidence from the required users must make exactly one option correct.
- Use the complementarity notes: the question should combine one distinct fact from each required user.

Style constraints:
- The question should sound like a natural daily-life memory question asked to an AR/VR assistant.
- Do not mention timestamps, video, footage, recording, frames, dataset, or camera.
- Use first-person phrasing when natural ("Where did I...", "What did we...").
- Avoid private speculation that is not visually supported.
- Treat EgoLife gaze CSV values as Aria CPF yaw/pitch/depth. Only use 2D gaze-to-object claims if `projection_status` is `projected`; if it is `missing_calibration`, do not claim image-pixel gaze or bbox proximity.

Evidence packet:
{packet_brief(packet)}

Use only the provided images, observations, complementarity notes, and packet metadata. If the evidence is not enough, still return JSON but set review.status to "reject_insufficient_evidence" and explain why.

Return one valid JSON object only, with this exact shape:
{json.dumps(GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_review_prompt(qa_item: dict[str, Any], packet: dict[str, Any]) -> str:
    return f"""You are reviewing a generated EgoLife two-user MCQ.

Hard pass criteria:
1. At least two required_users are essential.
2. No single required user can answer the question completely alone.
3. The combined evidence from required users supports exactly one correct option.
4. The wording is natural and does not mention timestamps, video, footage, recording, frames, dataset, or camera.
5. There are exactly five options and correct is one of A/B/C/D/E.
6. Any gaze-to-object claim uses projected 2D gaze only when `projection_status` is `projected`; unprojected CPF angle summaries are not treated as image pixels.

Evidence packet:
{packet_brief(packet)}

Generated QA:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return one valid JSON object only:
{{
  "qa_id": "{qa_item.get('qa_id', '')}",
  "review_passed": true,
  "fact_verification": "PASS/FAIL with short explanation",
  "single_user_check": "PASS/FAIL with short explanation",
  "combined_user_check": "PASS/FAIL with short explanation",
  "mcq_check": "PASS/FAIL with short explanation",
  "wording_check": "PASS/FAIL with short explanation",
  "modification_suggestions": ""
}}
"""
