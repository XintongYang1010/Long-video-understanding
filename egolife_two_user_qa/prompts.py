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
