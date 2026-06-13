"""Prompts for two-user EgoLife QA generation and review."""

from __future__ import annotations

import json
from typing import Any


GENERATION_SCHEMA = {
    "id": "string",
    "question": "natural everyday question without timestamps or words like video/footage/recording from first person (user) perspective",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "required_users": ["at least two user names"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "visual fact contributed by this user's view",
            "timeframe": "specific start-end time range or approximate moment in this user's video for generate the question",
            "frames_used": ["frame path or frame timestamp labels"],
        }
    ],
}


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question": "natural first-person question (ask from a AR glass user perspective) without timestamps or words like video/footage/recording/frame/camera",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "required_users": ["at least two user names"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "visual fact contributed by this user's own video",
            "timeframe": "specific start-end time range or approximate moment in this user's video",
            "frames_used": ["video-level evidence or approximate moment label"],
        }
    ],
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
    feedback_block = (
        "\nPrevious judger/evaluator feedback to fix:\n"
        f"{feedback}\n"
        "You must incorporate this feedback into the new question, options, answer, and evidence. "
        "Do not repeat the rejected issue.\n"
        if feedback
        else ""
    )
    return f"""You are an assistant tasked with generating one meaningful, contextually grounded MCQ from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other, or in different places. Look directly at the videos and use only visual evidence, video metadata, and the provided 2D gaze coordinates when available. Do not use captions, subtitles, transcripts, or pre-written observations.

Your job:
1. Generate exactly one five-option multiple-choice question.
2. The question_type must be "{question_type}": {type_instruction}
3. The question must require visual evidence from at least two required users.
4. Any single required user's video alone must be insufficient; the combined required users' videos must make exactly one option correct.
5. Fill the evidence field with each needed user's visual fact and a specific timeframe.

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

{feedback_block}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_judger_prompt(qa_item: dict[str, Any], packet: dict[str, Any]) -> str:
    return f"""You are a strict judger for EgoLife video-first two-user MCQ generation.

You will see the same raw egocentric videos used by the generator. Judge whether the generated question is acceptable.

Judge each dimension separately. A QA item should pass only when all blocking dimensions pass.

Blocking dimensions:
1. first_person_naturalness:
   - The question is asked from the user's first-person point of view.
   - The wording sounds like a natural everyday question someone would ask an AR or memory assistant.
   - It avoids awkward benchmark-style phrasing and does not name required users when "I", "me", "my", "we", or "our" should be used.
2. agent_perspective:
   - The question sounds like a natural first-person memory or AR-assistant question.
   - It uses "I/me/my/we/our" when a required user is the speaker.
   - It does not use dataset-observer or god-view wording.
   - It does not mention video, footage, recording, frame, dataset, camera, clip, or timestamp.
3. source_scope:
   - The answer can be judged from the provided raw videos and packet metadata only.
   - It does not rely on captions, pre-written observations, external knowledge, private thoughts, hidden identity, or off-screen facts.
4. question_type_semantics:
   - If question_type is commonality, the answer must be a shared/jointly verified fact across required users.
   - If question_type is difference, the answer must be a meaningful asymmetry or complementary detail between required users.
5. multi_video_necessity:
   - At least two required users contribute necessary, non-redundant visual evidence.
   - The question should not be answerable from one user's video alone.
   - For aligned same-time clips, the comparison should make sense as the same event/place/task or a clearly related interaction.
6. visual_grounding:
   - The correct option is visually grounded in concrete moments in the videos.
   - The per_user_evidence_claims and referred_timestamps, if present, are specific enough for a human auditor to check.
7. mcq_option_quality:
   - There are exactly five non-empty options.
   - Exactly one option is correct.
   - Distractors are plausible but not ambiguous or also correct.
8. gaze_safety:
   - Any gaze-to-object claim uses provided 2D image coordinates as human attention cues.
   - If 2D gaze coordinates are missing for a moment, the QA item must not invent gaze points or exact gaze-to-object proximity.
9. human_auditability:
   - The QA row includes enough user/video/time evidence for a human to inspect the same clips later.
   - The rationale explains why this question was asked and why each user matters.

Use FAIL for a clear violation, UNCERTAIN when the videos do not provide enough evidence to verify the check, and PASS only when the dimension is satisfied.

Evidence packet metadata:
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
