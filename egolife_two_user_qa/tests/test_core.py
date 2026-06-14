from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from egolife_two_user_qa.candidate_mining import mine_candidates
from egolife_two_user_qa.evidence import group_manifest_clips, summarize_gaze_csv
from egolife_two_user_qa.gaze_projection import gaussian_bbox_score, load_aria_projection_calibration, project_gaze_row
from egolife_two_user_qa.manifest import parse_egolife_path, seconds_from_time_token
from egolife_two_user_qa.prompts import build_video_generation_prompt
from egolife_two_user_qa.qwen3vl_runner import DryRunRunner, normalize_video_kwargs, split_video_inputs_and_metadata
from egolife_two_user_qa.schema import extract_json_object, validate_qa_item
from egolife_two_user_qa.video_qa_loop import (
    answerability_gate,
    build_review_from_gates,
    complete_generator_metadata,
    dry_run_qa,
    judge_gate,
    qa_for_judger_prompt,
)


class ManifestTests(unittest.TestCase):
    def test_parse_video_path(self) -> None:
        parsed = parse_egolife_path("A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4")
        self.assertEqual(parsed.day, "DAY1")
        self.assertEqual(parsed.agent_dir, "A1_JAKE")
        self.assertEqual(parsed.agent_name, "Jake")
        self.assertEqual(parsed.time_token, "11094208")
        self.assertEqual(parsed.clip_clock, "11:09:42.08")

    def test_seconds_from_time_token(self) -> None:
        self.assertAlmostEqual(seconds_from_time_token("11094208"), 40182.08)

    def test_group_manifest_clips(self) -> None:
        manifest = {
            "clips": [
                {"day": "DAY1", "time_token": "11100000", "agent_dir": "A1_JAKE"},
                {"day": "DAY1", "time_token": "11100000", "agent_dir": "A2_ALICE"},
                {"day": "DAY1", "time_token": "11103000", "agent_dir": "A1_JAKE"},
            ]
        }
        groups = group_manifest_clips(manifest)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["agents"], ["A1_JAKE", "A2_ALICE"])


class EvidenceTests(unittest.TestCase):
    def test_summarize_gaze_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gaze.csv"
            path.write_text(
                "tracking_timestamp_us,left_yaw_rads_cpf,right_yaw_rads_cpf,pitch_rads_cpf,depth_m\n"
                "1,0.1,0.3,-0.2,1.5\n"
                "2,0.2,0.4,-0.1,2.0\n",
                encoding="utf-8",
            )
            summary = summarize_gaze_csv(path)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["yaw_rads_summary"]["median"], 0.25)
        self.assertEqual(summary["depth_m_summary"]["max"], 2.0)
        self.assertEqual(summary["projection_status"], "missing_calibration")
        self.assertIsNone(summary["projected_gaze_summary"])

    def test_project_gaze_with_explicit_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = Path(tmp) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "camera": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
                        "T_camera_cpf": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            calibration = load_aria_projection_calibration(calibration_path)
            projected = project_gaze_row(
                {
                    "tracking_timestamp_us": "1",
                    "left_yaw_rads_cpf": "0.0",
                    "right_yaw_rads_cpf": "0.0",
                    "pitch_rads_cpf": "0.0",
                    "depth_m": "2.0",
                },
                calibration,
            )
        self.assertIsNotNone(projected)
        self.assertEqual(projected["x"], 320.0)
        self.assertEqual(projected["y"], 240.0)
        self.assertTrue(projected["in_frame"])

    def test_summarize_gaze_projects_only_with_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gaze_path = Path(tmp) / "gaze.csv"
            gaze_path.write_text(
                "tracking_timestamp_us,left_yaw_rads_cpf,right_yaw_rads_cpf,pitch_rads_cpf,depth_m\n"
                "1,0.0,0.0,0.0,2.0\n",
                encoding="utf-8",
            )
            calibration_path = Path(tmp) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "camera": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
                        "T_camera_cpf": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_gaze_csv(gaze_path, calibration_path=calibration_path)
        self.assertEqual(summary["projection_status"], "projected")
        self.assertEqual(summary["projected_gaze_summary"]["median_x"], 320.0)
        self.assertEqual(summary["projected_gaze_summary"]["median_y"], 240.0)

    def test_gaussian_bbox_score_prefers_near_center(self) -> None:
        near = gaussian_bbox_score((10.0, 10.0), (8.0, 8.0, 12.0, 12.0), sigma=10.0)
        far = gaussian_bbox_score((10.0, 10.0), (100.0, 100.0, 120.0, 120.0), sigma=10.0)
        self.assertGreater(near, far)


class CandidateMiningTests(unittest.TestCase):
    def test_mine_candidates_from_complementary_observations(self) -> None:
        rows = [
            {
                "clip_id": "DAY1_A1_JAKE_11100000",
                "clip": {
                    "clip_id": "DAY1_A1_JAKE_11100000",
                    "day": "DAY1",
                    "agent_dir": "A1_JAKE",
                    "agent_id": "A1",
                    "agent_name": "Jake",
                    "time_token": "11100000",
                    "clip_clock": "11:10:00.00",
                    "clock_seconds": 40200.0,
                    "video_url": "video_a",
                    "gaze_url": "gaze_a",
                    "frames": [],
                    "gaze_summary": {},
                },
                "observation": {
                    "status": "ok",
                    "location_guess": "kitchen table",
                    "visible_people": ["Alice"],
                    "salient_objects": ["red mug", "table"],
                    "actions": ["Jake sees Alice pick up the red mug"],
                    "gaze_focus": ["red mug"],
                    "key_facts": ["Alice picks up the red mug near the kitchen table"],
                },
            },
            {
                "clip_id": "DAY1_A2_ALICE_11100000",
                "clip": {
                    "clip_id": "DAY1_A2_ALICE_11100000",
                    "day": "DAY1",
                    "agent_dir": "A2_ALICE",
                    "agent_id": "A2",
                    "agent_name": "Alice",
                    "time_token": "11100000",
                    "clip_clock": "11:10:00.00",
                    "clock_seconds": 40200.0,
                    "video_url": "video_b",
                    "gaze_url": "gaze_b",
                    "frames": [],
                    "gaze_summary": {},
                },
                "observation": {
                    "status": "ok",
                    "location_guess": "kitchen table",
                    "visible_people": ["Jake"],
                    "salient_objects": ["red mug", "sink"],
                    "actions": ["Alice places the red mug beside the sink"],
                    "gaze_focus": ["sink"],
                    "key_facts": ["The red mug ends up beside the sink"],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            obs_path = Path(tmp) / "observations.jsonl"
            out_path = Path(tmp) / "candidates.jsonl"
            obs_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            candidates = mine_candidates(
                observations_path=obs_path,
                output_path=out_path,
                target_count=1,
                min_score=0,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_type"], "semantic_complementarity")
        self.assertEqual(candidates[0]["required_users"], ["Jake", "Alice"])
        self.assertIn("complementarity", candidates[0])


class SchemaTests(unittest.TestCase):
    def passed_judge_checks(self):
        return {
            name: {"status": "PASS", "reason": "ok", "fix": ""}
            for name in [
                "first_person_naturalness",
                "agent_perspective",
                "source_scope",
                "question_type_semantics",
                "multi_video_necessity",
                "visual_grounding",
                "mcq_option_quality",
                "gaze_safety",
                "human_auditability",
            ]
        }

    def valid_item(self):
        judger = {
            "review_passed": True,
            "checks": self.passed_judge_checks(),
            "blocking_failures": [],
            "feedback_to_generator": "",
            "gate": {"passed": True, "reason": "all structured judger checks passed", "failed_checks": []},
        }
        answerability = {
            "evaluations": [
                {"condition_id": "single_user::Jake", "condition_type": "single_user", "choice": "insufficient"},
                {"condition_id": "single_user::Alice", "condition_type": "single_user", "choice": "B"},
                {"condition_id": "combined_all_users::Jake+Alice", "condition_type": "combined_all_users", "choice": "A"},
            ],
            "gate": {"passed": True, "reason": "combined correct and singles insufficient or wrong"},
        }
        return {
            "qa_id": "QA_001",
            "question": "What did we put near the table?",
            "options": ["A cup", "A plate", "A book", "A phone", "A key"],
            "correct": "A",
            "answer": "A cup",
            "category": "environmental_interaction",
            "required_users": ["Jake", "Alice"],
            "evidence": [
                {"user": "Jake", "needed_fact": "saw the cup", "frames_used": ["f1"]},
                {"user": "Alice", "needed_fact": "saw the table", "frames_used": ["f2"]},
            ],
            "single_user_answerability": {
                "Jake": "insufficient because he only saw the object",
                "Alice": "insufficient because she only saw the destination",
            },
            "combined_answerability": "sufficient because together they support the answer",
            "review": {
                "status": "passed",
                "review_passed": True,
                "judger": judger,
                "answerability": answerability,
                "schema_validation": {"passed": True, "errors": []},
                "final_decision": {
                    "accepted": True,
                    "rejection_stage": None,
                    "reason": "passed all gates",
                },
            },
            "question_type": "commonality",
            "generator_rationale": "The question is natural and grounded in both users' views.",
            "why_two_users_needed": "Jake and Alice each provide a necessary visual fact.",
            "per_user_evidence_claims": [
                {"user": "Jake", "claim": "Jake saw the cup"},
                {"user": "Alice", "claim": "Alice saw the table"},
            ],
            "attempt_count": 1,
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "source_urls": {"videos": []},
            "video_evidence": [
                {
                    "user": "Jake",
                    "day": "DAY1",
                    "time_token": "11100000",
                    "video_url": "video_a",
                    "local_video": "jake.mp4",
                    "sampled_frames": [],
                }
            ],
            "referred_timestamps": [],
            "human_audit": {"evidence_id": "E1", "video_evidence": []},
            "generation_trace": [
                {
                    "attempt": 1,
                    "generation": {"prompt": "p", "raw_output": "{}"},
                    "judge": {"prompt": "j", "raw_output": "{}"},
                    "answerability": {},
                    "result": {"accepted": True},
                }
            ],
        }

    def test_validate_valid_item(self) -> None:
        self.assertEqual(validate_qa_item(self.valid_item(), strict_review=True), [])

    def test_validate_requires_two_users(self) -> None:
        item = self.valid_item()
        item["required_users"] = ["Jake"]
        errors = validate_qa_item(item)
        self.assertTrue(any("at least two" in error for error in errors))

    def test_extract_json_object_from_codeblock(self) -> None:
        self.assertEqual(extract_json_object("```json\n{\"a\": 1}\n```"), {"a": 1})

    def test_strict_validation_requires_video_first_fields(self) -> None:
        item = self.valid_item()
        del item["question_type"]
        errors = validate_qa_item(item, strict_review=True)
        self.assertTrue(any("missing video-first fields" in error for error in errors))

    def test_difference_question_type_validates(self) -> None:
        item = self.valid_item()
        item["question_type"] = "difference"
        self.assertEqual(validate_qa_item(item, strict_review=True), [])

    def test_strict_validation_requires_structured_judge_checks(self) -> None:
        item = self.valid_item()
        item["review"]["judger"] = {"review_passed": True, "gate": {"passed": True}}
        errors = validate_qa_item(item, strict_review=True)
        self.assertTrue(any("review.judger.checks" in error for error in errors))

    def test_strict_validation_uses_review_not_top_level_review_fields(self) -> None:
        item = self.valid_item()
        self.assertNotIn("judge_feedback", item)
        self.assertNotIn("answerability_eval", item)
        self.assertEqual(validate_qa_item(item, strict_review=True), [])

    def test_strict_validation_requires_review_answerability_gate(self) -> None:
        item = self.valid_item()
        item["review"]["answerability"]["gate"] = {"passed": False, "reason": "single user leaked answer"}
        errors = validate_qa_item(item, strict_review=True)
        self.assertTrue(any("review.answerability.gate.passed" in error for error in errors))


class VideoFirstTests(unittest.TestCase):
    def test_dry_run_runner_accepts_video_paths(self) -> None:
        raw = DryRunRunner().generate("prompt", image_paths=["a.jpg"], video_paths=["a.mp4", "b.mp4"])
        parsed = json.loads(raw)
        self.assertEqual(parsed["image_count"], 1)
        self.assertEqual(parsed["video_count"], 2)

    def test_normalize_video_kwargs_collapses_fps_list(self) -> None:
        self.assertEqual(normalize_video_kwargs({"fps": [1.0, 1.0]})["fps"], 1.0)
        self.assertEqual(normalize_video_kwargs({"fps": []})["fps"], 1.0)
        self.assertEqual(normalize_video_kwargs({"fps": 2.0})["fps"], 2.0)

    def test_split_video_inputs_and_metadata(self) -> None:
        video = object()
        metadata = {"fps": 30.0, "frames_indices": [0, 15], "total_num_frames": 60}
        videos, kwargs = split_video_inputs_and_metadata([(video, metadata)], {"fps": [1.0]})
        self.assertEqual(videos, [video])
        self.assertTrue(kwargs["return_metadata"])
        self.assertEqual(kwargs["fps"], 1.0)
        self.assertEqual(kwargs["video_metadata"][0].fps, 30.0)
        self.assertEqual(kwargs["video_metadata"][0].frames_indices, [0, 15])

    def test_video_generation_prompt_does_not_use_observation(self) -> None:
        packet = {
            "evidence_id": "E1",
            "required_users": ["Jake", "Alice"],
            "clips": [
                {
                    "agent_name": "Jake",
                    "day": "DAY1",
                    "clip_clock": "11:10:00.00",
                    "local_video": "jake.mp4",
                    "video_url": "video_a",
                    "observation": {"key_facts": ["SHOULD_NOT_APPEAR"]},
                    "gaze_summary": {"projection_status": "missing_calibration"},
                },
                {
                    "agent_name": "Alice",
                    "day": "DAY1",
                    "clip_clock": "11:10:00.00",
                    "local_video": "alice.mp4",
                    "video_url": "video_b",
                    "observation": {"key_facts": ["SHOULD_NOT_APPEAR"]},
                    "gaze_summary": {"projection_status": "missing_calibration"},
                },
            ],
        }
        prompt = build_video_generation_prompt(packet, "commonality")
        self.assertIn("Look directly at the videos", prompt)
        self.assertIn("local_video", prompt)
        self.assertNotIn("SHOULD_NOT_APPEAR", prompt)
        self.assertIn("single_user_answerability", prompt)
        self.assertIn("combined_answerability", prompt)
        self.assertIn("why_two_users_needed", prompt)

    def test_complete_generator_metadata_repairs_old_generator_shape(self) -> None:
        packet = {"required_users": ["Jake", "Alice"]}
        qa = {
            "qa_id": "Q1",
            "question": "After I left the table, what was still happening there?",
            "options": ["food prep", "phone charging", "dish washing", "door opening", "bag packing"],
            "correct": "A",
            "answer": "wrong text",
            "required_users": ["Jake", "Alice"],
            "evidence": [
                {
                    "user": "Jake",
                    "needed_fact": "Jake left the table.",
                    "timeframe": "early in the clip",
                    "frames_used": ["around 5s"],
                }
            ],
            "model_id": "dry-run",
            "source_urls": {},
        }
        complete_generator_metadata(qa, packet=packet, question_type="commonality")
        self.assertEqual(qa["answer"], "food prep")
        self.assertEqual(qa["question_type"], "commonality")
        self.assertIn("insufficient", qa["single_user_answerability"]["Jake"])
        self.assertIn("sufficient", qa["combined_answerability"])
        self.assertEqual(validate_qa_item(qa), [])

    def test_answerability_gate_requires_combined_correct_and_singles_not_correct(self) -> None:
        qa = {"correct": "A"}
        passed = answerability_gate(
            qa,
            [
                {"condition_id": "single_user::Jake", "condition_type": "single_user", "choice": "insufficient"},
                {"condition_id": "single_user::Alice", "condition_type": "single_user", "choice": "B"},
                {"condition_id": "combined_all_users::Jake+Alice", "condition_type": "combined_all_users", "choice": "A"},
            ],
        )
        self.assertTrue(passed["passed"])
        failed = answerability_gate(
            qa,
            [
                {"condition_id": "single_user::Jake", "condition_type": "single_user", "choice": "A"},
                {"condition_id": "combined_all_users::Jake+Alice", "condition_type": "combined_all_users", "choice": "A"},
            ],
        )
        self.assertFalse(failed["passed"])

    def test_judge_gate_requires_each_structured_check_to_pass(self) -> None:
        checks = {
            name: {"status": "PASS", "reason": "ok", "fix": ""}
            for name in [
                "first_person_naturalness",
                "agent_perspective",
                "source_scope",
                "question_type_semantics",
                "multi_video_necessity",
                "visual_grounding",
                "mcq_option_quality",
                "gaze_safety",
                "human_auditability",
            ]
        }
        self.assertTrue(judge_gate({"review_passed": True, "checks": checks})["passed"])
        checks["multi_video_necessity"] = {
            "status": "FAIL",
            "reason": "one video already answers",
            "fix": "ask for a complementary clue from the second user",
        }
        failed = judge_gate({"review_passed": True, "checks": checks})
        self.assertFalse(failed["passed"])
        self.assertIn("multi_video_necessity", failed["failed_checks"])

    def test_judge_gate_blocks_unnatural_non_first_person_question(self) -> None:
        checks = {
            name: {"status": "PASS", "reason": "ok", "fix": ""}
            for name in [
                "first_person_naturalness",
                "agent_perspective",
                "source_scope",
                "question_type_semantics",
                "multi_video_necessity",
                "visual_grounding",
                "mcq_option_quality",
                "gaze_safety",
                "human_auditability",
            ]
        }
        checks["first_person_naturalness"] = {
            "status": "FAIL",
            "reason": "question names Jake and Alice instead of asking from first person",
            "fix": "rewrite as a natural everyday first-person question using I or we",
        }
        failed = judge_gate({"review_passed": True, "checks": checks})
        self.assertFalse(failed["passed"])
        self.assertIn("first_person_naturalness", failed["failed_checks"])

    def test_build_review_from_gates_for_accepted_row(self) -> None:
        review = build_review_from_gates(
            judge={"review_passed": True, "gate": {"passed": True}},
            answerability={"gate": {"passed": True}, "evaluations": []},
            schema_errors=[],
            accepted=True,
            final_reason="passed all gates",
        )
        self.assertEqual(review["status"], "passed")
        self.assertTrue(review["review_passed"])
        self.assertTrue(review["judger"]["gate"]["passed"])
        self.assertTrue(review["answerability"]["gate"]["passed"])

    def test_build_review_from_gates_for_judger_rejection(self) -> None:
        review = build_review_from_gates(
            judge={
                "review_passed": False,
                "feedback_to_generator": "ask from the speaker's own memory gap",
                "gate": {"passed": False, "reason": "not first-person"},
            },
            answerability=None,
            schema_errors=[],
            accepted=False,
            rejection_stage="judger",
            final_reason="not first-person",
        )
        self.assertEqual(review["status"], "rejected_by_judger")
        self.assertFalse(review["review_passed"])
        self.assertEqual(review["final_decision"]["rejection_stage"], "judger")

    def test_build_review_from_gates_for_answerability_rejection(self) -> None:
        review = build_review_from_gates(
            judge={"review_passed": True, "gate": {"passed": True}},
            answerability={"gate": {"passed": False, "reason": "single user answered correctly"}},
            schema_errors=[],
            accepted=False,
            rejection_stage="answerability",
            final_reason="single user answered correctly",
        )
        self.assertEqual(review["status"], "rejected_by_answerability")
        self.assertFalse(review["answerability"]["gate"]["passed"])

    def test_build_review_from_gates_for_schema_rejection(self) -> None:
        review = build_review_from_gates(
            judge={"review_passed": True, "gate": {"passed": True}},
            answerability={"gate": {"passed": True}, "evaluations": []},
            schema_errors=["answer must equal options[correct]"],
            accepted=False,
            rejection_stage="schema",
            final_reason="strict validation failed",
        )
        self.assertEqual(review["status"], "rejected_by_schema")
        self.assertFalse(review["schema_validation"]["passed"])
        self.assertEqual(review["schema_validation"]["errors"], ["answer must equal options[correct]"])

    def test_dry_run_qa_includes_video_evidence_provenance(self) -> None:
        qa = dry_run_qa(
            {
                "evidence_id": "E1",
                "required_users": ["Jake", "Alice"],
                "source_urls": {"videos": ["video_a", "video_b"]},
                "clips": [
                    {
                        "agent_name": "Jake",
                        "agent_dir": "A1_JAKE",
                        "agent_id": "A1",
                        "day": "DAY1",
                        "time_token": "11100000",
                        "clip_clock": "11:10:00.00",
                        "duration_seconds": 30.0,
                        "video_url": "video_a",
                        "local_video": "jake.mp4",
                        "frames": [{"timestamp_seconds": 10.0, "path": "jake_10.jpg"}],
                    }
                ],
            },
            "commonality",
        )
        self.assertEqual(qa["video_evidence"][0]["user"], "Jake")
        self.assertEqual(qa["video_evidence"][0]["video_url"], "video_a")
        self.assertEqual(qa["video_evidence"][0]["sampled_frames"][0]["timestamp_seconds"], 10.0)
        self.assertEqual(qa["referred_timestamps"], [])
        self.assertIn("human_audit", qa)
        self.assertIn("generation_trace", qa)
        self.assertEqual(qa["generation_trace"][0]["stage"], "dry_run")

    def test_qa_for_judger_prompt_excludes_trace_fields(self) -> None:
        item = SchemaTests("test_validate_valid_item").valid_item()
        compact = qa_for_judger_prompt(item)
        self.assertIn("question", compact)
        self.assertIn("evidence", compact)
        self.assertNotIn("generation_trace", compact)
        self.assertNotIn("human_audit", compact)
        self.assertNotIn("video_evidence", compact)
        self.assertNotIn("source_urls", compact)


if __name__ == "__main__":
    unittest.main()
