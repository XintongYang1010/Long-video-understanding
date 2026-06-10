from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from egolife_two_user_qa.evidence import group_manifest_clips, summarize_gaze_csv
from egolife_two_user_qa.manifest import parse_egolife_path, seconds_from_time_token
from egolife_two_user_qa.schema import extract_json_object, validate_qa_item


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


class SchemaTests(unittest.TestCase):
    def valid_item(self):
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
            "review": {"review_passed": True},
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "source_urls": {"videos": []},
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


if __name__ == "__main__":
    unittest.main()

