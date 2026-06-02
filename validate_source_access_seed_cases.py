#!/usr/bin/env python3
"""First-pass validation for AR/VR source-access seed cases.

This is intentionally metadata-first:
- no video downloads
- no frame extraction
- no VLM inference
- no final annotation claims
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SEED_CSV = ROOT / "source_access_seed_cases_v0_3.csv"
CASTLE_INV = ROOT / "castle_modality_inventory.csv"
CASTLE_COUNTS = ROOT / "castle_modality_counts.csv"
CASTLE_FILE_INV = ROOT / "castle_poc" / "castle_file_inventory.csv"

SCHEMA = [
    "case_id",
    "question",
    "bucket",
    "answer_options",
    "proposed_route",
    "verified_route",
    "status",
    "caption_sufficient",
    "transcript_needed",
    "transcript_found",
    "raw_frame_needed",
    "raw_frame_found",
    "static_camera_needed",
    "external_user_candidate",
    "external_user_verified_needed",
    "auxiliary_needed",
    "evidence_source_paths",
    "evidence_timestamp_or_window",
    "supporting_evidence_summary",
    "cheaper_route_checked",
    "why_minimal_route",
    "confidence",
    "next_action",
]

PEOPLE = {
    "allie",
    "bao",
    "bjorn",
    "cathal",
    "florian",
    "klaus",
    "linh",
    "luca",
    "onanong",
    "stevan",
    "tien",
    "werner",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "who",
    "how",
    "many",
    "was",
    "were",
    "did",
    "does",
    "when",
    "where",
    "which",
    "that",
    "this",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "there",
    "their",
    "during",
    "while",
    "first",
    "second",
    "third",
    "round",
    "day",
    "morning",
    "answer",
    "options",
    "him",
    "her",
    "his",
    "she",
    "they",
    "them",
    "are",
    "had",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = SCHEMA) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", norm(text)) if len(t) >= 3 and t not in STOPWORDS]


def people_in(text: str) -> list[str]:
    found = sorted(set(tokens(text)) & PEOPLE)
    # The inventory uses Onanong; questions may spell the name as Ononang in some files.
    if "ononang" in text.lower() and "onanong" not in found:
        found.append("onanong")
    return found


def day_hint(text: str) -> str | None:
    t = norm(text)
    if "day 1" in t or "first day" in t:
        return "day1"
    if "day 2" in t or "second day" in t:
        return "day2"
    if "day 3" in t or "third day" in t:
        return "day3"
    if "day 4" in t or "fourth day" in t:
        return "day4"
    return None


def time_window_hint(text: str) -> str:
    day = day_hint(text)
    pieces = []
    if day:
        pieces.append(day.upper())
    t = norm(text)
    if "morning" in t:
        pieces.append("morning candidate, likely hours 08-12 if using CASTLE hourly files")
    elif "evening" in t:
        pieces.append("evening candidate")
    elif "first round" in t:
        pieces.append("event window unknown: first round")
    elif "potential power outage" in t:
        pieces.append("event window unknown: potential power outage")
    elif "first tray" in t:
        pieces.append("event window unknown: first tray")
    if not pieces:
        return "unknown; needs event/timestamp lookup before content access"
    return "; ".join(pieces)


def location_cameras(question: str) -> list[str]:
    t = norm(question)
    cams: list[str] = []
    if any(x in t for x in ["kitchen", "ginger", "lasagna", "pudding", "oven", "carrots", "center piece"]):
        cams.append("Kitchen")
    if any(x in t for x in ["couch", "christmas", "tree", "uno", "chess", "octopus", "lights", "paintings"]):
        cams.extend(["Living1", "Living2"])
    if any(x in t for x in ["meeting", "workshop", "camera", "tower", "table", "model airplane"]):
        cams.append("Meeting")
    # Preserve order while de-duplicating.
    out: list[str] = []
    for cam in cams:
        if cam not in out:
            out.append(cam)
    return out


def question_type(case: dict[str, str]) -> str:
    q = norm(case["question"])
    bucket = case["primary_category"]
    if q.startswith("how many"):
        return "count_quantity"
    if q.startswith("who"):
        return "person_identity_or_action_attribution"
    if q.startswith("where"):
        return "spatial_location"
    if "brand" in q or "number is on" in q or "colored" in q or "color" in q:
        return "fine_visual_or_ocr_detail"
    if bucket == "audio_speech":
        return "speech_or_conversation_fact"
    if bucket == "multi_user_external_source_candidate":
        return "multi_user_event_attribution"
    return bucket


def local_transcript_files() -> list[Path]:
    candidates: list[Path] = []
    # CASTLE transcript content is expected under raw dataset paths if present.
    for base in [ROOT / "main", ROOT / "castle_poc" / "main", ROOT / "castle_hpc" / "main"]:
        if base.exists():
            candidates.extend(base.glob("**/transcript/*.json"))
    # Also accept explicitly named top-level transcript JSON files without
    # recursively traversing large cache trees.
    for p in ROOT.glob("*transcript*.json"):
        if p not in candidates:
            candidates.append(p)
    return sorted(candidates)


def search_transcripts(case: dict[str, str], transcript_files: list[Path]) -> tuple[bool, list[str], str]:
    if not transcript_files:
        return False, [], "No local CASTLE transcript JSON content files found; only transcript paths are available in the inventory."
    query_terms = tokens(case["question"] + " " + case["answer_options"])
    people = people_in(case["question"])
    strong_terms = [t for t in query_terms if t not in people][:12]
    matches: list[str] = []
    snippets: list[str] = []
    for path in transcript_files:
        if len(matches) >= 8:
            break
        text = path.read_text(errors="ignore")
        text_l = text.lower()
        score = sum(1 for term in strong_terms if term in text_l) + sum(2 for name in people if name in text_l)
        if score >= 2:
            matches.append(str(path.relative_to(ROOT)))
            snippet = compact_snippet(text, strong_terms + people)
            if snippet:
                snippets.append(snippet)
    if matches:
        return True, matches, "Local transcript keyword hit(s): " + " | ".join(snippets[:3])
    return False, [], "Local transcript files were searched, but no keyword hit was found."


def compact_snippet(text: str, terms: list[str]) -> str:
    text_l = text.lower()
    pos = -1
    for term in terms:
        pos = text_l.find(term.lower())
        if pos >= 0:
            break
    if pos < 0:
        return ""
    start = max(0, pos - 80)
    end = min(len(text), pos + 160)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def index_inventory(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    idx: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["path"].endswith(".sha256"):
            continue
        idx[row["inferred_modality"]].append(row)
    return idx


def path_matches(row: dict[str, str], names: list[str], cameras: list[str], day: str | None) -> int:
    p = row["path"].lower()
    score = 0
    for name in names:
        if f"/{name.lower()}/" in p or f"participant={name.lower()}" in row.get("participant_or_camera_id_if_parseable", "").lower():
            score += 3
    for cam in cameras:
        if f"/{cam.lower()}/" in p or f"participant={cam.lower()}" in row.get("participant_or_camera_id_if_parseable", "").lower():
            score += 3
    if day and f"/{day.lower()}/" in p:
        score += 2
    return score


def sample_inventory_paths(
    idx: dict[str, list[dict[str, str]]],
    modalities: list[str],
    names: list[str],
    cameras: list[str],
    day: str | None,
    limit: int = 10,
) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for priority, mod in enumerate(modalities):
        for row in idx.get(mod, []):
            score = path_matches(row, names, cameras, day)
            # Keep generic static/audio examples if no participant/location clue exists.
            allow_generic = mod in {"ego_video", "photos", "auxiliary_video"}
            if score > 0 or (not names and not cameras and allow_generic and len(scored) < limit * 3):
                scored.append((score, priority, row["path"]))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    out: list[str] = []
    for _, _, path in scored:
        if path not in out:
            out.append(path)
        if len(out) >= limit:
            break
    return out


def availability_summary(
    idx: dict[str, list[dict[str, str]]],
    names: list[str],
    cameras: list[str],
    day: str | None,
) -> str:
    mods = [
        "ego_video",
        "static_camera_video",
        "transcript_audio",
        "auxiliary_video",
        "photos",
        "thermal",
        "heartrate",
        "gaze",
        "pose_imu_trajectory",
    ]
    use_match = bool(names or cameras or day)
    parts: list[str] = []
    for mod in mods:
        rows = idx.get(mod, [])
        if use_match:
            count = sum(1 for row in rows if path_matches(row, names, cameras, day) > 0)
        else:
            count = len(rows)
        parts.append(f"{mod}={count}")
    return ", ".join(parts)


def route_validation(
    case: dict[str, str],
    idx: dict[str, list[dict[str, str]]],
    transcript_files: list[Path],
) -> dict[str, Any]:
    q = case["question"]
    t = norm(q)
    bucket = case["primary_category"]
    proposed = case["likely_minimal_evidence_route"]
    qtype = question_type(case)
    names = people_in(q)
    cameras = location_cameras(q)
    day = day_hint(q)
    transcript_case = bucket == "audio_speech" or any(x in t for x in ["suggest", "asked", "according to", "want", "live from"])
    count_or_detail = qtype in {"count_quantity", "fine_visual_or_ocr_detail"}
    spatial = bucket == "spatial_location_pose_fov" or qtype == "spatial_location"
    multi = bucket == "multi_user_external_source_candidate" or qtype == "multi_user_event_attribution"

    transcript_found, transcript_paths, transcript_summary = (False, [], "Transcript not searched for this non-speech case.")
    outdoor_or_front = any(x in t for x in ["outside", "front door", "around the house", "cars are parked"])

    if transcript_case:
        transcript_found, transcript_paths, transcript_summary = search_transcripts(case, transcript_files)

    static_available = bool(cameras and sample_inventory_paths(idx, ["static_camera_video", "transcript_audio"], [], cameras, day, 4))
    static_camera_needed = bool((spatial or multi or count_or_detail) and static_available and not any(x in t for x in ["outside", "front door", "around the house", "cars are parked"]))
    external_candidate = "external_user" in proposed or multi or len(names) >= 2

    raw_frame_needed = False
    auxiliary_needed = False
    caption_sufficient = "unknown"
    verified_route = "self_caption"
    why = ""
    next_action = ""
    status = "candidate_only"
    confidence = "medium"
    cheaper_route_checked = "inventory paths"

    if transcript_case:
        raw_frame_needed = False
        if transcript_found:
            verified_route = "self_audio"
            status = "verified"
            caption_sufficient = "no"
            cheaper_route_checked = "local transcript keyword search"
            why = "Speech/conversation fact should be answered from transcript before any visual frame request."
            next_action = "Manually inspect matched transcript windows; do not extract frames unless transcript is ambiguous."
            confidence = "medium"
        else:
            verified_route = "transcript_audio"
            status = "candidate_only"
            caption_sufficient = "no"
            why = "Question asks for spoken suggestion, request, or stated fact; transcript/audio is cheaper than frames."
            next_action = "Locate raw CASTLE transcript JSON for the candidate event window and search before requesting video frames."
            confidence = "low"
    elif count_or_detail or spatial:
        raw_frame_needed = True
        caption_sufficient = "unlikely"
        if static_camera_needed:
            verified_route = "static_room_source;self_raw_frame"
            status = "weakened" if "external_user_view" in proposed else "candidate_only"
            why = "Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view."
            next_action = "Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window."
        else:
            verified_route = "self_raw_frame" if not outdoor_or_front else "self_raw_frame;auxiliary_modality"
            why = "Small counts, labels, object placement, or outdoor context are often compressed out of captions."
            next_action = "Find the event window, then extract a small number of candidate frames/crops with ffmpeg."
            if outdoor_or_front:
                auxiliary_needed = True
    elif multi:
        caption_sufficient = "maybe"
        if transcript_case:
            verified_route = "transcript_audio"
        elif static_camera_needed:
            verified_route = "static_room_source"
            why = "Multi-user room events can often be resolved by static camera before external user view."
            next_action = "Prefer static room camera frame/clip for the event window; request external user view only if static view is occluded."
            raw_frame_needed = True
            status = "weakened"
        else:
            verified_route = "self_history;static_room_source"
            why = "The event attribution is plausible from history/static evidence; external user view remains only a candidate."
            next_action = "Check self history/transcript/static room inventory for event window before external user evidence."
            raw_frame_needed = True
            status = "weakened"
    else:
        caption_sufficient = "maybe"
        verified_route = "self_caption"
        why = "No stronger evidence demand is visible from the question text."
        next_action = "Review manually; no frame extraction recommended until event window is known."

    external_verified_needed = False
    if external_candidate and not static_camera_needed and not transcript_found:
        # First-pass can only promote external-user need when no cheaper local source exists.
        # Here inventory still contains self/static/transcript candidates, so keep it false.
        external_verified_needed = False

    if external_candidate and "external_user" in proposed and not external_verified_needed:
        if status == "candidate_only":
            status = "weakened"
        why += " External_user_view is not verified as required in this pass."

    modalities = ["transcript_audio"] if transcript_case else []
    if raw_frame_needed:
        if static_camera_needed:
            modalities.extend(["static_camera_video", "ego_video", "photos", "auxiliary_video"])
        elif auxiliary_needed:
            modalities.extend(["ego_video", "photos", "auxiliary_video"])
        else:
            modalities.extend(["ego_video", "photos", "auxiliary_video", "static_camera_video"])
    if static_camera_needed and "static_camera_video" not in modalities:
        modalities.append("static_camera_video")
    modalities.extend(["gaze", "pose_imu_trajectory"])
    evidence_paths = []
    evidence_paths.extend(transcript_paths)
    evidence_paths.extend(sample_inventory_paths(idx, modalities, names, cameras, day, 12))
    evidence_paths = dedupe(evidence_paths)[:14]

    summary_bits = [
        f"Question type: {qtype}.",
        transcript_summary if transcript_case else "Transcript not required by question type.",
        f"Inventory candidates searched: {availability_summary(idx, names, cameras, day)}.",
        f"Route-focused modalities: {', '.join(sorted(set(modalities))) or 'none'}.",
    ]
    if static_camera_needed:
        summary_bits.append("Static camera should be checked before external user view.")
    elif static_available:
        summary_bits.append("Static camera exists as a possible room-level source, but may not cover this event well.")
    if auxiliary_needed:
        summary_bits.append("Outdoor/front-door/around-house wording makes auxiliary or ego evidence more likely than indoor static cameras.")

    return {
        "case_id": case["case_id"],
        "question": q,
        "bucket": bucket,
        "answer_options": case["answer_options"],
        "proposed_route": proposed,
        "verified_route": verified_route,
        "status": status,
        "caption_sufficient": caption_sufficient,
        "transcript_needed": int(transcript_case),
        "transcript_found": int(transcript_found),
        "raw_frame_needed": int(raw_frame_needed),
        "raw_frame_found": 0,
        "static_camera_needed": int(static_camera_needed),
        "external_user_candidate": int(external_candidate),
        "external_user_verified_needed": int(external_verified_needed),
        "auxiliary_needed": int(auxiliary_needed),
        "evidence_source_paths": "; ".join(evidence_paths),
        "evidence_timestamp_or_window": time_window_hint(q),
        "supporting_evidence_summary": " ".join(summary_bits),
        "cheaper_route_checked": cheaper_route_checked,
        "why_minimal_route": re.sub(r"\s+", " ", why).strip(),
        "confidence": confidence,
        "next_action": next_action,
    }


def dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def write_markdown(rows: list[dict[str, Any]]) -> None:
    status_counts = Counter(r["status"] for r in rows)
    lines = [
        "# Source-Access Seed Case Validation v0.3",
        "",
        "First-pass validation only. These are not final labels, benchmark annotations, or model-performance claims.",
        "",
        "No videos were downloaded or decoded. No frames were extracted. No VLM inference was run.",
        "",
        "## Summary",
        "",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            f"- Cases needing transcript/audio first: {sum(int(r['transcript_needed']) for r in rows)}",
            f"- Cases with local transcript keyword evidence: {sum(int(r['transcript_found']) for r in rows)}",
            f"- Cases needing targeted frame extraction later: {sum(int(r['raw_frame_needed']) for r in rows)}",
            f"- Cases where static camera is preferred before external user view: {sum(int(r['static_camera_needed']) for r in rows)}",
            f"- Cases where external user view is verified required: {sum(int(r['external_user_verified_needed']) for r in rows)}",
            "",
            "## Key Distinctions",
            "",
            "- `external_user_candidate` means another user's source might help.",
            "- `external_user_verified_needed` remains false unless cheaper self, transcript, and static-room evidence is insufficient.",
            "- Speech cases are routed to transcript/audio first; frames are not requested for them in this pass.",
            "- Static room sources are preferred over external user views for indoor room-level spatial or multi-user questions.",
            "",
            "## Cases",
            "",
        ]
    )
    for r in rows:
        lines.append(f"### {r['case_id']}")
        lines.append("")
        lines.append(f"- Question: {r['question']}")
        lines.append(f"- Bucket: {r['bucket']}")
        lines.append(f"- Proposed route: {r['proposed_route']}")
        lines.append(f"- First-pass route: {r['verified_route']}")
        lines.append(f"- Status: {r['status']}")
        lines.append(f"- Transcript needed/found: {bool(int(r['transcript_needed']))}/{bool(int(r['transcript_found']))}")
        lines.append(f"- Raw frame needed/found: {bool(int(r['raw_frame_needed']))}/{bool(int(r['raw_frame_found']))}")
        lines.append(f"- Static camera needed: {bool(int(r['static_camera_needed']))}")
        lines.append(f"- External user candidate/verified needed: {bool(int(r['external_user_candidate']))}/{bool(int(r['external_user_verified_needed']))}")
        lines.append(f"- Why: {r['why_minimal_route']}")
        lines.append(f"- Next action: {r['next_action']}")
        if r["evidence_source_paths"]:
            lines.append(f"- Evidence source candidates: `{r['evidence_source_paths']}`")
        lines.append("")
    (ROOT / "source_access_validation_v0_3.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if not SEED_CSV.exists():
        raise FileNotFoundError(SEED_CSV)
    if not CASTLE_INV.exists():
        raise FileNotFoundError(CASTLE_INV)
    seed_rows = read_csv(SEED_CSV)
    inv_rows = read_csv(CASTLE_INV)
    idx = index_inventory(inv_rows)
    transcript_files = local_transcript_files()

    validation_rows = [route_validation(case, idx, transcript_files) for case in seed_rows]
    write_csv(ROOT / "source_access_validation_v0_3.csv", validation_rows)
    write_markdown(validation_rows)
    write_csv(ROOT / "cases_needing_frame_extraction.csv", [r for r in validation_rows if int(r["raw_frame_needed"])])
    write_csv(ROOT / "cases_verified_by_transcript.csv", [r for r in validation_rows if int(r["transcript_found"])])
    write_csv(ROOT / "cases_to_reject_or_weaken.csv", [r for r in validation_rows if r["status"] in {"reject", "weakened"}])

    print("Generated first-pass validation outputs.")
    print(f"Seed cases: {len(seed_rows)}")
    print(f"Local transcript files searched: {len(transcript_files)}")
    print(f"Frame extraction recommended later: {sum(int(r['raw_frame_needed']) for r in validation_rows)}")
    print(f"Transcript verified cases: {sum(int(r['transcript_found']) for r in validation_rows)}")
    print(f"Weaken/reject cases: {sum(r['status'] in {'reject', 'weakened'} for r in validation_rows)}")


if __name__ == "__main__":
    main()
