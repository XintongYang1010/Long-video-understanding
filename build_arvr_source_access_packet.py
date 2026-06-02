#!/usr/bin/env python3
"""Build AR/VR source-access feasibility packet from local metadata files.

This script intentionally uses only JSON/CSV/inventory metadata. It does not
download videos, decode videos, run VLMs, or train models.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EGOVIS_JSON = ROOT / "EgoVis2026_CVPR_Questions.json"
CASTLE_INVENTORY_CLEAN = ROOT / "castle_file_inventory_clean.csv"
CASTLE_INVENTORY_FALLBACK = ROOT / "castle_poc" / "castle_file_inventory.csv"
MAEGOQA_JSON = ROOT / "ma_egoqa_reproduce" / "MA-EgoQA" / "data" / "MA-EgoQA.json"
MAEGOQA_CAPTION_DIR = ROOT / "ma_egoqa_reproduce" / "MA-EgoQA" / "data" / "caption"


CATEGORIES = [
    "count_quantity",
    "visual_detail_color",
    "brand_logo_ocr_text",
    "spatial_location",
    "audio_speech",
    "temporal_history",
    "multi_user_copresence",
    "external_source_candidate",
    "raw_visual_needed",
    "gaze_relevant",
    "pose_fov_relevant",
    "communication_sensitive",
]

ROUTES = [
    "self_caption",
    "self_audio",
    "self_raw_frame",
    "self_highres_crop",
    "self_short_clip",
    "self_history",
    "external_user_view",
    "external_user_audio",
    "static_room_source",
    "auxiliary_modality",
]

KNOWN_PEOPLE = {
    "allie",
    "alice",
    "bao",
    "bjorn",
    "cathal",
    "florian",
    "jake",
    "katrina",
    "klaus",
    "lucia",
    "luca",
    "onanong",
    "ononang",
    "shure",
    "stevan",
    "tasha",
    "tien",
    "werner",
}

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "colour",
    "color",
    "colored",
    "coloured",
    "green",
    "grey",
    "gray",
    "orange",
    "purple",
    "red",
    "white",
    "yellow",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_any(text: str, needles: list[str] | set[str]) -> bool:
    for needle in needles:
        if " " in needle or "/" in needle or "-" in needle:
            if needle in text:
                return True
        elif re.search(rf"\b{re.escape(needle)}\b", text):
            return True
    return False


def answer_options_text(answers: Any) -> str:
    if isinstance(answers, dict):
        return " | ".join(f"{k}: {v}" for k, v in sorted(answers.items()))
    if isinstance(answers, list):
        return " | ".join(str(x) for x in answers)
    return str(answers)


def people_in_text(text: str) -> list[str]:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return sorted(tokens & KNOWN_PEOPLE)


def label_egovis_question(query: str, answers: Any) -> tuple[dict[str, bool], list[str], str]:
    t = norm(query)
    answer_t = norm(answer_options_text(answers))
    combined = f"{t} {answer_t}"
    labels = {name: False for name in CATEGORIES}

    people = people_in_text(query)
    visual_terms = [
        "what colour",
        "what color",
        "which colour",
        "which color",
        "wearing",
        "look like",
        "appearance",
        "shape",
        "pattern",
        "item",
        "object",
        "paintings",
        "sculptures",
        "lights",
        "cups",
        "pens",
        "cookies",
        "oranges",
        "carrots",
        "cars",
        "chessboards",
        "flower pots",
        "tree",
        "tray",
    ]
    ocr_terms = [
        "brand",
        "logo",
        "label",
        "labelled",
        "labeled",
        "written",
        "text",
        "sign",
        "qr",
        "screen",
        "clock",
        "breaker box",
        "computer",
        "notebook",
    ]
    spatial_terms = [
        "where",
        "placed",
        "located",
        "next to",
        "right of",
        "left of",
        "in front",
        "behind",
        "outside",
        "inside",
        "around",
        "over",
        "under",
        "shelf",
        "table",
        "chair",
        "kitchen island",
        "sink",
        "front door",
        "fireplace",
        "couch",
        "room",
        "around the house",
        "of the house",
        "parked",
        "hanging",
    ]
    speech_terms = [
        "said",
        "state",
        "stated",
        "according to",
        "suggest",
        "suggested",
        "ask",
        "asked",
        "asking",
        "chimed",
        "called",
        "told",
        "want",
        "wanted",
        "conversation",
        "discuss",
        "discussed",
        "quiz question",
        "answer for",
        "topic",
        "father of ai",
        "speaking",
        "singing",
        "music",
        "sound",
    ]
    temporal_terms = [
        "day",
        "last",
        "first",
        "second",
        "third",
        "after",
        "before",
        "when",
        "while",
        "during",
        "morning",
        "evening",
        "opening",
        "end",
        "start",
        "then",
        "previous",
    ]

    labels["count_quantity"] = (
        t.startswith("how many")
        or t.startswith("how much")
        or t.startswith("how fast")
        or t.startswith("at what rate")
        or t.startswith("at what temperature")
        or has_any(t, ["number of", "rate", "temperature", "km", "speed", "levels", "layers"])
    )
    labels["visual_detail_color"] = bool((set(re.findall(r"[a-z]+", combined)) & COLOR_WORDS) or has_any(t, visual_terms))
    labels["brand_logo_ocr_text"] = has_any(t, ocr_terms)
    labels["spatial_location"] = t.startswith("where") or has_any(t, spatial_terms)
    labels["audio_speech"] = has_any(t, speech_terms)
    labels["temporal_history"] = has_any(t, temporal_terms)
    labels["multi_user_copresence"] = (
        len(people) >= 2
        or has_any(t, ["team", "people", "person", "who ", "with ", "next to", "against", "group", "together"])
    )
    labels["external_source_candidate"] = (
        labels["multi_user_copresence"]
        or has_any(t, ["around the house", "outside", "parked", "front door", "behind", "static", "room"])
    )
    labels["raw_visual_needed"] = (
        labels["count_quantity"]
        or labels["visual_detail_color"]
        or labels["brand_logo_ocr_text"]
        or labels["spatial_location"]
        or has_any(t, ["which", "what item", "what object", "wearing", "parked", "hanging"])
    )
    labels["gaze_relevant"] = labels["raw_visual_needed"] or has_any(t, ["look", "see", "watch", "facing", "view"])
    labels["pose_fov_relevant"] = labels["spatial_location"] or has_any(
        t, ["next to", "behind", "in front", "outside", "around", "over", "under", "left", "right", "fov"]
    )
    labels["communication_sensitive"] = labels["external_source_candidate"] or (
        labels["audio_speech"] and labels["multi_user_copresence"]
    )

    primary_route = "self_caption"
    if labels["audio_speech"] and labels["external_source_candidate"]:
        primary_route = "external_user_audio"
    elif labels["audio_speech"]:
        primary_route = "self_audio"
    elif labels["brand_logo_ocr_text"]:
        primary_route = "self_highres_crop"
    elif labels["raw_visual_needed"]:
        primary_route = "self_raw_frame"
    elif labels["temporal_history"]:
        primary_route = "self_history"

    routes = [primary_route]
    if labels["temporal_history"]:
        routes.append("self_history")
    if labels["audio_speech"]:
        routes.append("self_audio")
    if labels["raw_visual_needed"]:
        routes.append("self_raw_frame")
    if labels["brand_logo_ocr_text"] or (labels["visual_detail_color"] and labels["count_quantity"]):
        routes.append("self_highres_crop")
    if labels["temporal_history"] and has_any(t, ["after", "before", "when", "while", "during", "end", "opening"]):
        routes.append("self_short_clip")
    if labels["external_source_candidate"]:
        routes.append("external_user_view")
    if labels["audio_speech"] and labels["external_source_candidate"]:
        routes.append("external_user_audio")
    if has_any(t, ["kitchen", "around the house", "of the house", "room", "outside", "front door", "fireplace", "couch", "island"]):
        routes.append("static_room_source")
    if labels["gaze_relevant"] or labels["pose_fov_relevant"] or has_any(t, ["rate", "speed", "temperature"]):
        routes.append("auxiliary_modality")

    deduped_routes = []
    for route in routes:
        if route in ROUTES and route not in deduped_routes:
            deduped_routes.append(route)
    route_text = ";".join(deduped_routes)
    return labels, deduped_routes, route_text


def build_egovis_taxonomy() -> list[dict[str, Any]]:
    if not EGOVIS_JSON.exists():
        raise FileNotFoundError(f"Missing {EGOVIS_JSON}")
    questions = json.loads(EGOVIS_JSON.read_text())
    rows: list[dict[str, Any]] = []
    for item in questions:
        qid = str(item.get("id", ""))
        query = str(item.get("query", ""))
        answers = item.get("answers", {})
        labels, routes, route_text = label_egovis_question(query, answers)
        row: dict[str, Any] = {
            "question_id": qid,
            "query": query,
            "answer_options": answer_options_text(answers),
            "likely_minimal_evidence_route": route_text,
            "primary_evidence_route": routes[0] if routes else "self_caption",
        }
        row.update({name: int(labels[name]) for name in CATEGORIES})
        rows.append(row)

    fieldnames = [
        "question_id",
        "query",
        "answer_options",
        *CATEGORIES,
        "primary_evidence_route",
        "likely_minimal_evidence_route",
    ]
    write_csv(ROOT / "egovis_arvr_query_taxonomy.csv", rows, fieldnames)
    write_egovis_counts(rows)
    write_egovis_examples(rows)
    write_egovis_report(rows)
    write_taxonomy_plot(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_egovis_counts(rows: list[dict[str, Any]]) -> None:
    out_rows: list[dict[str, Any]] = []
    total = len(rows)
    for cat in CATEGORIES:
        count = sum(int(r[cat]) for r in rows)
        out_rows.append({"group": "category", "label": cat, "count": count, "percent": f"{count / total * 100:.1f}"})
    route_counts = Counter()
    primary_counts = Counter(r["primary_evidence_route"] for r in rows)
    for r in rows:
        for route in str(r["likely_minimal_evidence_route"]).split(";"):
            if route:
                route_counts[route] += 1
    for route in ROUTES:
        out_rows.append(
            {
                "group": "route_any",
                "label": route,
                "count": route_counts[route],
                "percent": f"{route_counts[route] / total * 100:.1f}",
            }
        )
    for route in ROUTES:
        out_rows.append(
            {
                "group": "route_primary",
                "label": route,
                "count": primary_counts[route],
                "percent": f"{primary_counts[route] / total * 100:.1f}",
            }
        )
    write_csv(ROOT / "egovis_arvr_query_taxonomy_counts.csv", out_rows, ["group", "label", "count", "percent"])


def write_egovis_examples(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# EgoVis2026 AR/VR Query Taxonomy Examples",
        "",
        "These are heuristic multi-label assignments for source-access planning, not final ground-truth labels.",
        "",
    ]
    for cat in CATEGORIES:
        lines.append(f"## {cat}")
        examples = [r for r in rows if int(r[cat])]
        if not examples:
            lines.append("")
            lines.append("No heuristic matches.")
            lines.append("")
            continue
        for r in examples[:5]:
            lines.append(f"- {r['question_id']}: {r['query']}")
            lines.append(f"  - Route: {r['likely_minimal_evidence_route']}")
        lines.append("")
    (ROOT / "egovis_arvr_query_taxonomy_examples.md").write_text("\n".join(lines) + "\n")


def write_egovis_report(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    cat_counts = {cat: sum(int(r[cat]) for r in rows) for cat in CATEGORIES}
    primary_counts = Counter(r["primary_evidence_route"] for r in rows)
    lines = [
        "# EgoVis2026 AR/VR Source-Access Taxonomy Report",
        "",
        "Scope: heuristic taxonomy over the downloaded `EgoVis2026_CVPR_Questions.json` metadata only.",
        "",
        "This report does not claim model performance, benchmark completion, or final annotation quality. It is a feasibility packet for AR/VR-native evidence demand and source-access planning.",
        "",
        f"- Questions parsed: {total}",
        f"- Most common heuristic category: {max(cat_counts.items(), key=lambda kv: kv[1])[0]} ({max(cat_counts.values())}/{total})",
        f"- Most common primary evidence route: {primary_counts.most_common(1)[0][0]} ({primary_counts.most_common(1)[0][1]}/{total})",
        "",
        "## Category Counts",
        "",
        "| Category | Count | Percent |",
        "| --- | ---: | ---: |",
    ]
    for cat in CATEGORIES:
        count = cat_counts[cat]
        lines.append(f"| {cat} | {count} | {count / total * 100:.1f}% |")
    lines.extend(["", "## Primary Evidence Routes", "", "| Route | Count | Percent |", "| --- | ---: | ---: |"])
    for route, count in primary_counts.most_common():
        lines.append(f"| {route} | {count} | {count / total * 100:.1f}% |")
    lines.extend(
        [
            "",
            "## Interpretation For AR/VR Source Access",
            "",
            "- Raw visual demand is treated as present when a question asks for counts, small objects, text, color, or object placement that captions often compress away.",
            "- Audio/speech demand is treated as present when exact spoken content, stated facts, quiz answers, or conversational attribution is needed.",
            "- Gaze and pose/FOV are treated as routing signals: they can narrow which self frames, external views, or static room sources should be inspected.",
            "- External-source candidates are questions where a self-first memory may be insufficient because other participants, static cameras, or off-FOV evidence may carry the answer.",
            "- Communication-sensitive cases are cases where asking another source has likely cost or privacy implications, so progressive retrieval should start with metadata and low-bandwidth evidence.",
        ]
    )
    (ROOT / "egovis_arvr_query_taxonomy_report.md").write_text("\n".join(lines) + "\n")


def write_taxonomy_plot(rows: list[dict[str, Any]]) -> None:
    counts = [(cat, sum(int(r[cat]) for r in rows)) for cat in CATEGORIES]
    label_map = {
        "count_quantity": "count/quantity",
        "visual_detail_color": "visual detail/color",
        "brand_logo_ocr_text": "brand/OCR/text",
        "spatial_location": "spatial location",
        "audio_speech": "audio/speech",
        "temporal_history": "temporal/history",
        "multi_user_copresence": "multi-user co-presence",
        "external_source_candidate": "external source",
        "raw_visual_needed": "raw visual needed",
        "gaze_relevant": "gaze relevant",
        "pose_fov_relevant": "pose/FOV relevant",
        "communication_sensitive": "comm sensitive",
    }
    width, height = 1200, 720
    img = SimpleImage(width, height, (255, 255, 255))
    margin_left, margin_top, margin_right, margin_bottom = 330, 70, 60, 80
    chart_w = width - margin_left - margin_right
    bar_h = 32
    gap = 14
    max_count = max(c for _, c in counts) or 1
    img.text(300, 24, "EgoVis2026 heuristic AR/VR category counts", (20, 35, 50), scale=3)
    img.text(300, 54, "Metadata-only labels; not final annotations", (90, 90, 90), scale=2)
    for idx, (cat, count) in enumerate(counts):
        y = margin_top + idx * (bar_h + gap)
        bar_w = int(chart_w * count / max_count)
        color = palette(idx)
        img.text(18, y + 7, label_map.get(cat, cat), (20, 35, 50), scale=2)
        img.rect(margin_left, y, chart_w, bar_h, (236, 239, 242))
        img.rect(margin_left, y, bar_w, bar_h, color)
        img.text(margin_left + bar_w + 12, y + 7, str(count), (20, 35, 50), scale=2)
    img.text(margin_left, height - margin_bottom + 34, "Question count by category", (90, 90, 90), scale=2)
    img.save(ROOT / "egovis_arvr_query_taxonomy_plot.png")


def palette(i: int) -> tuple[int, int, int]:
    colors = [
        (44, 123, 182),
        (215, 95, 47),
        (48, 145, 115),
        (126, 87, 194),
        (199, 138, 38),
        (70, 130, 180),
        (190, 80, 130),
        (75, 150, 60),
        (120, 110, 70),
        (40, 160, 170),
        (170, 75, 65),
        (90, 120, 200),
    ]
    return colors[i % len(colors)]


class SimpleImage:
    """Tiny RGB PNG writer with a compact 5x7 bitmap font."""

    FONT = {
        " ": ["00000"] * 7,
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
        "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
        "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
        "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
        "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
        ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
        ";": ["00000", "00100", "00100", "00000", "00100", "00100", "01000"],
        ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
        ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
        "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    }

    def __init__(self, width: int, height: int, bg: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * width * height)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        for yy in range(y0, y1):
            row = yy * self.width * 3
            for xx in range(x0, x1):
                idx = row + xx * 3
                self.pixels[idx : idx + 3] = bytes(color)

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cx = x
        for ch in text.upper():
            glyph = self.FONT.get(ch, self.FONT.get(" ", ["00000"] * 7))
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cx + gx * scale, y + gy * scale, scale, scale, color)
            cx += 6 * scale

    def save(self, path: Path) -> None:
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            start = y * self.width * 3
            raw.extend(self.pixels[start : start + self.width * 3])
        png = bytearray()
        png.extend(b"\x89PNG\r\n\x1a\n")
        png.extend(png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        png.extend(png_chunk(b"IEND", b""))
        path.write_bytes(bytes(png))


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def infer_castle_modality(row: dict[str, str]) -> str:
    path = row["path"]
    p = path.lower()
    ext = row.get("file_extension", "").lower()
    stream = row.get("stream_type_if_parseable", "").lower()

    if "question" in p or "egovis" in p:
        return "questions"
    if "/transcript/" in p:
        return "transcript_audio"
    if "/gaze/" in p:
        return "gaze"
    if "/heartrate/" in p:
        return "heartrate"
    if "/thermal/" in p:
        return "thermal"
    if "/auxiliary/video/" in p or p.startswith("auxiliary/video/"):
        return "auxiliary_video"
    if "/auxiliary/photo/" in p or p.startswith("auxiliary/photo/"):
        if ext in {".jpg", ".jpeg", ".png", ".bmp"} and "/meta/" not in p:
            return "photos"
        return "metadata"
    if "/video/" in p and ext in {".mp4", ".mov", ".novideo"}:
        if stream == "static":
            return "static_camera_video"
        return "ego_video"
    if ext == ".gpx":
        return "pose_imu_trajectory"
    if "/metadata/" in p:
        if re.search(r"\.(accl|gyro|grav|cori|iori|gps5)\.csv(?:\.sha256)?$", p):
            return "pose_imu_trajectory"
        return "metadata"
    if ext in {".csv", ".json", ".md", ".txt", ".html", ".sha256", ".bin"}:
        return "metadata"
    return "metadata"


def castle_can_analyze_without_video(modality: str) -> bool:
    return modality in {
        "transcript_audio",
        "gaze",
        "pose_imu_trajectory",
        "thermal",
        "heartrate",
        "photos",
        "metadata",
        "questions",
    }


def castle_requires_ffmpeg(modality: str) -> bool:
    return modality in {"ego_video", "static_camera_video", "auxiliary_video"}


def castle_source_support(modality: str, row: dict[str, str]) -> str:
    stream = row.get("stream_type_if_parseable", "").lower()
    p = row["path"].lower()
    if modality == "static_camera_video" or stream == "static":
        return "static_room_source"
    if modality in {"ego_video", "transcript_audio", "gaze", "pose_imu_trajectory", "heartrate"}:
        return "self_or_external_user_source"
    if modality in {"auxiliary_video", "photos", "thermal"}:
        return "auxiliary_or_external_source"
    if "auxiliary/" in p:
        return "auxiliary_source"
    return "metadata_support"


def build_castle_inventory() -> list[dict[str, Any]]:
    inventory_path = CASTLE_INVENTORY_CLEAN if CASTLE_INVENTORY_CLEAN.exists() else CASTLE_INVENTORY_FALLBACK
    if not inventory_path.exists():
        raise FileNotFoundError("Missing CASTLE file inventory CSV")
    with inventory_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        modality = infer_castle_modality(row)
        size_text = row.get("file_size_if_available", "")
        try:
            size = int(size_text)
        except ValueError:
            size = 0
        is_checksum = row["path"].lower().endswith(".sha256")
        out_rows.append(
            {
                **row,
                "source_inventory_file": str(inventory_path.relative_to(ROOT)),
                "inferred_modality": modality,
                "is_checksum_sidecar": int(is_checksum),
                "can_analyze_without_video": int(castle_can_analyze_without_video(modality)),
                "requires_ffmpeg_for_content": int(castle_requires_ffmpeg(modality)),
                "source_access_support": castle_source_support(modality, row),
                "numeric_file_size": size,
            }
        )
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    write_csv(ROOT / "castle_modality_inventory.csv", out_rows, fieldnames)
    write_castle_counts(out_rows)
    write_castle_examples(out_rows)
    write_castle_report(out_rows, inventory_path)
    return out_rows


def write_castle_counts(rows: list[dict[str, Any]]) -> None:
    by_modality: dict[str, dict[str, Any]] = {}
    for r in rows:
        mod = r["inferred_modality"]
        d = by_modality.setdefault(
            mod,
            {
                "inferred_modality": mod,
                "total_files": 0,
                "non_checksum_files": 0,
                "total_size_if_available": 0,
                "can_analyze_without_video": int(castle_can_analyze_without_video(mod)),
                "requires_ffmpeg_for_content": int(castle_requires_ffmpeg(mod)),
                "source_access_support_values": set(),
            },
        )
        d["total_files"] += 1
        d["total_size_if_available"] += int(r["numeric_file_size"])
        if not int(r["is_checksum_sidecar"]):
            d["non_checksum_files"] += 1
        d["source_access_support_values"].add(r["source_access_support"])
    count_rows = []
    for mod in sorted(by_modality):
        d = by_modality[mod]
        count_rows.append(
            {
                "inferred_modality": mod,
                "total_files": d["total_files"],
                "non_checksum_files": d["non_checksum_files"],
                "total_size_if_available": d["total_size_if_available"],
                "can_analyze_without_video": d["can_analyze_without_video"],
                "requires_ffmpeg_for_content": d["requires_ffmpeg_for_content"],
                "source_access_support_values": ";".join(sorted(d["source_access_support_values"])),
            }
        )
    write_csv(
        ROOT / "castle_modality_counts.csv",
        count_rows,
        [
            "inferred_modality",
            "total_files",
            "non_checksum_files",
            "total_size_if_available",
            "can_analyze_without_video",
            "requires_ffmpeg_for_content",
            "source_access_support_values",
        ],
    )


def write_castle_examples(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if len(grouped[r["inferred_modality"]]) < 8:
            grouped[r["inferred_modality"]].append(r)
    lines = [
        "# CASTLE Modality Inventory Examples",
        "",
        "Examples are inferred from `castle_file_inventory.csv` path metadata. They are not decoded content.",
        "",
    ]
    for modality in sorted(grouped):
        lines.append(f"## {modality}")
        for r in grouped[modality]:
            lines.append(f"- `{r['path']}`")
        lines.append("")
    (ROOT / "castle_modality_examples.md").write_text("\n".join(lines) + "\n")


def write_castle_report(rows: list[dict[str, Any]], inventory_path: Path) -> None:
    non_checksum = [r for r in rows if not int(r["is_checksum_sidecar"])]
    counts = Counter(r["inferred_modality"] for r in non_checksum)
    signals = [
        "egocentric video streams",
        "static room camera streams",
        "transcript/audio JSON files",
        "gaze CSV files",
        "pose/IMU/GPS trajectory metadata",
        "thermal image files",
        "heart-rate CSV files",
        "auxiliary photos and videos",
        "timestamped metadata side channels",
    ]
    no_video = [m for m in sorted(counts) if castle_can_analyze_without_video(m)]
    needs_ffmpeg = [m for m in sorted(counts) if castle_requires_ffmpeg(m)]
    lines = [
        "# CASTLE Modality Inventory Report",
        "",
        f"Inventory source: `{inventory_path.relative_to(ROOT)}`",
        "",
        "This is metadata-only path inference. No videos were decoded and no full videos were downloaded.",
        "",
        f"- Total inventory rows: {len(rows)}",
        f"- Non-checksum content rows: {len(non_checksum)}",
        "",
        "## Inferred Modalities",
        "",
        "| Modality | Non-checksum files |",
        "| --- | ---: |",
    ]
    for mod, count in counts.most_common():
        lines.append(f"| {mod} | {count} |")
    lines.extend(["", "## AR/VR-Native Signals Available", ""])
    for signal in signals:
        lines.append(f"- {signal}")
    lines.extend(["", "## Can Be Analyzed Without Videos", ""])
    for mod in no_video:
        lines.append(f"- {mod}")
    lines.extend(["", "## Requires ffmpeg Or Equivalent Video Decoding", ""])
    for mod in needs_ffmpeg:
        lines.append(f"- {mod}")
    lines.extend(
        [
            "",
            "## Self/External Source-Access Support",
            "",
            "- Self-first sources: ego video, self transcript/audio, gaze, pose/IMU/GPS, heart rate, and personal photos.",
            "- External user sources: other participants' ego streams, transcripts, gaze, and pose/IMU/GPS metadata can support perspective handoff.",
            "- Static room sources: static camera video and associated metadata can support room-level events, off-FOV objects, and cross-user disambiguation.",
            "- Auxiliary sources: phone videos/photos, thermal, and other side channels can support cases where the glasses view is insufficient.",
            "- Progressive communication is feasible because many routing signals are JSON/CSV metadata before any frame or clip retrieval.",
        ]
    )
    (ROOT / "castle_modality_inventory_report.md").write_text("\n".join(lines) + "\n")


def primary_for_seed(row: dict[str, Any]) -> str:
    if int(row["audio_speech"]):
        return "audio_speech"
    if int(row["spatial_location"]) or int(row["pose_fov_relevant"]):
        return "spatial_location_pose_fov"
    if int(row["multi_user_copresence"]) or int(row["external_source_candidate"]):
        return "multi_user_external_source_candidate"
    if int(row["raw_visual_needed"]):
        return "raw_visual_needed"
    return "self_caption"


def select_seed_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    buckets = [
        ("raw_visual_needed", lambda r: int(r["raw_visual_needed"]) and not int(r["audio_speech"]), ["brand_logo_ocr_text", "visual_detail_color", "count_quantity"]),
        ("audio_speech", lambda r: int(r["audio_speech"]), ["audio_speech", "temporal_history", "multi_user_copresence"]),
        ("spatial_location_pose_fov", lambda r: (int(r["spatial_location"]) or int(r["pose_fov_relevant"])) and not int(r["audio_speech"]), ["spatial_location", "pose_fov_relevant", "gaze_relevant"]),
        (
            "multi_user_external_source_candidate",
            lambda r: (int(r["multi_user_copresence"]) or int(r["external_source_candidate"])),
            ["external_source_candidate", "multi_user_copresence", "communication_sensitive"],
        ),
    ]

    for bucket_name, predicate, score_fields in buckets:
        candidates = [r for r in rows if predicate(r) and r["question_id"] not in used]

        def score(row: dict[str, Any]) -> tuple[int, int, str]:
            field_score = sum(int(row[f]) for f in score_fields)
            route_bonus = int("external" in row["likely_minimal_evidence_route"]) + int("highres" in row["likely_minimal_evidence_route"])
            return (field_score + route_bonus, -len(row["query"]), row["question_id"])

        candidates.sort(key=score, reverse=True)
        for r in candidates[:5]:
            used.add(r["question_id"])
            selected.append(make_seed_row(r, bucket_name, len([x for x in selected if x["primary_category"] == bucket_name]) + 1))
    return selected


def make_seed_row(row: dict[str, Any], primary_category: str, idx: int) -> dict[str, Any]:
    raw = bool(int(row["raw_visual_needed"]))
    audio = bool(int(row["audio_speech"]))
    external = bool(int(row["external_source_candidate"]))
    progressive = bool(int(row["communication_sensitive"]) or external or audio)
    if audio:
        signal = "audio transcript/speech attribution"
        caption_note = "A visual caption may miss exact wording, speaker attribution, or stated facts."
        action = "Inspect transcript JSON around the candidate interval before requesting any frames."
    elif primary_category == "spatial_location_pose_fov":
        signal = "pose/FOV and gaze for viewpoint narrowing"
        caption_note = "A caption may describe the scene but miss exact placement, viewpoint, or off-FOV context."
        action = "Use pose/FOV and gaze metadata to shortlist the interval, then inspect a frame or static view if needed."
    elif external:
        signal = "co-presence, source ownership, and static/other-user view routing"
        caption_note = "A self caption may not include evidence from another participant or static room camera."
        action = "Compare self metadata with other-user/static source availability before pulling external evidence."
    else:
        signal = "gaze plus high-resolution visual crop/frame"
        caption_note = "A caption may omit small counts, colors, labels, or fine-grained object details."
        action = "Inspect a timestamped raw frame or high-resolution crop; avoid VLM inference until evidence is shortlisted."
    return {
        "case_id": f"{primary_category}_{idx:02d}_{row['question_id']}",
        "question_id": row["question_id"],
        "question": row["query"],
        "answer_options": row["answer_options"],
        "primary_category": primary_category,
        "likely_minimal_evidence_route": row["likely_minimal_evidence_route"],
        "relevant_ar_signal": signal,
        "why_caption_may_or_may_not_be_enough": caption_note,
        "raw_frame_needed": int(raw),
        "audio_needed": int(audio),
        "external_source_may_be_needed": int(external),
        "progressive_communication_matters": int(progressive),
        "suggested_next_inspection_action": action,
    }


def write_seed_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed_rows = select_seed_cases(rows)
    fields = [
        "case_id",
        "question_id",
        "question",
        "answer_options",
        "primary_category",
        "likely_minimal_evidence_route",
        "relevant_ar_signal",
        "why_caption_may_or_may_not_be_enough",
        "raw_frame_needed",
        "audio_needed",
        "external_source_may_be_needed",
        "progressive_communication_matters",
        "suggested_next_inspection_action",
    ]
    write_csv(ROOT / "source_access_seed_cases_v0_3.csv", seed_rows, fields)
    lines = [
        "# Source-Access Seed Cases v0.3",
        "",
        "These 20 cases are heuristic representatives for discussion. They are not benchmark results.",
        "",
    ]
    for row in seed_rows:
        lines.append(f"## {row['case_id']}")
        lines.append("")
        lines.append(f"- Question: {row['question']}")
        lines.append(f"- Answer options: {row['answer_options']}")
        lines.append(f"- Primary category: {row['primary_category']}")
        lines.append(f"- Likely minimal evidence route: {row['likely_minimal_evidence_route']}")
        lines.append(f"- Relevant AR signal: {row['relevant_ar_signal']}")
        lines.append(f"- Caption sufficiency: {row['why_caption_may_or_may_not_be_enough']}")
        lines.append(f"- Raw frame needed: {bool(int(row['raw_frame_needed']))}")
        lines.append(f"- Audio needed: {bool(int(row['audio_needed']))}")
        lines.append(f"- External source may be needed: {bool(int(row['external_source_may_be_needed']))}")
        lines.append(f"- Progressive communication matters: {bool(int(row['progressive_communication_matters']))}")
        lines.append(f"- Suggested next inspection action: {row['suggested_next_inspection_action']}")
        lines.append("")
    (ROOT / "source_access_seed_cases_v0_3.md").write_text("\n".join(lines) + "\n")
    return seed_rows


def summarize_maegoqa() -> None:
    if not MAEGOQA_JSON.exists():
        return
    data = json.loads(MAEGOQA_JSON.read_text())
    category_counts = Counter(str(x.get("category", "")) for x in data)
    subcategory_counts = Counter(str(x.get("subcategory", "")) for x in data)
    context_lengths = [len(x.get("contexts", {})) for x in data if isinstance(x.get("contexts", {}), dict)]
    agents = Counter()
    timestamp_keys = Counter()
    for item in data:
        contexts = item.get("contexts", {})
        if isinstance(contexts, dict):
            for ts, names in contexts.items():
                timestamp_keys[ts] += 1
                if isinstance(names, list):
                    agents.update(str(n) for n in names)
    caption_files = []
    if MAEGOQA_CAPTION_DIR.exists():
        caption_files = sorted(MAEGOQA_CAPTION_DIR.glob("*/*.json"))
    caption_group_counts = Counter(p.parent.name for p in caption_files)
    sample_caption_info = "No caption files found."
    if caption_files:
        sample_path = caption_files[0]
        sample = json.loads(sample_path.read_text())
        sample_caption_info = f"`{sample_path.relative_to(ROOT)}`: {type(sample).__name__} with {len(sample)} top-level entries"

    lines = [
        "# MA-EgoQA Schema Summary",
        "",
        "Local MA-EgoQA files were found, so this optional schema summary was generated without downloading a dataset.",
        "",
        f"- QA file: `{MAEGOQA_JSON.relative_to(ROOT)}`",
        f"- QA items: {len(data)}",
        f"- Top-level keys in sample: {', '.join(sorted(data[0].keys())) if data else 'n/a'}",
        f"- Context spans per item: min {min(context_lengths) if context_lengths else 0}, median {median(context_lengths) if context_lengths else 0}, max {max(context_lengths) if context_lengths else 0}",
        "",
        "## Question Schema",
        "",
        "- `question`: natural-language QA prompt.",
        "- `category`: broad reasoning category.",
        "- `subcategory`: more specific reasoning pattern.",
        "- `options`: answer candidates.",
        "- `answer`: correct option text.",
        "- `contexts`: timestamp-window keys mapped to visible/participating agents.",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    for cat, count in category_counts.most_common():
        lines.append(f"| {cat} | {count} |")
    lines.extend(["", "## Subcategory Counts", "", "| Subcategory | Count |", "| --- | ---: |"])
    for cat, count in subcategory_counts.most_common():
        lines.append(f"| {cat} | {count} |")
    lines.extend(["", "## Agents", ""])
    for name, count in agents.most_common():
        lines.append(f"- {name}: {count} context mentions")
    lines.extend(["", "## Captions/Transcripts", ""])
    for group, count in sorted(caption_group_counts.items()):
        lines.append(f"- {group}: {count} JSON files")
    lines.extend(["", f"Sample caption file: {sample_caption_info}", ""])
    lines.extend(
        [
            "## Evidence Labels",
            "",
            "The local QA schema exposes context windows and agents, but no explicit per-modality evidence-route labels were found in the sample schema. Captions are available at multiple temporal granularities and can seed memory/history baselines.",
        ]
    )
    (ROOT / "maegoqa_schema_summary.md").write_text("\n".join(lines) + "\n")

    choice_lines = [
        "# MA-EgoQA vs EgoVis Dataset Choice",
        "",
        "Recommended framing for the Monday/Tuesday packet: use EgoVis2026 questions as the primary taxonomy surface, and use MA-EgoQA as auxiliary context for multi-agent memory/history schema comparison.",
        "",
        "## Why EgoVis2026 Is Primary Here",
        "",
        "- The downloaded EgoVis JSON provides compact, direct multiple-choice questions that can be rapidly labeled for evidence demand.",
        "- The questions include visual detail, counting, spatial layout, spoken facts, and co-presence cases that map cleanly to AR/VR source-access routes.",
        "- The taxonomy can be produced without videos, model inference, or full dataset downloads.",
        "",
        "## Where MA-EgoQA Helps",
        "",
        "- MA-EgoQA has explicit multi-agent contexts and caption folders at 30-second, 1-hour, 10-minute, and 1-day granularities.",
        "- It is useful for self-history and multi-agent context schema, but the local QA schema does not appear to expose explicit raw-evidence labels.",
        "- It can support future comparisons between caption/history baselines and evidence-routing needs.",
        "",
        "## Meeting Position",
        "",
        "Do not claim benchmark completion. Claim a metadata-only feasibility packet showing that AR/VR-native signals can route questions to self caption, self audio, raw frames, history, external user sources, static room sources, or auxiliary modalities.",
    ]
    (ROOT / "maegoqa_vs_egovis_dataset_choice.md").write_text("\n".join(choice_lines) + "\n")


def median(values: list[int]) -> int | float:
    if not values:
        return 0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def write_final_reports(egovis_rows: list[dict[str, Any]], castle_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> None:
    total_q = len(egovis_rows)
    cat_counts = {cat: sum(int(r[cat]) for r in egovis_rows) for cat in CATEGORIES}
    primary_routes = Counter(r["primary_evidence_route"] for r in egovis_rows)
    castle_non_checksum = [r for r in castle_rows if not int(r["is_checksum_sidecar"])]
    modality_counts = Counter(r["inferred_modality"] for r in castle_non_checksum)
    outputs = [
        "egovis_arvr_query_taxonomy.csv",
        "egovis_arvr_query_taxonomy_counts.csv",
        "egovis_arvr_query_taxonomy_examples.md",
        "egovis_arvr_query_taxonomy_report.md",
        "egovis_arvr_query_taxonomy_plot.png",
        "castle_modality_inventory.csv",
        "castle_modality_counts.csv",
        "castle_modality_examples.md",
        "castle_modality_inventory_report.md",
        "source_access_seed_cases_v0_3.csv",
        "source_access_seed_cases_v0_3.md",
        "maegoqa_schema_summary.md" if MAEGOQA_JSON.exists() else "",
        "maegoqa_vs_egovis_dataset_choice.md" if MAEGOQA_JSON.exists() else "",
        "arvr_source_access_feasibility_report.md",
        "monday_teacher_update.md",
        "tuesday_meta_update.md",
    ]
    outputs = [o for o in outputs if o]
    lines = [
        "# AR/VR Source-Access Feasibility Report",
        "",
        "Purpose: fast meeting packet for reframing the project as AR/VR-native self-first evidence access.",
        "",
        "Scope and constraints followed:",
        "",
        "- Metadata first: JSON/CSV/inventory files only.",
        "- No videos downloaded, decoded, or fully inspected.",
        "- No VLM inference and no training.",
        "- Heuristic taxonomy only; no model performance or benchmark completion claimed.",
        "",
        "## Core Claim For Meeting",
        "",
        "The EgoVis2026 question surface has enough evidence diversity to motivate a source-access controller: start from self memory/caption, then route to self audio, raw frames/crops, self history, external user views/audio, static room cameras, or auxiliary modalities using AR/VR signals such as gaze, pose/FOV, co-presence, source ownership, privacy, and communication cost.",
        "",
        "## EgoVis2026 Metadata Taxonomy",
        "",
        f"- Questions parsed: {total_q}",
        f"- Raw visual needed: {cat_counts['raw_visual_needed']} ({cat_counts['raw_visual_needed'] / total_q * 100:.1f}%)",
        f"- Audio/speech: {cat_counts['audio_speech']} ({cat_counts['audio_speech'] / total_q * 100:.1f}%)",
        f"- Spatial location: {cat_counts['spatial_location']} ({cat_counts['spatial_location'] / total_q * 100:.1f}%)",
        f"- Pose/FOV relevant: {cat_counts['pose_fov_relevant']} ({cat_counts['pose_fov_relevant'] / total_q * 100:.1f}%)",
        f"- External-source candidates: {cat_counts['external_source_candidate']} ({cat_counts['external_source_candidate'] / total_q * 100:.1f}%)",
        f"- Communication-sensitive: {cat_counts['communication_sensitive']} ({cat_counts['communication_sensitive'] / total_q * 100:.1f}%)",
        "",
        "Primary route distribution:",
        "",
    ]
    for route, count in primary_routes.most_common():
        lines.append(f"- {route}: {count}")
    lines.extend(["", "## CASTLE Source Inventory", ""])
    for mod, count in modality_counts.most_common():
        lines.append(f"- {mod}: {count} non-checksum files")
    lines.extend(
        [
            "",
            "Available AR/VR-native signals include ego video, static room video, transcript/audio JSON, gaze CSV, pose/IMU/GPS metadata, thermal images, heart-rate CSV, photos, auxiliary videos, and timestamp metadata.",
            "",
            "Without videos, the immediate analysis path is transcripts, gaze, pose/IMU/GPS, heart rate, thermal/photo metadata, inventory metadata, and question metadata. ffmpeg is only needed once the meeting work moves to targeted frame/clip extraction from ego, static, or auxiliary videos.",
            "",
            "## Seed Cases",
            "",
            f"- Representative cases selected: {len(seed_rows)}",
            "- Buckets: raw visual, audio/speech, spatial pose/FOV, multi-user/external source candidates.",
            "- Each seed case includes a route, AR signal, caption sufficiency note, and next inspection action.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for out in outputs:
        lines.append(f"- `{out}`")
    lines.extend(
        [
            "",
            "## Recommended Meeting Position",
            "",
            "- Present this as a feasibility and framing packet, not a completed benchmark.",
            "- Emphasize that labels are heuristic and designed to expose evidence demand.",
            "- Ask whether the next milestone should be manual validation of 20 seed cases or implementation of a lightweight source-access simulator.",
        ]
    )
    (ROOT / "arvr_source_access_feasibility_report.md").write_text("\n".join(lines) + "\n")

    monday_lines = [
        "# Monday Teacher Update",
        "",
        "I reframed the project around AR/VR-native self-first evidence access rather than generic multi-agent QA.",
        "",
        "What I prepared:",
        "",
        f"- Parsed {total_q} EgoVis2026 questions from the small JSON metadata file.",
        "- Built a heuristic multi-label taxonomy for raw visual need, audio/speech, spatial location, temporal history, co-presence, external-source need, gaze, pose/FOV, and communication sensitivity.",
        "- Mapped questions to likely minimal evidence routes: self caption, self audio, raw frame/crop, short clip, self history, external user view/audio, static room source, and auxiliary modality.",
        f"- Parsed the local CASTLE inventory with {len(castle_non_checksum)} non-checksum content rows and inferred modalities.",
        "- Selected 20 representative source-access seed cases for discussion.",
        "",
        "Important caveat: this is a heuristic feasibility packet, not final labels, not model performance, and not benchmark completion.",
        "",
        "Ask for Monday: should I next manually validate the 20 seed cases and turn them into source-access decision examples, or first build a small controller/simulator over the metadata routes?",
    ]
    (ROOT / "monday_teacher_update.md").write_text("\n".join(monday_lines) + "\n")

    tuesday_lines = [
        "# Tuesday Meta Update",
        "",
        "The project direction is now: AR/VR-native personal memory should first check self memory, then decide whether to access self raw evidence, self history, audio, another user's view/audio, static room cameras, or auxiliary modalities.",
        "",
        "Progress made:",
        "",
        "- EgoVis2026 questions were converted into a source-access taxonomy.",
        "- CASTLE inventory shows routing signals that can be inspected before video decoding: transcripts, gaze, pose/IMU/GPS, heart rate, metadata, and source ownership from paths.",
        "- Seed cases show why captions alone are often insufficient: exact counts, small visual details, labels/OCR, exact speech, off-FOV spatial facts, and cross-user co-presence.",
        "",
        "Next technical step:",
        "",
        "Build a small evidence-router prototype that takes a question plus lightweight AR/VR metadata and outputs a progressive retrieval plan with estimated communication/privacy cost.",
        "",
        "What not to claim yet:",
        "",
        "- No model accuracy.",
        "- No benchmark completion.",
        "- No trained model.",
        "- No large-scale video processing.",
    ]
    (ROOT / "tuesday_meta_update.md").write_text("\n".join(tuesday_lines) + "\n")


def main() -> None:
    egovis_rows = build_egovis_taxonomy()
    castle_rows = build_castle_inventory()
    seed_rows = write_seed_cases(egovis_rows)
    summarize_maegoqa()
    write_final_reports(egovis_rows, castle_rows, seed_rows)
    print("Generated AR/VR source-access packet files.")
    print(f"EgoVis questions: {len(egovis_rows)}")
    print(f"CASTLE inventory rows: {len(castle_rows)}")
    print(f"Seed cases: {len(seed_rows)}")


if __name__ == "__main__":
    main()
