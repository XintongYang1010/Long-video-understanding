#!/usr/bin/env python3
"""
Staged network diagnosis for CASTLE remote frame extraction.

This script probes one selected HuggingFace video URL without using
hf_hub_download and without saving video content. It writes small probe logs,
deletes the 1KB curl range file after measuring it, and caps local probe
artifacts at 100 MB by default.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_LOCAL_MB = 100.0
HTTP_TIMEOUT_SEC = 30
COMMAND_TIMEOUT_SEC = 75


@dataclass(frozen=True)
class ProbeTarget:
    video_path: str
    remote_url: str
    target_clock_time: str
    offset_seconds: str


def read_first_target(path: Path) -> ProbeTarget:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
    return ProbeTarget(
        video_path=row["video_path"],
        remote_url=row["remote_url"],
        target_clock_time=row["target_clock_time"],
        offset_seconds=row["offset_seconds"],
    )


def auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def probe_artifact_size() -> int:
    paths = [
        Path("castle_remote_http_probe.txt"),
        Path("castle_curl_probe.txt"),
        Path("castle_ffprobe_probe.txt"),
        Path("castle_ffmpeg_remote_seek_probe.txt"),
        Path("castle_ffmpeg_post_input_seek_probe.txt"),
        Path("castle_remote_test_frame.jpg"),
        Path("castle_range_test.bin"),
    ]
    total = 0
    for path in paths:
        if path.exists() and path.is_file():
            total += path.stat().st_size
    return total


def enforce_storage_cap() -> None:
    max_bytes = int(MAX_LOCAL_MB * 1024 * 1024)
    size = probe_artifact_size()
    if size > max_bytes:
        raise RuntimeError(f"local probe artifacts exceed {MAX_LOCAL_MB:.1f}MB: {size} bytes")


def header_items(headers: Any) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def python_http_probe(target: ProbeTarget) -> dict[str, Any]:
    lines = [
        "CASTLE Python HTTP probe",
        "",
        f"video_path: {target.video_path}",
        f"remote_url: {target.remote_url}",
        f"target_clock_time: {target.target_clock_time}",
        f"offset_seconds: {target.offset_seconds}",
        "",
    ]
    result: dict[str, Any] = {
        "head_ok": False,
        "range_ok": False,
        "head_status": "",
        "head_final_url": "",
        "head_content_length": "",
        "head_accept_ranges": "",
        "range_status": "",
        "range_bytes": 0,
        "error": "",
    }

    try:
        head_request = urllib.request.Request(target.remote_url, method="HEAD", headers=auth_headers())
        with urllib.request.urlopen(head_request, timeout=HTTP_TIMEOUT_SEC) as response:
            headers = header_items(response.headers)
            result.update(
                {
                    "head_ok": True,
                    "head_status": str(response.status),
                    "head_final_url": response.geturl(),
                    "head_content_length": headers.get("content-length", ""),
                    "head_accept_ranges": headers.get("accept-ranges", ""),
                }
            )
            lines.extend(
                [
                    "HEAD:",
                    f"  status: {response.status}",
                    f"  final_url: {response.geturl()}",
                    f"  content-length: {headers.get('content-length', '')}",
                    f"  accept-ranges: {headers.get('accept-ranges', '')}",
                    "",
                ]
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] += f"HEAD failed: {exc}\n"
        lines.extend(["HEAD:", f"  error: {exc}", ""])

    try:
        range_headers = {"Range": "bytes=0-1023", **auth_headers()}
        range_request = urllib.request.Request(target.remote_url, headers=range_headers)
        with urllib.request.urlopen(range_request, timeout=HTTP_TIMEOUT_SEC) as response:
            data = response.read(1024)
            result.update({"range_ok": True, "range_status": str(response.status), "range_bytes": len(data)})
            lines.extend(
                [
                    "RANGE GET bytes=0-1023:",
                    f"  status: {response.status}",
                    f"  final_url: {response.geturl()}",
                    f"  bytes_received: {len(data)}",
                    "",
                ]
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] += f"Range GET failed: {exc}\n"
        lines.extend(["RANGE GET bytes=0-1023:", f"  error: {exc}", ""])

    Path("castle_remote_http_probe.txt").write_text("\n".join(lines), encoding="utf-8")
    enforce_storage_cap()
    return result


def redact_command(command: list[str]) -> str:
    output = []
    skip_next = False
    for part in command:
        if skip_next:
            output.append("Authorization: Bearer ***")
            skip_next = False
            continue
        output.append(part)
        if part in {"-H", "-headers"}:
            skip_next = True
    return " ".join(output)


def run_command(command: list[str], timeout_sec: int) -> dict[str, Any]:
    start = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "timeout": False,
            "elapsed_sec": time.time() - start,
            "command": redact_command(command),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "returncode": "",
            "stdout": stdout,
            "stderr": stderr,
            "timeout": True,
            "elapsed_sec": time.time() - start,
            "command": redact_command(command),
        }


def write_command_log(path: Path, title: str, entries: list[dict[str, Any]]) -> None:
    lines = [title, ""]
    for entry in entries:
        lines.extend(
            [
                f"COMMAND: {entry['command']}",
                f"RETURN_CODE: {entry['returncode']}",
                f"TIMEOUT: {entry['timeout']}",
                f"ELAPSED_SEC: {entry['elapsed_sec']:.3f}",
                "",
                "STDOUT:",
                (entry["stdout"][:10000] if entry["stdout"] else ""),
                "",
                "STDERR:",
                (entry["stderr"][:10000] if entry["stderr"] else ""),
                "",
                "-" * 80,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    enforce_storage_cap()


def curl_probe(target: ProbeTarget) -> dict[str, Any]:
    curl = shutil.which("curl")
    if not curl:
        result = {"available": False, "head_ok": False, "range_ok": False, "range_size": 0, "entries": []}
        Path("castle_curl_probe.txt").write_text("curl not found\n", encoding="utf-8")
        return result

    auth = auth_headers()
    auth_args = ["-H", f"Authorization: {auth['Authorization']}"] if auth else []
    head_command = [curl, "-I", "-L", "--max-time", "45", "--connect-timeout", "15", *auth_args, target.remote_url]
    range_file = Path("castle_range_test.bin")
    if range_file.exists():
        range_file.unlink()
    range_command = [
        curl,
        "-L",
        "--max-time",
        "45",
        "--connect-timeout",
        "15",
        "-r",
        "0-1023",
        *auth_args,
        target.remote_url,
        "-o",
        str(range_file),
    ]
    head_entry = run_command(head_command, 50)
    range_entry = run_command(range_command, 50)
    range_size = range_file.stat().st_size if range_file.exists() else 0
    range_deleted = False
    if range_file.exists():
        range_file.unlink()
        range_deleted = True
    range_entry["stdout"] += f"\nrange_file_size_before_delete={range_size}\nrange_file_deleted={range_deleted}\n"
    write_command_log(Path("castle_curl_probe.txt"), "CASTLE curl probe", [head_entry, range_entry])
    return {
        "available": True,
        "head_ok": head_entry["returncode"] == 0,
        "range_ok": range_entry["returncode"] == 0 and range_size <= 1024 and range_size > 0,
        "range_size": range_size,
        "entries": [head_entry, range_entry],
    }


def ffprobe_probe(target: ProbeTarget) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        result = {"available": False, "ok": False, "entry": None}
        Path("castle_ffprobe_probe.txt").write_text("ffprobe not found\n", encoding="utf-8")
        return result
    command = [
        ffprobe,
        "-v",
        "error",
        "-rw_timeout",
        "30000000",
        "-reconnect",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
    ]
    auth = auth_headers()
    if auth:
        command.extend(["-headers", f"Authorization: {auth['Authorization']}"])
    command.extend(["-show_entries", "format=duration,size:stream=index,codec_type,codec_name", "-of", "json", target.remote_url])
    entry = run_command(command, COMMAND_TIMEOUT_SEC)
    write_command_log(Path("castle_ffprobe_probe.txt"), "CASTLE ffprobe probe", [entry])
    return {"available": True, "ok": entry["returncode"] == 0, "entry": entry}


def ffmpeg_frame_probe(target: ProbeTarget) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        Path("castle_ffmpeg_remote_seek_probe.txt").write_text("ffmpeg not found\n", encoding="utf-8")
        return {"available": False, "first_ok": False, "second_ok": False, "entries": []}

    output_path = Path("castle_remote_test_frame.jpg")
    if output_path.exists():
        output_path.unlink()
    common_before_input = [
        ffmpeg,
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-rw_timeout",
        "30000000",
        "-reconnect",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-multiple_requests",
        "1",
        "-seekable",
        "1",
    ]
    auth = auth_headers()
    if auth:
        common_before_input.extend(["-headers", f"Authorization: {auth['Authorization']}"])

    first_command = [
        *common_before_input,
        "-ss",
        target.offset_seconds,
        "-i",
        target.remote_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    first = run_command(first_command, COMMAND_TIMEOUT_SEC)
    first_size = output_path.stat().st_size if output_path.exists() else 0
    first["stdout"] += f"\noutput_frame_size={first_size}\n"
    write_command_log(Path("castle_ffmpeg_remote_seek_probe.txt"), "CASTLE ffmpeg remote seek probe", [first])
    first_ok = first["returncode"] == 0 and first_size > 0
    if first_ok:
        return {"available": True, "first_ok": True, "second_ok": False, "entries": [first], "output_size": first_size}

    if output_path.exists():
        output_path.unlink()
    # This can require more remote reads than input-side seek, so it remains
    # bounded by a short timeout and only writes one JPEG locally.
    second_command = [
        *common_before_input,
        "-i",
        target.remote_url,
        "-ss",
        target.offset_seconds,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    second = run_command(second_command, COMMAND_TIMEOUT_SEC)
    second_size = output_path.stat().st_size if output_path.exists() else 0
    second["stdout"] += f"\noutput_frame_size={second_size}\n"
    write_command_log(Path("castle_ffmpeg_post_input_seek_probe.txt"), "CASTLE ffmpeg post-input seek probe", [second])
    second_ok = second["returncode"] == 0 and second_size > 0
    return {
        "available": True,
        "first_ok": False,
        "second_ok": second_ok,
        "entries": [first, second],
        "output_size": second_size,
    }


def failure_text(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    return f"{entry.get('stderr', '')}\n{entry.get('stdout', '')}".lower()


def write_final_diagnosis(
    target: ProbeTarget,
    python_result: dict[str, Any],
    curl_result: dict[str, Any],
    ffprobe_result: dict[str, Any],
    ffmpeg_result: dict[str, Any],
) -> None:
    python_ok = bool(python_result.get("head_ok") and python_result.get("range_ok"))
    curl_ok = bool(curl_result.get("head_ok") and curl_result.get("range_ok"))
    ffprobe_ok = bool(ffprobe_result.get("ok"))
    ffmpeg_ok = bool(ffmpeg_result.get("first_ok") or ffmpeg_result.get("second_ok"))
    ffmpeg_entries = ffmpeg_result.get("entries", [])
    ffmpeg_text = "\n".join(failure_text(entry) for entry in ffmpeg_entries)
    ffprobe_text = failure_text(ffprobe_result.get("entry"))

    if python_ok and curl_ok and not ffprobe_ok:
        likely = "Python/curl can reach the URL and read byte ranges, but ffprobe cannot open it; likely ffmpeg HTTPS/redirect/range handling or ffmpeg network options."
    elif (python_ok or curl_ok) and ffprobe_ok and not ffmpeg_ok:
        likely = "Remote URL opens in ffprobe but frame extraction fails; likely seeking/keyframe/index behavior or ffmpeg option placement."
    elif not python_ok and not curl_ok and not ffprobe_ok:
        likely = "All HTTP clients failed; likely local network/firewall/proxy or HuggingFace access issue."
    elif "timed out" in ffmpeg_text or "operation timed out" in ffmpeg_text or "timed out" in ffprobe_text:
        likely = "ffmpeg-family tools time out against HuggingFace from this network."
    else:
        likely = "Mixed results; inspect probe logs for details."

    recommendation = "retry on server"
    if python_ok and curl_ok and not ffmpeg_ok:
        recommendation = "retry on a server with stable HuggingFace egress and/or try ffmpeg with authenticated headers/proxy settings"
    if not python_ok and not curl_ok:
        recommendation = "retry on server or another network before considering video downloads"
    if ffmpeg_ok:
        recommendation = "scale cautiously to 3 windows and 12 streams while keeping storage limits"

    lines = [
        "CASTLE remote access diagnosis",
        "",
        "Probe target:",
        f"  video_path: {target.video_path}",
        f"  remote_url: {target.remote_url}",
        f"  target_clock_time: {target.target_clock_time}",
        f"  offset_seconds: {target.offset_seconds}",
        "",
        "1. Does Python HEAD/Range access work?",
        f"   {'Yes' if python_ok else 'No'}",
        f"   HEAD status: {python_result.get('head_status', '')}",
        f"   final URL: {python_result.get('head_final_url', '')}",
        f"   content-length: {python_result.get('head_content_length', '')}",
        f"   accept-ranges: {python_result.get('head_accept_ranges', '')}",
        f"   Range status: {python_result.get('range_status', '')}",
        f"   Range bytes: {python_result.get('range_bytes', 0)}",
        "",
        "2. Does curl Range access work?",
        f"   {'Yes' if curl_ok else 'No'}",
        f"   curl available: {curl_result.get('available')}",
        f"   range bytes saved then deleted: {curl_result.get('range_size', 0)}",
        "",
        "3. Does ffprobe open the remote URL?",
        f"   {'Yes' if ffprobe_ok else 'No'}",
        "",
        "4. Does ffmpeg extract one frame?",
        f"   {'Yes' if ffmpeg_ok else 'No'}",
        f"   first command input-side -ss ok: {ffmpeg_result.get('first_ok')}",
        f"   second command post-input -ss ok: {ffmpeg_result.get('second_ok')}",
        "",
        "5. If Python/curl work but ffmpeg fails, what is the likely ffmpeg issue?",
        f"   {likely if (python_ok or curl_ok) and not ffmpeg_ok else 'Not applicable.'}",
        "",
        "6. If all fail, is this likely local network/firewall/proxy?",
        f"   {'Yes' if not python_ok and not curl_ok and not ffprobe_ok else 'No, at least one non-ffmpeg HTTP probe worked.'}",
        "",
        "7. Recommendation:",
        f"   {recommendation}",
        "",
        "Options:",
        "   a) retry on server: recommended if ffmpeg-family tools keep timing out locally",
        "   b) use transcript-only validation: acceptable as a non-visual fallback, but weaker",
        "   c) download 1-2 full videos as fallback: only with explicit approval because files are multi-GB",
        "   d) abandon CASTLE visual validation: not recommended until tested from a better network/server",
        "",
        f"Local probe artifact size: {probe_artifact_size()} bytes",
    ]
    Path("castle_remote_access_diagnosis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    target = read_first_target(Path("castle_remote_url_test_table.csv"))
    print(f"video_path: {target.video_path}")
    print(f"remote_url: {target.remote_url}")
    print(f"target_clock_time: {target.target_clock_time}")
    print(f"offset_seconds: {target.offset_seconds}")

    python_result = python_http_probe(target)
    curl_result = curl_probe(target)
    ffprobe_result = ffprobe_probe(target)
    ffmpeg_result = ffmpeg_frame_probe(target)
    write_final_diagnosis(target, python_result, curl_result, ffprobe_result, ffmpeg_result)

    print("Wrote castle_remote_http_probe.txt")
    print("Wrote castle_curl_probe.txt")
    print("Wrote castle_ffprobe_probe.txt")
    print("Wrote castle_ffmpeg_remote_seek_probe.txt")
    if Path("castle_ffmpeg_post_input_seek_probe.txt").exists():
        print("Wrote castle_ffmpeg_post_input_seek_probe.txt")
    print("Wrote castle_remote_access_diagnosis.txt")
    print(f"Local probe artifact size: {probe_artifact_size()} bytes")


if __name__ == "__main__":
    main()
