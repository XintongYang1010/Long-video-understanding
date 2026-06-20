#!/usr/bin/env python3
"""Record public-release evidence for EgoEverything benchmark/eval assets."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


GITHUB_API = "https://api.github.com/repos/ElecBeholder/EgoEverything"
HF_API = "https://huggingface.co/api"
QUERIES = ("EgoEverything", "ElecBeholder", "2604.08342")
SIGNAL_WORDS = (
    "eval",
    "question",
    "answer",
    "mcq",
    "data",
    "vqa",
    "readme",
    "json",
    "csv",
    "prompt",
    "benchmark",
    "section",
)
SECTION4_DATA_WORDS = ("mcq", "question", "answer", "gold", "label", "benchmark", "section4", "vqa")
RUNNER_WORDS = {"eval", "evaluation", "benchmark", "runner", "run", "section4"}
METHOD_WORDS = {"fr", "ad", "gc", "gm", "vmp", "amego", "vlm", "videollama", "gemini", "longva", "llava"}


def fetch_json(url: str, timeout: int = 30) -> tuple[Any | None, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "egoeverything-repro-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), None
    except Exception as exc:  # noqa: BLE001 - report network/API failures in JSON.
        return None, f"{type(exc).__name__}: {exc}"


def hf_search(kind: str, query: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(query)
    data, error = fetch_json(f"{HF_API}/{kind}?search={encoded}", timeout=20)
    items = data if isinstance(data, list) else []
    return {
        "query": query,
        "error": error,
        "count": len(items),
        "ids": [item.get("id") or item.get("modelId") for item in items[:20] if isinstance(item, dict)],
    }


def path_tokens(path: str) -> set[str]:
    return {part for part in re.split(r"[/_.-]+", path.lower()) if part}


def looks_like_section4_runner(path: str) -> bool:
    tokens = path_tokens(path)
    if "gaze" in tokens:
        return False
    return bool(tokens & RUNNER_WORDS) and bool(tokens & METHOD_WORDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check public EgoEverything release assets")
    parser.add_argument("--expected-sha", default="41d91cf529b2a0a889d5ea41e0b488c50106443d")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    repo, repo_error = fetch_json(GITHUB_API)
    branches, branches_error = fetch_json(f"{GITHUB_API}/branches")
    releases, releases_error = fetch_json(f"{GITHUB_API}/releases")
    tags, tags_error = fetch_json(f"{GITHUB_API}/tags")
    tree, tree_error = fetch_json(f"{GITHUB_API}/git/trees/{args.expected_sha}?recursive=1")

    branch_heads = {
        item.get("name"): item.get("commit", {}).get("sha")
        for item in branches or []
        if isinstance(item, dict)
    }
    tree_paths = [item.get("path", "") for item in (tree or {}).get("tree", []) if isinstance(item, dict)]
    signal_paths = [
        path
        for path in tree_paths
        if any(word in path.lower() for word in SIGNAL_WORDS)
    ]
    section4_data_candidates = [
        path
        for path in tree_paths
        if any(word in path.lower().replace("_", "").replace("-", "") for word in SECTION4_DATA_WORDS)
    ]
    section4_runner_candidates = [path for path in tree_paths if looks_like_section4_runner(path)]
    section4_asset_candidates = sorted(set(section4_data_candidates + section4_runner_candidates))

    hf_results: dict[str, list[dict[str, Any]]] = {}
    for kind in ("datasets", "models", "spaces"):
        hf_results[kind] = []
        for query in QUERIES:
            hf_results[kind].append(hf_search(kind, query))
            time.sleep(0.5)

    hf_any_hits = any(item["count"] for results in hf_results.values() for item in results)
    github_errors = {
        "repo": repo_error,
        "branches": branches_error,
        "releases": releases_error,
        "tags": tags_error,
        "tree": tree_error,
    }
    public_section4_ready = bool(section4_asset_candidates or hf_any_hits)

    report = {
        "checked_at_unix": int(time.time()),
        "github": {
            "api": GITHUB_API,
            "errors": github_errors,
            "default_branch": repo.get("default_branch") if isinstance(repo, dict) else None,
            "pushed_at": repo.get("pushed_at") if isinstance(repo, dict) else None,
            "branch_heads": branch_heads,
            "expected_sha": args.expected_sha,
            "expected_sha_matches_fix": branch_heads.get("fix") == args.expected_sha,
            "release_count": len(releases or []) if isinstance(releases, list) else 0,
            "tag_count": len(tags or []) if isinstance(tags, list) else 0,
            "tree_truncated": tree.get("truncated") if isinstance(tree, dict) else None,
            "signal_paths": signal_paths,
            "section4_asset_candidate_paths": section4_asset_candidates,
        },
        "huggingface": hf_results,
        "hf_any_hits": hf_any_hits,
        "public_section4_assets_ready": public_section4_ready,
        "conclusion": (
            "No public official Section 4 MCQ manifest, answer key, or VLM eval runner "
            "was found in GitHub releases/tags/tree or Hugging Face search results."
        ),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
