#!/usr/bin/env bash
# Targeted existing-6 extraction using Hugging Face remote video URLs.
# This does NOT require local CASTLE mp4 files under /scratch.
# It extracts only source-isolated frames for Q1/F009 and Q5/F012 recovery.

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xy3257/castle_poc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/existing6_targeted_frames_hf}"
LOG_CSV="${OUTPUT_ROOT}/extraction_log.csv"

REPO_BASE="${REPO_BASE:-https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024/resolve/main}"

mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${LOG_CSV}" ]]; then
  echo "target_id,source_id,video_url,offset_time,output_path,status" > "${LOG_CSV}"
fi

extract_one() {
  local target_id="$1"
  local source_id="$2"
  local video_rel="$3"
  local offset_time="$4"

  local out_dir="${OUTPUT_ROOT}/${target_id}/${source_id}"
  local safe_time="${offset_time//:/}"
  local output_path="${out_dir}/${source_id}_${safe_time}.jpg"
  local video_url="${REPO_BASE}/${video_rel}"

  mkdir -p "${out_dir}"

  if [[ -f "${output_path}" && "${FORCE}" != "1" ]]; then
    echo "${target_id},${source_id},${video_url},${offset_time},${output_path},exists_skip" >> "${LOG_CSV}"
    return 0
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    ffmpeg -hide_banner -loglevel error \
      -ss "${offset_time}" \
      -headers "Authorization: Bearer ${HF_TOKEN}" \
      -i "${video_url}" \
      -frames:v 1 -q:v 2 "${output_path}" \
      && echo "${target_id},${source_id},${video_url},${offset_time},${output_path},ok" >> "${LOG_CSV}" \
      || echo "${target_id},${source_id},${video_url},${offset_time},${output_path},ffmpeg_failed" >> "${LOG_CSV}"
  else
    ffmpeg -hide_banner -loglevel error \
      -ss "${offset_time}" \
      -i "${video_url}" \
      -frames:v 1 -q:v 2 "${output_path}" \
      && echo "${target_id},${source_id},${video_url},${offset_time},${output_path},ok" >> "${LOG_CSV}" \
      || echo "${target_id},${source_id},${video_url},${offset_time},${output_path},ffmpeg_failed" >> "${LOG_CSV}"
  fi
}

extract_source() {
  local target_id="$1"
  local source_id="$2"
  local video_rel="$3"
  shift 3
  for offset_time in "$@"; do
    extract_one "${target_id}" "${source_id}" "${video_rel}" "${offset_time}"
  done
}

# Q1 / F009 presentation recovery.
# DAY1_100500000 = 10:05:00-10:10:00.
# Hour-10 videos start at around 10:00, so clock 10:05:30 becomes offset 00:05:30.
Q1_TIMES=("00:05:30" "00:06:30" "00:07:30" "00:08:30" "00:09:30")
extract_source "EX6_HPC_Q1_PRESENTATION" "Werner"  "main/day1/Werner/video/10.mp4"  "${Q1_TIMES[@]}"
extract_source "EX6_HPC_Q1_PRESENTATION" "Meeting" "main/day1/Meeting/video/10.mp4" "${Q1_TIMES[@]}"
extract_source "EX6_HPC_Q1_PRESENTATION" "Tien"    "main/day1/Tien/video/10.mp4"    "${Q1_TIMES[@]}"
extract_source "EX6_HPC_Q1_PRESENTATION" "Onanong" "main/day1/Onanong/video/10.mp4" "${Q1_TIMES[@]}"
extract_source "EX6_HPC_Q1_PRESENTATION" "Reading" "main/day1/Reading/video/10.mp4" "${Q1_TIMES[@]}"

# Q5 / F012 tabletop recovery.
# DAY3_174500000 = 17:45:00-17:50:00.
# Hour-17 videos start at around 17:00, so clock 17:45:30 becomes offset 00:45:30.
Q5_TIMES=("00:45:30" "00:46:30" "00:47:30" "00:48:30" "00:49:30")
extract_source "EX6_HPC_Q5_TABLETOP" "Florian" "main/day3/Florian/video/17.mp4" "${Q5_TIMES[@]}"
extract_source "EX6_HPC_Q5_TABLETOP" "Cathal"  "main/day3/Cathal/video/17.mp4"  "${Q5_TIMES[@]}"
extract_source "EX6_HPC_Q5_TABLETOP" "Onanong" "main/day3/Onanong/video/17.mp4" "${Q5_TIMES[@]}"
extract_source "EX6_HPC_Q5_TABLETOP" "Allie"   "main/day3/Allie/video/17.mp4"   "${Q5_TIMES[@]}"
extract_source "EX6_HPC_Q5_TABLETOP" "Meeting" "main/day3/Meeting/video/17.mp4" "${Q5_TIMES[@]}"

echo "Done. Output root: ${OUTPUT_ROOT}"
echo "JPG count:"
find "${OUTPUT_ROOT}" -type f -name '*.jpg' | wc -l
echo "Log: ${LOG_CSV}"
