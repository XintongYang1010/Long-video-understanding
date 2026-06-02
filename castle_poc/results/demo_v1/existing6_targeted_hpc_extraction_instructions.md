# Existing-6 Targeted HPC Extraction Instructions

This is a plan only. Do not treat these targets as new benchmark labels until the extracted source-isolated frames are visually reviewed.

Run only on the HPC environment where the CASTLE videos are available. This task did not run HPC extraction, ssh, ffmpeg, cropping, model inference, or API calls.

Recommended immediate targets:
- `EX6_HPC_Q1_PRESENTATION`: `Werner;Meeting;Tien;Onanong;Reading` at `10:05:30;10:06:30;10:07:30;10:08:30;10:09:30`.
- `EX6_HPC_Q5_TABLETOP`: `Florian;Cathal;Onanong;Allie;Meeting` at `17:45:30;17:46:30;17:47:30;17:48:30;17:49:30`.

Template script:
- `scripts/hpc_extract_existing6_targeted_frames.sh`

Before running the script on HPC, confirm:
- `PROJECT_ROOT` or `VIDEO_ROOT` points to the local CASTLE dataset root.
- Video paths like `main/day1/Werner/video/10.mp4` and `main/day3/Florian/video/17.mp4` exist.
- `OUTPUT_ROOT` points to a new source-isolated frame directory.

The script extracts only 5 frames per listed source, not dense video frames.
