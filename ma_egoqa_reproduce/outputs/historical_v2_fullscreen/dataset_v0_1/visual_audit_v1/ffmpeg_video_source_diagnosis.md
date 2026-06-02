# FFmpeg And EgoLife Video Source Diagnosis

## Environment Commands

- `hostname`: `torch-login-b-2.hpc-infra.svc.cluster.local`
- `pwd`: `/scratch/xy3257/ma_egoqa_reproduce`
- `which python`: `/usr/bin/python`
- `python --version`: `Python 3.12.12`
- `which ffmpeg || true`: no `ffmpeg` in current PATH
- `ffmpeg -version || true`: `ffmpeg: command not found`
- `which ffprobe || true`: no `ffprobe` in current PATH
- `module avail ffmpeg 2>&1 | head -80 || true`: `No module(s) or extension(s) found!`
- `module spider ffmpeg 2>&1 | head -80 || true`: `Unable to find: "ffmpeg"`
- `conda info --envs || true`: `conda: command not found` in the current shell PATH
- `conda list | grep -i ffmpeg || true`: `conda: command not found` in the current shell PATH

Using the known Miniforge executable directly:

- `/scratch/xy3257/miniforge3/bin/conda info --envs` lists:
  - `/scratch/xy3257/castle_hpc/envs/castle`
  - `/scratch/xy3257/castle_poc/cenv`
  - `/scratch/xy3257/ma_egoqa_reproduce/envs/maegoqa`
  - `/scratch/xy3257/miniforge3`
- `/scratch/xy3257/miniforge3/bin/conda list -p /scratch/xy3257/ma_egoqa_reproduce/envs/maegoqa | grep -i ffmpeg`: no ffmpeg package
- `/scratch/xy3257/miniforge3/bin/conda list -p /scratch/xy3257/castle_hpc/envs/castle | grep -i ffmpeg`: `ffmpeg 8.0.1 ... conda-forge`
- `/scratch/xy3257/miniforge3/bin/conda list -p /scratch/xy3257/castle_poc/cenv | grep -i ffmpeg`: `ffmpeg 6.1.2 ... conda-forge`

## Working FFmpeg Found

- Direct binary: `/scratch/xy3257/castle_hpc/envs/castle/bin/ffmpeg`
- Direct ffprobe: `/scratch/xy3257/castle_hpc/envs/castle/bin/ffprobe`
- `ffmpeg -version`: `ffmpeg version 8.0.1 Copyright (c) 2000-2025 the FFmpeg developers`
- `ffprobe -version`: `ffprobe version 8.0.1 Copyright (c) 2007-2025 the FFmpeg developers`

Minimal activation/PATH option:

```bash
export PATH=/scratch/xy3257/castle_hpc/envs/castle/bin:$PATH
```

Equivalent conda activation if Miniforge is initialized:

```bash
source /scratch/xy3257/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/xy3257/castle_hpc/envs/castle
```

For the MA-EgoQA scripts, I used the explicit path:

```bash
/scratch/xy3257/castle_hpc/envs/castle/bin/ffmpeg
```

## CASTLE FFmpeg Usage Records

Searches found prior CASTLE ffmpeg use in:

- `/scratch/xy3257/castle_hpc/castle_event_relevant_ffmpeg_logs/.../*.stderr.txt`
- `/scratch/xy3257/castle_hpc/castle_low_bandwidth_ffmpeg_logs/.../*.stderr.txt`
- `/scratch/xy3257/castle_hpc/castle_ffmpeg_remote_seek_probe.txt`
- `/scratch/xy3257/castle_hpc/castle_event_relevant_view_selection.py`
- `/scratch/xy3257/castle_hpc/castle_low_bandwidth_remote_frame_test.py`
- `/scratch/xy3257/castle_hpc/castle_remote_access_diagnosis.py`

Representative CASTLE command:

```bash
/scratch/xy3257/castle_hpc/envs/castle/bin/ffmpeg -hide_banner -v error -nostdin -y -rw_timeout 30000000 -reconnect 1 -reconnect_on_network_error 1 -reconnect_streamed 1 -reconnect_delay_max 5 -multiple_requests 1 -seekable 1 -ss 360.000 -i https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024/resolve/main/main/day1/Onanong/video/10.mp4 -frames:v 1 -q:v 2 castle_remote_test_frame.jpg
```

## Required Answers

1. Current shell can directly call ffmpeg: **No**. `which ffmpeg` fails and `ffmpeg -version` is command-not-found.
2. Module ffmpeg exists: **No**. Both `module avail ffmpeg` and `module spider ffmpeg` failed to find it.
3. Some conda env has ffmpeg: **Yes**. `/scratch/xy3257/castle_hpc/envs/castle` has `ffmpeg 8.0.1`; `/scratch/xy3257/castle_poc/cenv` has `ffmpeg 6.1.2`.
4. Previous CASTLE ffmpeg command location: CASTLE logs under `/scratch/xy3257/castle_hpc/castle_*ffmpeg_logs/` and probe files such as `/scratch/xy3257/castle_hpc/castle_ffmpeg_remote_seek_probe.txt`.
5. Why MA-EgoQA visual audit did not inherit previous ffmpeg: the MA-EgoQA shell was using `/usr/bin/python` and did not have `/scratch/xy3257/castle_hpc/envs/castle/bin` on PATH; `conda` was also not initialized in PATH, and the local MA-EgoQA env `/scratch/xy3257/ma_egoqa_reproduce/envs/maegoqa` does not contain ffmpeg.

## Video Source Diagnosis

The previous caption-only visual audit had `raw_key` values such as `DAY7_A6_SHURE_11413000.mp4` but no local video root. Mapping these keys to Hugging Face paths fixes the source issue:

```text
source_agent Shure -> A6_SHURE
day 7 -> DAY7
raw_key DAY7_A6_SHURE_11413000.mp4
hf_path A6_SHURE/DAY7/DAY7_A6_SHURE_11413000.mp4
```

The generated `visual_video_source_plan_v1.csv` resolves all 240 planned frame rows to EgoLife Hugging Face paths, covering 80 unique mp4 clips.
