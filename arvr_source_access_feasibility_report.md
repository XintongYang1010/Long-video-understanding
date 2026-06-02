# AR/VR Source-Access Feasibility Report

Purpose: fast meeting packet for reframing the project as AR/VR-native self-first evidence access.

Scope and constraints followed:

- Metadata first: JSON/CSV/inventory files only.
- No videos downloaded, decoded, or fully inspected.
- No VLM inference and no training.
- Heuristic taxonomy only; no model performance or benchmark completion claimed.

## Core Claim For Meeting

The EgoVis2026 question surface has enough evidence diversity to motivate a source-access controller: start from self memory/caption, then route to self audio, raw frames/crops, self history, external user views/audio, static room cameras, or auxiliary modalities using AR/VR signals such as gaze, pose/FOV, co-presence, source ownership, privacy, and communication cost.

## EgoVis2026 Metadata Taxonomy

- Questions parsed: 185
- Raw visual needed: 96 (51.9%)
- Audio/speech: 13 (7.0%)
- Spatial location: 24 (13.0%)
- Pose/FOV relevant: 25 (13.5%)
- External-source candidates: 80 (43.2%)
- Communication-sensitive: 80 (43.2%)

Primary route distribution:

- self_raw_frame: 77
- self_history: 50
- self_caption: 32
- self_highres_crop: 13
- external_user_audio: 7
- self_audio: 6

## CASTLE Source Inventory

- metadata: 16291 non-checksum files
- pose_imu_trajectory: 3829 non-checksum files
- photos: 2602 non-checksum files
- transcript_audio: 659 non-checksum files
- ego_video: 514 non-checksum files
- static_camera_video: 260 non-checksum files
- heartrate: 39 non-checksum files
- thermal: 39 non-checksum files
- auxiliary_video: 16 non-checksum files
- gaze: 7 non-checksum files

Available AR/VR-native signals include ego video, static room video, transcript/audio JSON, gaze CSV, pose/IMU/GPS metadata, thermal images, heart-rate CSV, photos, auxiliary videos, and timestamp metadata.

Without videos, the immediate analysis path is transcripts, gaze, pose/IMU/GPS, heart rate, thermal/photo metadata, inventory metadata, and question metadata. ffmpeg is only needed once the meeting work moves to targeted frame/clip extraction from ego, static, or auxiliary videos.

## Seed Cases

- Representative cases selected: 20
- Buckets: raw visual, audio/speech, spatial pose/FOV, multi-user/external source candidates.
- Each seed case includes a route, AR signal, caption sufficiency note, and next inspection action.

## Generated Files

- `egovis_arvr_query_taxonomy.csv`
- `egovis_arvr_query_taxonomy_counts.csv`
- `egovis_arvr_query_taxonomy_examples.md`
- `egovis_arvr_query_taxonomy_report.md`
- `egovis_arvr_query_taxonomy_plot.png`
- `castle_modality_inventory.csv`
- `castle_modality_counts.csv`
- `castle_modality_examples.md`
- `castle_modality_inventory_report.md`
- `source_access_seed_cases_v0_3.csv`
- `source_access_seed_cases_v0_3.md`
- `maegoqa_schema_summary.md`
- `maegoqa_vs_egovis_dataset_choice.md`
- `arvr_source_access_feasibility_report.md`
- `monday_teacher_update.md`
- `tuesday_meta_update.md`

## Recommended Meeting Position

- Present this as a feasibility and framing packet, not a completed benchmark.
- Emphasize that labels are heuristic and designed to expose evidence demand.
- Ask whether the next milestone should be manual validation of 20 seed cases or implementation of a lightweight source-access simulator.
