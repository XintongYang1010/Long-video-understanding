# CASTLE Modality Inventory Report

Inventory source: `castle_poc/castle_file_inventory.csv`

This is metadata-only path inference. No videos were decoded and no full videos were downloaded.

- Total inventory rows: 49155
- Non-checksum content rows: 24256

## Inferred Modalities

| Modality | Non-checksum files |
| --- | ---: |
| metadata | 16291 |
| pose_imu_trajectory | 3829 |
| photos | 2602 |
| transcript_audio | 659 |
| ego_video | 514 |
| static_camera_video | 260 |
| heartrate | 39 |
| thermal | 39 |
| auxiliary_video | 16 |
| gaze | 7 |

## AR/VR-Native Signals Available

- egocentric video streams
- static room camera streams
- transcript/audio JSON files
- gaze CSV files
- pose/IMU/GPS trajectory metadata
- thermal image files
- heart-rate CSV files
- auxiliary photos and videos
- timestamped metadata side channels

## Can Be Analyzed Without Videos

- gaze
- heartrate
- metadata
- photos
- pose_imu_trajectory
- thermal
- transcript_audio

## Requires ffmpeg Or Equivalent Video Decoding

- auxiliary_video
- ego_video
- static_camera_video

## Self/External Source-Access Support

- Self-first sources: ego video, self transcript/audio, gaze, pose/IMU/GPS, heart rate, and personal photos.
- External user sources: other participants' ego streams, transcripts, gaze, and pose/IMU/GPS metadata can support perspective handoff.
- Static room sources: static camera video and associated metadata can support room-level events, off-FOV objects, and cross-user disambiguation.
- Auxiliary sources: phone videos/photos, thermal, and other side channels can support cases where the glasses view is insufficient.
- Progressive communication is feasible because many routing signals are JSON/CSV metadata before any frame or clip retrieval.
