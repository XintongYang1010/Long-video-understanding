# EgoVis2026 AR/VR Source-Access Taxonomy Report

Scope: heuristic taxonomy over the downloaded `EgoVis2026_CVPR_Questions.json` metadata only.

This report does not claim model performance, benchmark completion, or final annotation quality. It is a feasibility packet for AR/VR-native evidence demand and source-access planning.

- Questions parsed: 185
- Most common heuristic category: gaze_relevant (97/185)
- Most common primary evidence route: self_raw_frame (77/185)

## Category Counts

| Category | Count | Percent |
| --- | ---: | ---: |
| count_quantity | 37 | 20.0% |
| visual_detail_color | 46 | 24.9% |
| brand_logo_ocr_text | 13 | 7.0% |
| spatial_location | 24 | 13.0% |
| audio_speech | 13 | 7.0% |
| temporal_history | 89 | 48.1% |
| multi_user_copresence | 76 | 41.1% |
| external_source_candidate | 80 | 43.2% |
| raw_visual_needed | 96 | 51.9% |
| gaze_relevant | 97 | 52.4% |
| pose_fov_relevant | 25 | 13.5% |
| communication_sensitive | 80 | 43.2% |

## Primary Evidence Routes

| Route | Count | Percent |
| --- | ---: | ---: |
| self_raw_frame | 77 | 41.6% |
| self_history | 50 | 27.0% |
| self_caption | 32 | 17.3% |
| self_highres_crop | 13 | 7.0% |
| external_user_audio | 7 | 3.8% |
| self_audio | 6 | 3.2% |

## Interpretation For AR/VR Source Access

- Raw visual demand is treated as present when a question asks for counts, small objects, text, color, or object placement that captions often compress away.
- Audio/speech demand is treated as present when exact spoken content, stated facts, quiz answers, or conversational attribution is needed.
- Gaze and pose/FOV are treated as routing signals: they can narrow which self frames, external views, or static room sources should be inspected.
- External-source candidates are questions where a self-first memory may be insufficient because other participants, static cameras, or off-FOV evidence may carry the answer.
- Communication-sensitive cases are cases where asking another source has likely cost or privacy implications, so progressive retrieval should start with metadata and low-bandwidth evidence.
