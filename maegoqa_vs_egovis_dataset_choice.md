# MA-EgoQA vs EgoVis Dataset Choice

Recommended framing for the Monday/Tuesday packet: use EgoVis2026 questions as the primary taxonomy surface, and use MA-EgoQA as auxiliary context for multi-agent memory/history schema comparison.

## Why EgoVis2026 Is Primary Here

- The downloaded EgoVis JSON provides compact, direct multiple-choice questions that can be rapidly labeled for evidence demand.
- The questions include visual detail, counting, spatial layout, spoken facts, and co-presence cases that map cleanly to AR/VR source-access routes.
- The taxonomy can be produced without videos, model inference, or full dataset downloads.

## Where MA-EgoQA Helps

- MA-EgoQA has explicit multi-agent contexts and caption folders at 30-second, 1-hour, 10-minute, and 1-day granularities.
- It is useful for self-history and multi-agent context schema, but the local QA schema does not appear to expose explicit raw-evidence labels.
- It can support future comparisons between caption/history baselines and evidence-routing needs.

## Meeting Position

Do not claim benchmark completion. Claim a metadata-only feasibility packet showing that AR/VR-native signals can route questions to self caption, self audio, raw frames, history, external user sources, static room sources, or auxiliary modalities.
