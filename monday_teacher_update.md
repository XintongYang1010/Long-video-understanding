# Monday Teacher Update

I reframed the project around AR/VR-native self-first evidence access rather than generic multi-agent QA.

What I prepared:

- Parsed 185 EgoVis2026 questions from the small JSON metadata file.
- Built a heuristic multi-label taxonomy for raw visual need, audio/speech, spatial location, temporal history, co-presence, external-source need, gaze, pose/FOV, and communication sensitivity.
- Mapped questions to likely minimal evidence routes: self caption, self audio, raw frame/crop, short clip, self history, external user view/audio, static room source, and auxiliary modality.
- Parsed the local CASTLE inventory with 24256 non-checksum content rows and inferred modalities.
- Selected 20 representative source-access seed cases for discussion.

Important caveat: this is a heuristic feasibility packet, not final labels, not model performance, and not benchmark completion.

Ask for Monday: should I next manually validate the 20 seed cases and turn them into source-access decision examples, or first build a small controller/simulator over the metadata routes?
