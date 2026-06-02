# EgoVis2026 AR/VR Query Taxonomy Examples

These are heuristic multi-label assignments for source-access planning, not final ground-truth labels.

## count_quantity
- 2026_q0001: At what rate was Cathal charging his car on the last day?
  - Route: self_raw_frame;self_history;auxiliary_modality
- 2026_q0002: At what temperature did Allie put the sausages in the oven?
  - Route: self_raw_frame;auxiliary_modality
- 2026_q0009: How fast did Cathal state he was going in an 80 km/h zone?
  - Route: self_audio;self_raw_frame;auxiliary_modality
- 2026_q0010: How many carrots did Luca peel for Allie?
  - Route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality
- 2026_q0011: How many cars are parked around the house?
  - Route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality

## visual_detail_color
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0010: How many carrots did Luca peel for Allie?
  - Route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality
- 2026_q0011: How many cars are parked around the house?
  - Route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- 2026_q0012: How many chessboards are there?
  - Route: self_raw_frame;self_highres_crop;auxiliary_modality
- 2026_q0013: How many colored pens were in the box?
  - Route: self_raw_frame;self_highres_crop;auxiliary_modality

## brand_logo_ocr_text
- 2026_q0003: During the first day workshop opening presentations, where was the notebook computer placed?
  - Route: self_highres_crop;self_history;self_raw_frame;self_short_clip;auxiliary_modality
- 2026_q0022: How many Laptops were used to show the QR clock?
  - Route: self_highres_crop;self_raw_frame;auxiliary_modality
- 2026_q0027: How many of the circuit breakers in the main breaker box of the house are labelled
  - Route: self_highres_crop;self_raw_frame;static_room_source;auxiliary_modality
- 2026_q0047: What brand is the dishwasher?
  - Route: self_highres_crop;self_raw_frame;auxiliary_modality
- 2026_q0048: What brand is the fridge in the kitchen?
  - Route: self_highres_crop;self_raw_frame;static_room_source;auxiliary_modality

## spatial_location
- 2026_q0003: During the first day workshop opening presentations, where was the notebook computer placed?
  - Route: self_highres_crop;self_history;self_raw_frame;self_short_clip;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0011: How many cars are parked around the house?
  - Route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- 2026_q0015: How many cups were on the kitchen island around the sink on Day 2 in the morning?
  - Route: self_raw_frame;self_history;self_highres_crop;static_room_source;auxiliary_modality
- 2026_q0016: How many duck sculptures are on the top shelf to the right of the fireplace?
  - Route: self_raw_frame;self_highres_crop;static_room_source;auxiliary_modality

## audio_speech
- 2026_q0005: During what topic on the Happy Quiz did Florian's SD card fill up?
  - Route: self_audio;self_history;self_short_clip
- 2026_q0007: How did Florian get the answer for the quiz question "Who is the father of AI?"
  - Route: external_user_audio;self_audio;external_user_view
- 2026_q0009: How fast did Cathal state he was going in an 80 km/h zone?
  - Route: self_audio;self_raw_frame;auxiliary_modality
- 2026_q0021: How many km away did Allie live from Cambodia, according to Cathal?
  - Route: external_user_audio;self_audio;self_raw_frame;external_user_view;auxiliary_modality
- 2026_q0031: How many people in the world are called Bjorn Thor Jonsson, according to Bjorn?
  - Route: external_user_audio;self_audio;self_raw_frame;external_user_view;auxiliary_modality

## temporal_history
- 2026_q0001: At what rate was Cathal charging his car on the last day?
  - Route: self_raw_frame;self_history;auxiliary_modality
- 2026_q0003: During the first day workshop opening presentations, where was the notebook computer placed?
  - Route: self_highres_crop;self_history;self_raw_frame;self_short_clip;auxiliary_modality
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0005: During what topic on the Happy Quiz did Florian's SD card fill up?
  - Route: self_audio;self_history;self_short_clip
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality

## multi_user_copresence
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0007: How did Florian get the answer for the quiz question "Who is the father of AI?"
  - Route: external_user_audio;self_audio;external_user_view
- 2026_q0008: How did the chess game, that Allie played after winning against Werner, end?
  - Route: self_history;self_short_clip;external_user_view
- 2026_q0010: How many carrots did Luca peel for Allie?
  - Route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality

## external_source_candidate
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0007: How did Florian get the answer for the quiz question "Who is the father of AI?"
  - Route: external_user_audio;self_audio;external_user_view
- 2026_q0008: How did the chess game, that Allie played after winning against Werner, end?
  - Route: self_history;self_short_clip;external_user_view
- 2026_q0010: How many carrots did Luca peel for Allie?
  - Route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality

## raw_visual_needed
- 2026_q0001: At what rate was Cathal charging his car on the last day?
  - Route: self_raw_frame;self_history;auxiliary_modality
- 2026_q0002: At what temperature did Allie put the sausages in the oven?
  - Route: self_raw_frame;auxiliary_modality
- 2026_q0003: During the first day workshop opening presentations, where was the notebook computer placed?
  - Route: self_highres_crop;self_history;self_raw_frame;self_short_clip;auxiliary_modality
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality

## gaze_relevant
- 2026_q0001: At what rate was Cathal charging his car on the last day?
  - Route: self_raw_frame;self_history;auxiliary_modality
- 2026_q0002: At what temperature did Allie put the sausages in the oven?
  - Route: self_raw_frame;auxiliary_modality
- 2026_q0003: During the first day workshop opening presentations, where was the notebook computer placed?
  - Route: self_highres_crop;self_history;self_raw_frame;self_short_clip;auxiliary_modality
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality

## pose_fov_relevant
- 2026_q0003: During the first day workshop opening presentations, where was the notebook computer placed?
  - Route: self_highres_crop;self_history;self_raw_frame;self_short_clip;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0011: How many cars are parked around the house?
  - Route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- 2026_q0015: How many cups were on the kitchen island around the sink on Day 2 in the morning?
  - Route: self_raw_frame;self_history;self_highres_crop;static_room_source;auxiliary_modality
- 2026_q0016: How many duck sculptures are on the top shelf to the right of the fireplace?
  - Route: self_raw_frame;self_highres_crop;static_room_source;auxiliary_modality

## communication_sensitive
- 2026_q0004: During the Mamamoo Happy Quiz round, what colour is the quiz buzzer that team Cathal and Ononang used?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0006: For what challenge was Cathal working on queries while Allie and Klaus were playing chess next to him?
  - Route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- 2026_q0007: How did Florian get the answer for the quiz question "Who is the father of AI?"
  - Route: external_user_audio;self_audio;external_user_view
- 2026_q0008: How did the chess game, that Allie played after winning against Werner, end?
  - Route: self_history;self_short_clip;external_user_view
- 2026_q0010: How many carrots did Luca peel for Allie?
  - Route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality

