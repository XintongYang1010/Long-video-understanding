# Source-Access Seed Cases v0.3

These 20 cases are heuristic representatives for discussion. They are not benchmark results.

## raw_visual_needed_01_2026_q0010

- Question: How many carrots did Luca peel for Allie?
- Answer options: a: 5 | b: 9 | c: 3 | d: 7
- Primary category: raw_visual_needed
- Likely minimal evidence route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## raw_visual_needed_02_2026_q0011

- Question: How many cars are parked around the house?
- Answer options: a: 1 | b: 3 | c: 2 | d: 4
- Primary category: raw_visual_needed
- Likely minimal evidence route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## raw_visual_needed_03_2026_q0020

- Question: How many flower pots are on the black bench outside the front door of the house?
- Answer options: a: 2 | b: 3 | c: 4 | d: 1
- Primary category: raw_visual_needed
- Likely minimal evidence route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## raw_visual_needed_04_2026_q0012

- Question: How many chessboards are there?
- Answer options: a: One | b: Four | c: Two | d: Three
- Primary category: raw_visual_needed
- Likely minimal evidence route: self_raw_frame;self_highres_crop;auxiliary_modality
- Relevant AR signal: gaze plus high-resolution visual crop/frame
- Caption sufficiency: A caption may omit small counts, colors, labels, or fine-grained object details.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: False
- Progressive communication matters: False
- Suggested next inspection action: Inspect a timestamped raw frame or high-resolution crop; avoid VLM inference until evidence is shortlisted.

## raw_visual_needed_05_2026_q0013

- Question: How many colored pens were in the box?
- Answer options: a: 24 | b: 6 | c: 12 | d: 18
- Primary category: raw_visual_needed
- Likely minimal evidence route: self_raw_frame;self_highres_crop;auxiliary_modality
- Relevant AR signal: gaze plus high-resolution visual crop/frame
- Caption sufficiency: A caption may omit small counts, colors, labels, or fine-grained object details.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: False
- Progressive communication matters: False
- Suggested next inspection action: Inspect a timestamped raw frame or high-resolution crop; avoid VLM inference until evidence is shortlisted.

## audio_speech_01_2026_q0165

- Question: Who suggested turning the camera on during the potential power outage?
- Answer options: a: Cathal | b: Stevan | c: Florian | d: Allie
- Primary category: audio_speech
- Likely minimal evidence route: external_user_audio;self_history;self_audio;self_short_clip;external_user_view
- Relevant AR signal: audio transcript/speech attribution
- Caption sufficiency: A visual caption may miss exact wording, speaker attribution, or stated facts.
- Raw frame needed: False
- Audio needed: True
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Inspect transcript JSON around the candidate interval before requesting any frames.

## audio_speech_02_2026_q0104

- Question: What task did Luca want Bjorn and Werner to do for him when he was baking on day 2 morning?
- Answer options: a: Licking the hand mixer | b: Cleaning the pot | c: Buying more milk | d: Heating the oven
- Primary category: audio_speech
- Likely minimal evidence route: external_user_audio;self_history;self_audio;self_short_clip;external_user_view
- Relevant AR signal: audio transcript/speech attribution
- Caption sufficiency: A visual caption may miss exact wording, speaker attribution, or stated facts.
- Raw frame needed: False
- Audio needed: True
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Inspect transcript JSON around the candidate interval before requesting any frames.

## audio_speech_03_2026_q0166

- Question: Who was asked by Bjorn to chop the ginger finely?
- Answer options: a: Werner | b: Luca | c: Stevan | d: Tien
- Primary category: audio_speech
- Likely minimal evidence route: external_user_audio;self_audio;external_user_view
- Relevant AR signal: audio transcript/speech attribution
- Caption sufficiency: A visual caption may miss exact wording, speaker attribution, or stated facts.
- Raw frame needed: False
- Audio needed: True
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Inspect transcript JSON around the candidate interval before requesting any frames.

## audio_speech_04_2026_q0078

- Question: What game did Bjorn suggest to play with the cameras?
- Answer options: a: Hide and seek | b: Chicken | c: Freeze tag | d: Red light, green light
- Primary category: audio_speech
- Likely minimal evidence route: external_user_audio;self_audio;self_raw_frame;external_user_view;auxiliary_modality
- Relevant AR signal: audio transcript/speech attribution
- Caption sufficiency: A visual caption may miss exact wording, speaker attribution, or stated facts.
- Raw frame needed: True
- Audio needed: True
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Inspect transcript JSON around the candidate interval before requesting any frames.

## audio_speech_05_2026_q0021

- Question: How many km away did Allie live from Cambodia, according to Cathal?
- Answer options: a: 10 km | b: 100 km | c: 250 km | d: 50 km
- Primary category: audio_speech
- Likely minimal evidence route: external_user_audio;self_audio;self_raw_frame;external_user_view;auxiliary_modality
- Relevant AR signal: audio transcript/speech attribution
- Caption sufficiency: A visual caption may miss exact wording, speaker attribution, or stated facts.
- Raw frame needed: True
- Audio needed: True
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Inspect transcript JSON around the candidate interval before requesting any frames.

## spatial_location_pose_fov_01_2026_q0114

- Question: What was the brand of model airplane that was sitting on the table while people were building a tower with the wooden blocks?
- Answer options: a: Superfly | b: Revell | c: Tamina | d: Airfix
- Primary category: spatial_location_pose_fov
- Likely minimal evidence route: self_highres_crop;self_history;self_raw_frame;self_short_clip;external_user_view;auxiliary_modality
- Relevant AR signal: pose/FOV and gaze for viewpoint narrowing
- Caption sufficiency: A caption may describe the scene but miss exact placement, viewpoint, or off-FOV context.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Use pose/FOV and gaze metadata to shortlist the interval, then inspect a frame or static view if needed.

## spatial_location_pose_fov_02_2026_q0093

- Question: What number is on the door of Cathal's room?
- Answer options: a: 7 | b: 3 | c: 9 | d: 5
- Primary category: spatial_location_pose_fov
- Likely minimal evidence route: self_raw_frame;external_user_view;static_room_source;auxiliary_modality
- Relevant AR signal: pose/FOV and gaze for viewpoint narrowing
- Caption sufficiency: A caption may describe the scene but miss exact placement, viewpoint, or off-FOV context.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Use pose/FOV and gaze metadata to shortlist the interval, then inspect a frame or static view if needed.

## spatial_location_pose_fov_03_2026_q0030

- Question: How many paintings are hanging over the large couch?
- Answer options: a: 3 | b: 6 | c: 4 | d: 5
- Primary category: spatial_location_pose_fov
- Likely minimal evidence route: self_raw_frame;self_highres_crop;static_room_source;auxiliary_modality
- Relevant AR signal: pose/FOV and gaze for viewpoint narrowing
- Caption sufficiency: A caption may describe the scene but miss exact placement, viewpoint, or off-FOV context.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: False
- Progressive communication matters: False
- Suggested next inspection action: Use pose/FOV and gaze metadata to shortlist the interval, then inspect a frame or static view if needed.

## spatial_location_pose_fov_04_2026_q0025

- Question: How many lights are hanging over the kitchen island?
- Answer options: a: 3 | b: 1 | c: 4 | d: 2
- Primary category: spatial_location_pose_fov
- Likely minimal evidence route: self_raw_frame;self_highres_crop;static_room_source;auxiliary_modality
- Relevant AR signal: pose/FOV and gaze for viewpoint narrowing
- Caption sufficiency: A caption may describe the scene but miss exact placement, viewpoint, or off-FOV context.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: False
- Progressive communication matters: False
- Suggested next inspection action: Use pose/FOV and gaze metadata to shortlist the interval, then inspect a frame or static view if needed.

## spatial_location_pose_fov_05_2026_q0127

- Question: Where is the yellow octopus when Linh is playing chess with Allie?
- Answer options: a: In his back pocket | b: Next to his elbow | c: On his head | d: On the chair behind him
- Primary category: spatial_location_pose_fov
- Likely minimal evidence route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- Relevant AR signal: pose/FOV and gaze for viewpoint narrowing
- Caption sufficiency: A caption may describe the scene but miss exact placement, viewpoint, or off-FOV context.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Use pose/FOV and gaze metadata to shortlist the interval, then inspect a frame or static view if needed.

## multi_user_external_source_candidate_01_2026_q0184

- Question: Who won the first round of UNO?
- Answer options: a: Tien | b: Klaus | c: Bjorn | d: Allie
- Primary category: multi_user_external_source_candidate
- Likely minimal evidence route: self_history;external_user_view
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: False
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## multi_user_external_source_candidate_02_2026_q0151

- Question: Who had a bandage on their finger?
- Answer options: a: Allie | b: Bao | c: Klaus | d: Luca
- Primary category: multi_user_external_source_candidate
- Likely minimal evidence route: self_caption;external_user_view
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: False
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## multi_user_external_source_candidate_03_2026_q0150

- Question: Who got the center piece of lasagna?
- Answer options: a: Florian | b: Linh | c: Allie | d: Luca
- Primary category: multi_user_external_source_candidate
- Likely minimal evidence route: self_caption;external_user_view
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: False
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## multi_user_external_source_candidate_04_2026_q0164

- Question: Who set the Christmas pudding on fire?
- Answer options: a: Stevan | b: Florian | c: Luca | d: Cathal
- Primary category: multi_user_external_source_candidate
- Likely minimal evidence route: self_caption;external_user_view
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: False
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

## multi_user_external_source_candidate_05_2026_q0161

- Question: Who put the Christmas tree in the box?
- Answer options: a: Florian | b: Linh | c: Onanong | d: Klaus
- Primary category: multi_user_external_source_candidate
- Likely minimal evidence route: self_raw_frame;external_user_view;auxiliary_modality
- Relevant AR signal: co-presence, source ownership, and static/other-user view routing
- Caption sufficiency: A self caption may not include evidence from another participant or static room camera.
- Raw frame needed: True
- Audio needed: False
- External source may be needed: True
- Progressive communication matters: True
- Suggested next inspection action: Compare self metadata with other-user/static source availability before pulling external evidence.

