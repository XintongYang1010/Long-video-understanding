# Source-Access Seed Case Validation v0.3

First-pass validation only. These are not final labels, benchmark annotations, or model-performance claims.

No videos were downloaded or decoded. No frames were extracted. No VLM inference was run.

## Summary

- weakened: 16
- candidate_only: 4
- Cases needing transcript/audio first: 5
- Cases with local transcript keyword evidence: 0
- Cases needing targeted frame extraction later: 15
- Cases where static camera is preferred before external user view: 10
- Cases where external user view is verified required: 0

## Key Distinctions

- `external_user_candidate` means another user's source might help.
- `external_user_verified_needed` remains false unless cheaper self, transcript, and static-room evidence is insufficient.
- Speech cases are routed to transcript/audio first; frames are not requested for them in this pass.
- Static room sources are preferred over external user views for indoor room-level spatial or multi-user questions.

## Cases

### raw_visual_needed_01_2026_q0010

- Question: How many carrots did Luca peel for Allie?
- Bucket: raw_visual_needed
- Proposed route: self_raw_frame;self_highres_crop;external_user_view;auxiliary_modality
- First-pass route: static_room_source;self_raw_frame
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view. External_user_view is not verified as required in this pass.
- Next action: Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window.
- Evidence source candidates: `main/day1/Kitchen/video/08.mp4; main/day1/Kitchen/video/09.mp4; main/day1/Kitchen/video/10.mp4; main/day1/Kitchen/video/11.mp4; main/day1/Kitchen/video/12.mp4; main/day1/Kitchen/video/13.mp4; main/day1/Kitchen/video/14.mp4; main/day1/Kitchen/video/15.mp4; main/day1/Kitchen/video/16.mp4; main/day1/Kitchen/video/17.mp4; main/day1/Kitchen/video/18.mp4; main/day1/Kitchen/video/19.mp4`

### raw_visual_needed_02_2026_q0011

- Question: How many cars are parked around the house?
- Bucket: raw_visual_needed
- Proposed route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- First-pass route: self_raw_frame;auxiliary_modality
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Small counts, labels, object placement, or outdoor context are often compressed out of captions. External_user_view is not verified as required in this pass.
- Next action: Find the event window, then extract a small number of candidate frames/crops with ffmpeg.
- Evidence source candidates: `main/day1/Allie/video/08.mp4; main/day1/Allie/video/09.mp4; main/day1/Allie/video/10.mp4; main/day1/Allie/video/11.mp4; main/day1/Allie/video/12.mp4; main/day1/Allie/video/13.mp4; main/day1/Allie/video/14.mp4; main/day1/Allie/video/15.mp4; main/day1/Allie/video/16.novideo; main/day1/Allie/video/17.novideo; main/day1/Allie/video/18.mp4; main/day1/Allie/video/19.mp4`

### raw_visual_needed_03_2026_q0020

- Question: How many flower pots are on the black bench outside the front door of the house?
- Bucket: raw_visual_needed
- Proposed route: self_raw_frame;self_highres_crop;external_user_view;static_room_source;auxiliary_modality
- First-pass route: self_raw_frame;auxiliary_modality
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Small counts, labels, object placement, or outdoor context are often compressed out of captions. External_user_view is not verified as required in this pass.
- Next action: Find the event window, then extract a small number of candidate frames/crops with ffmpeg.
- Evidence source candidates: `main/day1/Allie/video/08.mp4; main/day1/Allie/video/09.mp4; main/day1/Allie/video/10.mp4; main/day1/Allie/video/11.mp4; main/day1/Allie/video/12.mp4; main/day1/Allie/video/13.mp4; main/day1/Allie/video/14.mp4; main/day1/Allie/video/15.mp4; main/day1/Allie/video/16.novideo; main/day1/Allie/video/17.novideo; main/day1/Allie/video/18.mp4; main/day1/Allie/video/19.mp4`

### raw_visual_needed_04_2026_q0012

- Question: How many chessboards are there?
- Bucket: raw_visual_needed
- Proposed route: self_raw_frame;self_highres_crop;auxiliary_modality
- First-pass route: static_room_source;self_raw_frame
- Status: candidate_only
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: False/False
- Why: Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view.
- Next action: Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window.
- Evidence source candidates: `main/day1/Living1/video/08.mp4; main/day1/Living1/video/09.mp4; main/day1/Living1/video/10.mp4; main/day1/Living1/video/11.mp4; main/day1/Living1/video/12.mp4; main/day1/Living1/video/13.mp4; main/day1/Living1/video/14.mp4; main/day1/Living1/video/15.mp4; main/day1/Living1/video/16.mp4; main/day1/Living1/video/17.mp4; main/day1/Living1/video/18.mp4; main/day1/Living1/video/19.mp4`

### raw_visual_needed_05_2026_q0013

- Question: How many colored pens were in the box?
- Bucket: raw_visual_needed
- Proposed route: self_raw_frame;self_highres_crop;auxiliary_modality
- First-pass route: self_raw_frame
- Status: candidate_only
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: False
- External user candidate/verified needed: False/False
- Why: Small counts, labels, object placement, or outdoor context are often compressed out of captions.
- Next action: Find the event window, then extract a small number of candidate frames/crops with ffmpeg.
- Evidence source candidates: `main/day1/Allie/video/08.mp4; main/day1/Allie/video/09.mp4; main/day1/Allie/video/10.mp4; main/day1/Allie/video/11.mp4; main/day1/Allie/video/12.mp4; main/day1/Allie/video/13.mp4; main/day1/Allie/video/14.mp4; main/day1/Allie/video/15.mp4; main/day1/Allie/video/16.novideo; main/day1/Allie/video/17.novideo; main/day1/Allie/video/18.mp4; main/day1/Allie/video/19.mp4`

### audio_speech_01_2026_q0165

- Question: Who suggested turning the camera on during the potential power outage?
- Bucket: audio_speech
- Proposed route: external_user_audio;self_history;self_audio;self_short_clip;external_user_view
- First-pass route: transcript_audio
- Status: weakened
- Transcript needed/found: True/False
- Raw frame needed/found: False/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Question asks for spoken suggestion, request, or stated fact; transcript/audio is cheaper than frames. External_user_view is not verified as required in this pass.
- Next action: Locate raw CASTLE transcript JSON for the candidate event window and search before requesting video frames.
- Evidence source candidates: `main/day1/Meeting/transcript/08.json; main/day1/Meeting/transcript/10.json; main/day1/Meeting/transcript/11.json; main/day1/Meeting/transcript/12.json; main/day1/Meeting/transcript/13.json; main/day1/Meeting/transcript/14.json; main/day1/Meeting/transcript/15.json; main/day1/Meeting/transcript/16.json; main/day1/Meeting/transcript/17.json; main/day1/Meeting/transcript/18.json; main/day1/Meeting/transcript/19.json; main/day1/Meeting/transcript/20.json`

### audio_speech_02_2026_q0104

- Question: What task did Luca want Bjorn and Werner to do for him when he was baking on day 2 morning?
- Bucket: audio_speech
- Proposed route: external_user_audio;self_history;self_audio;self_short_clip;external_user_view
- First-pass route: transcript_audio
- Status: weakened
- Transcript needed/found: True/False
- Raw frame needed/found: False/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Question asks for spoken suggestion, request, or stated fact; transcript/audio is cheaper than frames. External_user_view is not verified as required in this pass.
- Next action: Locate raw CASTLE transcript JSON for the candidate event window and search before requesting video frames.
- Evidence source candidates: `main/day2/Bjorn/transcript/10.json; main/day2/Bjorn/transcript/11.json; main/day2/Bjorn/transcript/12.json; main/day2/Bjorn/transcript/14.json; main/day2/Bjorn/transcript/15.json; main/day2/Bjorn/transcript/16.json; main/day2/Bjorn/transcript/17.json; main/day2/Bjorn/transcript/18.json; main/day2/Bjorn/transcript/19.json; main/day2/Bjorn/transcript/20.json; main/day2/Luca/transcript/09.json; main/day2/Luca/transcript/10.json`

### audio_speech_03_2026_q0166

- Question: Who was asked by Bjorn to chop the ginger finely?
- Bucket: audio_speech
- Proposed route: external_user_audio;self_audio;external_user_view
- First-pass route: transcript_audio
- Status: weakened
- Transcript needed/found: True/False
- Raw frame needed/found: False/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Question asks for spoken suggestion, request, or stated fact; transcript/audio is cheaper than frames. External_user_view is not verified as required in this pass.
- Next action: Locate raw CASTLE transcript JSON for the candidate event window and search before requesting video frames.
- Evidence source candidates: `main/day1/Bjorn/transcript/08.json; main/day1/Bjorn/transcript/10.json; main/day1/Bjorn/transcript/11.json; main/day1/Bjorn/transcript/12.json; main/day1/Bjorn/transcript/13.json; main/day1/Bjorn/transcript/14.json; main/day1/Bjorn/transcript/15.json; main/day1/Bjorn/transcript/16.json; main/day1/Bjorn/transcript/17.json; main/day1/Bjorn/transcript/18.json; main/day1/Bjorn/transcript/19.json; main/day1/Bjorn/transcript/20.json`

### audio_speech_04_2026_q0078

- Question: What game did Bjorn suggest to play with the cameras?
- Bucket: audio_speech
- Proposed route: external_user_audio;self_audio;self_raw_frame;external_user_view;auxiliary_modality
- First-pass route: transcript_audio
- Status: weakened
- Transcript needed/found: True/False
- Raw frame needed/found: False/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Question asks for spoken suggestion, request, or stated fact; transcript/audio is cheaper than frames. External_user_view is not verified as required in this pass.
- Next action: Locate raw CASTLE transcript JSON for the candidate event window and search before requesting video frames.
- Evidence source candidates: `main/day1/Bjorn/transcript/08.json; main/day1/Bjorn/transcript/10.json; main/day1/Bjorn/transcript/11.json; main/day1/Bjorn/transcript/12.json; main/day1/Bjorn/transcript/13.json; main/day1/Bjorn/transcript/14.json; main/day1/Bjorn/transcript/15.json; main/day1/Bjorn/transcript/16.json; main/day1/Bjorn/transcript/17.json; main/day1/Bjorn/transcript/18.json; main/day1/Bjorn/transcript/19.json; main/day1/Bjorn/transcript/20.json`

### audio_speech_05_2026_q0021

- Question: How many km away did Allie live from Cambodia, according to Cathal?
- Bucket: audio_speech
- Proposed route: external_user_audio;self_audio;self_raw_frame;external_user_view;auxiliary_modality
- First-pass route: transcript_audio
- Status: weakened
- Transcript needed/found: True/False
- Raw frame needed/found: False/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Question asks for spoken suggestion, request, or stated fact; transcript/audio is cheaper than frames. External_user_view is not verified as required in this pass.
- Next action: Locate raw CASTLE transcript JSON for the candidate event window and search before requesting video frames.
- Evidence source candidates: `main/day1/Allie/transcript/08.json; main/day1/Allie/transcript/09.json; main/day1/Allie/transcript/10.json; main/day1/Allie/transcript/11.json; main/day1/Allie/transcript/12.json; main/day1/Allie/transcript/13.json; main/day1/Allie/transcript/14.json; main/day1/Allie/transcript/15.json; main/day1/Allie/transcript/18.json; main/day1/Allie/transcript/19.json; main/day1/Allie/transcript/20.json; main/day1/Cathal/transcript/08.json`

### spatial_location_pose_fov_01_2026_q0114

- Question: What was the brand of model airplane that was sitting on the table while people were building a tower with the wooden blocks?
- Bucket: spatial_location_pose_fov
- Proposed route: self_highres_crop;self_history;self_raw_frame;self_short_clip;external_user_view;auxiliary_modality
- First-pass route: static_room_source;self_raw_frame
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view. External_user_view is not verified as required in this pass.
- Next action: Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window.
- Evidence source candidates: `main/day1/Meeting/video/08.mp4; main/day1/Meeting/video/09.mp4; main/day1/Meeting/video/10.mp4; main/day1/Meeting/video/11.mp4; main/day1/Meeting/video/12.mp4; main/day1/Meeting/video/13.mp4; main/day1/Meeting/video/14.mp4; main/day1/Meeting/video/15.mp4; main/day1/Meeting/video/16.mp4; main/day1/Meeting/video/17.mp4; main/day1/Meeting/video/18.mp4; main/day1/Meeting/video/19.mp4`

### spatial_location_pose_fov_02_2026_q0093

- Question: What number is on the door of Cathal's room?
- Bucket: spatial_location_pose_fov
- Proposed route: self_raw_frame;external_user_view;static_room_source;auxiliary_modality
- First-pass route: self_raw_frame
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: Small counts, labels, object placement, or outdoor context are often compressed out of captions. External_user_view is not verified as required in this pass.
- Next action: Find the event window, then extract a small number of candidate frames/crops with ffmpeg.
- Evidence source candidates: `main/day1/Cathal/video/08.mp4; main/day1/Cathal/video/09.mp4; main/day1/Cathal/video/10.mp4; main/day1/Cathal/video/11.mp4; main/day1/Cathal/video/12.mp4; main/day1/Cathal/video/13.mp4; main/day1/Cathal/video/14.mp4; main/day1/Cathal/video/15.mp4; main/day1/Cathal/video/16.mp4; main/day1/Cathal/video/17.mp4; main/day1/Cathal/video/18.mp4; main/day1/Cathal/video/19.mp4`

### spatial_location_pose_fov_03_2026_q0030

- Question: How many paintings are hanging over the large couch?
- Bucket: spatial_location_pose_fov
- Proposed route: self_raw_frame;self_highres_crop;static_room_source;auxiliary_modality
- First-pass route: static_room_source;self_raw_frame
- Status: candidate_only
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: False/False
- Why: Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view.
- Next action: Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window.
- Evidence source candidates: `main/day1/Living1/video/08.mp4; main/day1/Living1/video/09.mp4; main/day1/Living1/video/10.mp4; main/day1/Living1/video/11.mp4; main/day1/Living1/video/12.mp4; main/day1/Living1/video/13.mp4; main/day1/Living1/video/14.mp4; main/day1/Living1/video/15.mp4; main/day1/Living1/video/16.mp4; main/day1/Living1/video/17.mp4; main/day1/Living1/video/18.mp4; main/day1/Living1/video/19.mp4`

### spatial_location_pose_fov_04_2026_q0025

- Question: How many lights are hanging over the kitchen island?
- Bucket: spatial_location_pose_fov
- Proposed route: self_raw_frame;self_highres_crop;static_room_source;auxiliary_modality
- First-pass route: static_room_source;self_raw_frame
- Status: candidate_only
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: False/False
- Why: Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view.
- Next action: Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window.
- Evidence source candidates: `main/day1/Kitchen/video/08.mp4; main/day1/Kitchen/video/09.mp4; main/day1/Kitchen/video/10.mp4; main/day1/Kitchen/video/11.mp4; main/day1/Kitchen/video/12.mp4; main/day1/Kitchen/video/13.mp4; main/day1/Kitchen/video/14.mp4; main/day1/Kitchen/video/15.mp4; main/day1/Kitchen/video/16.mp4; main/day1/Kitchen/video/17.mp4; main/day1/Kitchen/video/18.mp4; main/day1/Kitchen/video/19.mp4`

### spatial_location_pose_fov_05_2026_q0127

- Question: Where is the yellow octopus when Linh is playing chess with Allie?
- Bucket: spatial_location_pose_fov
- Proposed route: self_raw_frame;self_history;self_short_clip;external_user_view;auxiliary_modality
- First-pass route: static_room_source;self_raw_frame
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Indoor spatial/count/detail evidence can often be checked with a static room source before asking another user's view. External_user_view is not verified as required in this pass.
- Next action: Use inventory/event metadata to identify the room camera and then run targeted ffmpeg frame extraction only for that window.
- Evidence source candidates: `main/day1/Living1/video/08.mp4; main/day1/Living1/video/09.mp4; main/day1/Living1/video/10.mp4; main/day1/Living1/video/11.mp4; main/day1/Living1/video/12.mp4; main/day1/Living1/video/13.mp4; main/day1/Living1/video/14.mp4; main/day1/Living1/video/15.mp4; main/day1/Living1/video/16.mp4; main/day1/Living1/video/17.mp4; main/day1/Living1/video/18.mp4; main/day1/Living1/video/19.mp4`

### multi_user_external_source_candidate_01_2026_q0184

- Question: Who won the first round of UNO?
- Bucket: multi_user_external_source_candidate
- Proposed route: self_history;external_user_view
- First-pass route: static_room_source
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Multi-user room events can often be resolved by static camera before external user view. External_user_view is not verified as required in this pass.
- Next action: Prefer static room camera frame/clip for the event window; request external user view only if static view is occluded.
- Evidence source candidates: `main/day1/Living1/video/08.mp4; main/day1/Living1/video/09.mp4; main/day1/Living1/video/10.mp4; main/day1/Living1/video/11.mp4; main/day1/Living1/video/12.mp4; main/day1/Living1/video/13.mp4; main/day1/Living1/video/14.mp4; main/day1/Living1/video/15.mp4; main/day1/Living1/video/16.mp4; main/day1/Living1/video/17.mp4; main/day1/Living1/video/18.mp4; main/day1/Living1/video/19.mp4`

### multi_user_external_source_candidate_02_2026_q0151

- Question: Who had a bandage on their finger?
- Bucket: multi_user_external_source_candidate
- Proposed route: self_caption;external_user_view
- First-pass route: self_history;static_room_source
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: False
- External user candidate/verified needed: True/False
- Why: The event attribution is plausible from history/static evidence; external user view remains only a candidate. External_user_view is not verified as required in this pass.
- Next action: Check self history/transcript/static room inventory for event window before external user evidence.
- Evidence source candidates: `main/day1/Allie/video/08.mp4; main/day1/Allie/video/09.mp4; main/day1/Allie/video/10.mp4; main/day1/Allie/video/11.mp4; main/day1/Allie/video/12.mp4; main/day1/Allie/video/13.mp4; main/day1/Allie/video/14.mp4; main/day1/Allie/video/15.mp4; main/day1/Allie/video/16.novideo; main/day1/Allie/video/17.novideo; main/day1/Allie/video/18.mp4; main/day1/Allie/video/19.mp4`

### multi_user_external_source_candidate_03_2026_q0150

- Question: Who got the center piece of lasagna?
- Bucket: multi_user_external_source_candidate
- Proposed route: self_caption;external_user_view
- First-pass route: static_room_source
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Multi-user room events can often be resolved by static camera before external user view. External_user_view is not verified as required in this pass.
- Next action: Prefer static room camera frame/clip for the event window; request external user view only if static view is occluded.
- Evidence source candidates: `main/day1/Kitchen/video/08.mp4; main/day1/Kitchen/video/09.mp4; main/day1/Kitchen/video/10.mp4; main/day1/Kitchen/video/11.mp4; main/day1/Kitchen/video/12.mp4; main/day1/Kitchen/video/13.mp4; main/day1/Kitchen/video/14.mp4; main/day1/Kitchen/video/15.mp4; main/day1/Kitchen/video/16.mp4; main/day1/Kitchen/video/17.mp4; main/day1/Kitchen/video/18.mp4; main/day1/Kitchen/video/19.mp4`

### multi_user_external_source_candidate_04_2026_q0164

- Question: Who set the Christmas pudding on fire?
- Bucket: multi_user_external_source_candidate
- Proposed route: self_caption;external_user_view
- First-pass route: static_room_source
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Multi-user room events can often be resolved by static camera before external user view. External_user_view is not verified as required in this pass.
- Next action: Prefer static room camera frame/clip for the event window; request external user view only if static view is occluded.
- Evidence source candidates: `main/day1/Kitchen/video/08.mp4; main/day1/Kitchen/video/09.mp4; main/day1/Kitchen/video/10.mp4; main/day1/Kitchen/video/11.mp4; main/day1/Kitchen/video/12.mp4; main/day1/Kitchen/video/13.mp4; main/day1/Kitchen/video/14.mp4; main/day1/Kitchen/video/15.mp4; main/day1/Kitchen/video/16.mp4; main/day1/Kitchen/video/17.mp4; main/day1/Kitchen/video/18.mp4; main/day1/Kitchen/video/19.mp4`

### multi_user_external_source_candidate_05_2026_q0161

- Question: Who put the Christmas tree in the box?
- Bucket: multi_user_external_source_candidate
- Proposed route: self_raw_frame;external_user_view;auxiliary_modality
- First-pass route: static_room_source
- Status: weakened
- Transcript needed/found: False/False
- Raw frame needed/found: True/False
- Static camera needed: True
- External user candidate/verified needed: True/False
- Why: Multi-user room events can often be resolved by static camera before external user view. External_user_view is not verified as required in this pass.
- Next action: Prefer static room camera frame/clip for the event window; request external user view only if static view is occluded.
- Evidence source candidates: `main/day1/Living1/video/08.mp4; main/day1/Living1/video/09.mp4; main/day1/Living1/video/10.mp4; main/day1/Living1/video/11.mp4; main/day1/Living1/video/12.mp4; main/day1/Living1/video/13.mp4; main/day1/Living1/video/14.mp4; main/day1/Living1/video/15.mp4; main/day1/Living1/video/16.mp4; main/day1/Living1/video/17.mp4; main/day1/Living1/video/18.mp4; main/day1/Living1/video/19.mp4`

