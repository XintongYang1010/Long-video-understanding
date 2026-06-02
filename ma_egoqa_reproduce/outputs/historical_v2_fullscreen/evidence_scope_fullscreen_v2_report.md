# Evidence Scope Full Screening V2 Report

Scope: caption-only full screening over MA-EgoQA questions to find candidate current-only vs current+historical evidence coverage cases. No videos were downloaded, no VLM/LLM/API was used, original MA-EgoQA files were not modified, and these rows are not final labels.

## Configuration

- Retrieval query mode: question + gold answer text, for candidate evidence discovery only.
- Top-k per scope: 10.
- Current caption margin around parsed context windows: 120 seconds.
- Historical evidence rule: caption end_time must be before parsed context_start; same-day future captions are excluded from history.
- Top-k provenance table: `evidence_scope_fullscreen_v2_topk_evidence.csv`.

## Required Counts

1. Total MA-EgoQA questions loaded: 1741.
2. Parseable questions: 1739.
3. Processed questions: 1739.
4. Tier A/B/C/D/E counts: A=22, B=111, C=1190, D=137, E=279.
5. current-only answerability distribution: {'likely_answerable': 1190, 'partially_answerable': 407, 'unclear': 97, 'not_answerable': 45}.
6. history-only answerability distribution: {'likely_answerable': 1197, 'partially_answerable': 388, 'unclear': 100, 'not_answerable': 54}.
7. current+historical answerability distribution: {'likely_answerable': 1227, 'partially_answerable': 382, 'unclear': 91, 'not_answerable': 39}.
8. current+historical better than current-only count: 73 yes; 95 unclear_but_promising.

## Top 20 Tier A/B Cases To Inspect

- Q313 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] When Katrina discussed her breakup, how did the group mostly respond? | history: [history_only rank=1 score=24.9177 source=Shure D2 22:32:00.00-22:32:30.00 30sec] The video begins with me sitting with a group of people as Katrina shares a story about her breakup and how her ex-boyfriend attempted...
- Q744 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What part of the shared task did Shure take responsibility for after lunch plans? | history: [history_only rank=1 score=14.7359 source=Shure D6 13:04:30.00-13:05:00.00 30sec] I'm diligently adjusting sound effects on my laptop, visible in front of me. It's proving to be a painstaking process, requiring carefu...
- Q981 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What did Nicous think when Katrina asked to see his phone while discussing the letter? | history: [history_only rank=1 score=21.6088 source=Lucia D2 21:49:30.00-21:50:00.00 30sec] the video begins with Nicous playing with his hair while discussing bracelets, mentioning they are thirty yuan each and that he bought...
- Q419 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] How was coordination handled in dividing cleanup responsibilities? | history: [history_only rank=1 score=11.8798 source=Lucia D1 13:01:30.00-13:02:00.00 30sec] The video begins with me noticing that the software interface has unexpectedly changed to "network Arrow" after unplugging it, causing...
- Q437 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] How did Jake contribute to the group's tasks during the time Shure prepared coffee? | history: [history_only rank=1 score=14.5590 source=Jake D2 13:55:00.00-13:55:30.00 30sec] The video begins with me glancing in the direction of Nicous while hearing comments about latte art and a coffee machine's supposed faul...
- Q617 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] What decision-making role did Jake take in the group task involving glasses? | history: [history_only rank=1 score=17.1330 source=Katrina D2 12:05:30.00-12:06:00.00 30sec] I initiate a turn, and then proceed to descend a set of stairs while holding a sheet of stickers. Throughout my descent, I overhear a...
- Q1073 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] Why did Jake smile and respond to Nicous by referring to a 'national guardian'? | history: [history_only rank=1 score=15.5212 source=Jake D6 19:32:30.00-19:33:00.00 30sec] I am walking forward while listening to Lucia, who is commenting on a video, stating, "The last one, the last one is me," and "From that...
- Q1078 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] Why did Shure laugh when Katrina mentioned her voice messages? | history: [history_only rank=1 score=19.1031 source=Katrina D4 16:43:30.00-16:44:00.00 30sec] I'm sitting at a table in what appears to be a living room and kitchen area. Shure is standing and chatting with me, sharing about hi...
- Q508 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What decision did the group make when the milk mixture became insufficient? | history: [history_only rank=1 score=16.7211 source=Alice D2 10:40:00.00-10:50:00.00 10min] we were all hanging out in the kitchen, probably getting ready for lunch on Day 2. I started by grabbing the coffee, but then we decide...
- Q676 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What method did Katrina and Shure align on for preparing their creative project? | history: [history_only rank=1 score=22.2628 source=Katrina D6 11:05:00.00-11:05:30.00 30sec] I'm continuing to work on this project, carefully smoothing out the details and trying to make everything fit together seamlessly. I...
- Q1004 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] Why did Katrina think writing about dogecoin on her profile was funny? | history: [history_only rank=1 score=12.8198 source=Katrina D2 22:08:30.00-22:09:00.00 30sec] The video begins at approximately 22:08:30, labeled as "DAY 2", and presents a dark, egocentric view of an outdoor setting where I am...
- Q503 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What role did Lucia play in the team’s collaboration efforts? | history: [history_only rank=1 score=19.8280 source=Lucia D1 12:33:00.00-12:33:30.00 30sec] everyone's attention seems to be captured by something small and intriguing. Lucia speculates, "Automatic detection?" then adds, "It th...
- Q698 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What challenge did Shure highlight while helping assist Nious with grocery tasks? | history: [history_only rank=1 score=18.2859 source=Tasha D3 12:05:30.00-12:06:00.00 30sec] I'm at a table with a group of people, including Alice, Shure, Jake, and Lucia, and we are having pizza. I am wearing a green jacket an...
- Q880 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] Why does Katrina think there might not be memory left on a device, leading to confusion with Jake? | history: [history_only rank=1 score=27.0298 source=Tasha D1 19:08:00.00-19:08:30.00 30sec] I am in conversation with Jake and Lucia as we seem to be taking a break, possibly due to a memory lapse regarding a task at hand; Luci...
- Q1040 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] Who incorrectly assumed everyone would want an auction group of four items? | history: [history_only rank=1 score=19.3051 source=Tasha D1 12:05:30.00-12:06:00.00 30sec] I am participating in a meeting where the topic of discussion revolves around an auction. Katrina initiates the conversation by asking,...
- Q433 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] Why did they decide on having transition shots in the choreography? | history: [history_only rank=1 score=15.7940 source=Katrina D2 11:40:00.00-11:50:00.00 10min] Sitting on the floor, I started watching videos on my tablet, a mix of dance performances and random content. I remember questioning...
- Q523 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] What supported the collaborative mood around the cake task? | history: [history_only rank=1 score=18.9686 source=Lucia D1 21:48:00.00-21:48:30.00 30sec] I ate the cake entirely. I hear Alice asking, "Can you open it from the outside?" and then stating, "Let me see if it can lock" and lat...
- Q700 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] Who was concerned with sugar quantity during ingredient preparation, and who executed the task? | history: [history_only rank=1 score=30.2231 source=Lucia D3 20:23:30.00-20:24:00.00 30sec] We laughed while falling backwards out of frame at the start of this video. I'm currently in the middle of a baking session, attempting...
- Q982 [tier_A_demo_ready; gain=yes; current=unclear; plus=likely_answerable] Why didn't Katrina seem fully involved in the conversation about renting costumes during 18:23? | history: [history_only rank=1 score=22.2542 source=Jake D4 13:20:00.00-13:30:00.00 10min] I was sitting around a table with Tasha, Lucia, Jake, Nicous, Katrina, and later Shure, enjoying a dumpling feast. Tasha was showing me...
- Q846 [tier_A_demo_ready; gain=yes; current=partially_answerable; plus=likely_answerable] What consistent roles did people play in egg preparation across two events? | history: [history_only rank=1 score=25.0658 source=Jake D3 20:46:30.00-20:47:00.00 30sec] The video begins with a conversation while walking outside, hearing Shure saying "Steady, steady" and Katrina complimenting with "You're...

## Why V1 Only Produced 1 Constructed Case

- V1 first narrowed the problem to 30 heuristic candidates instead of screening all 1,741 questions.
- It used conservative gates around known/provisional query_user, non-global questions, current+history gain, and constructed-case confidence.
- The constructed subset required readable caption evidence and rejected many global/statistical/order questions where caption retrieval looked too local.
- Therefore V1 was useful as a pilot, but its candidate pool and final construction criteria were too small and conservative to test whether historical evidence has broader signal.

## Preliminary Signal

- Caption-only full screening suggests candidate cases where current+historical evidence may improve coverage over current-only (73 yes and 95 unclear_but_promising under the heuristic).
- This is preliminary evidence coverage signal only. It should be used to choose cases for manual inspection and later controlled packet construction.

## Current Claim Boundary

- Can claim: caption-only full screening suggests candidate cases where current+historical evidence may improve coverage over current-only.
- Cannot claim: benchmark complete, labels final, model accuracy improves, historical memory proven useful, or self-first routing solved.
- Each row keeps `needs_user_check=yes`; answerability is a conservative draft, not answer accuracy.

## Additional Diagnostics

- query_user status distribution: {'unknown': 1316, 'inferred_weak': 423}.
- gain distribution: {'no': 1533, 'yes': 73, 'unclear_but_promising': 95, 'unclear': 38}.

