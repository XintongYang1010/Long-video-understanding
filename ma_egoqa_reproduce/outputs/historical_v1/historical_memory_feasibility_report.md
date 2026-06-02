# Historical Memory Feasibility Report V1

Scope: text/caption-only candidate generation for human audit. No videos were downloaded and no VLM/LLM was run.

## Required Questions

1. MA-EgoQA questions loaded: 1741.
2. Contexts parseable: 1739/1741.
3. Source-isolated caption memory chunks indexed: 33730.
4. Candidate questions selected: 30. Category mix: {'environmental_interaction': 6, 'task_coordination': 6, 'temporal_reasoning': 6, 'social_interaction': 6, 'theory_of_mind': 6}.
5. query_user status: unknown=23, inferred_weak=7, explicit=0. All selected cases have needs_human_review=yes.
6. historical_pool non-empty: 30/30.
7. Historical retrieval signal: all_agents_history positive in 30/30; self_history positive in 7/30; external_history positive in 7/30. These are lexical retrieval signals, not answer accuracy.
8. Five promising cases for manual audit:

- HISTV1_030 Q866 [theory_of_mind] query_user=Jake likely=needs_manual_review: Why did Jake ask if preserving ice cubes could be eaten? Signal: Jake D1 17:33:30.00-17:34:00.00: The video begins with me pushing a shopping cart through a grocery store, my hands visible on the handle. We're in an aisle with refriger... || Jake D1 11:55:00.00-11:55:30.00: The video starts with a group of us sitting aro...
- HISTV1_027 Q1013 [theory_of_mind] query_user=Jake likely=external_history_needed_candidate: Why did Jake try to direct someone to the 'first floor Huawei' during their shopping trip? Signal: Jake D5 15:53:00.00-15:53:30.00: The video starts as I'm walking through what appears to be a mall area. I notice a small shop with "Huawei" written on it, and I'm surpri... || Jake D5 15:52:30.00-15:53:00.00: I began by scrolling through WeChat, searching ...
- HISTV1_026 Q915 [theory_of_mind] query_user=Katrina likely=external_history_needed_candidate: Why was Katrina initially confused about what was happening with the ball during the game? Signal: Katrina D3 16:00:00.00-16:10:00.00: I started off by walking through a brightly lit arcade, marveling at all the games. I was with Katrina, Violet, Shure, and Jake, and we w... || Katrina D2 22:40:00.00-22:50:00.00: I was initially engrossed in my phone, th...
- HISTV1_028 Q861 [theory_of_mind] query_user=Jake likely=external_history_needed_candidate: Why did Jake joke about putting all the puzzle pieces in randomly while everyone else took the task seriously? Signal: Jake D1 12:29:00.00-12:29:30.00: I am here, looking for puzzle pieces with everyone. Tasha says, "Hmm." I'm still searching for the right piece of the puzzle. Katrina men... || Jake D1 12:29:30.00-12:30:00.00: the video starts with me joining others around ...
- HISTV1_011 Q475 [task_coordination] query_user=Tasha likely=external_history_needed_candidate: What did Tasha propose when organizing the gift exchange task? Signal: Tasha D1 19:27:00.00-19:27:30.00: Sitting on the bed, I begin by expressing how I feel full and hungry quickly, considering eating less, but more frequently as a solution.... || Tasha D1 12:36:00.00-12:36:30.00: Alright, I'm at a table on Day 1 with Jake, A...

9. Main blockers:

- query_user missing: yes. Most or all query users are unknown or weakly inferred; this blocks final self/external claims.
- context parsing hard: partly. Object-style MA-EgoQA contexts parse cleanly, but timestamp-list contexts include malformed timestamps and mixed natural-language evidence strings.
- captions too coarse: partly. 30s captions are useful; 10min captions help history but can blur source/event boundaries.
- no frames: yes. This run is text-only; no local EgoLife clips/frames were found in the inventory.
- lack of human labels: yes. All case types are provisional and need manual audit.

10. Recommended next step:

Manually audit the selected 30 cases, first checking whether the provisional query_user is meaningful. Then label only the evidence availability columns: self_current, self_history, external_current, external_history. Do not run models until a smaller verified case set exists.

## Retrieval Rows By Mode

- all_agents_current: 150
- all_agents_history: 150
- external_current: 35
- external_history: 35
- self_current: 26
- self_history: 35
