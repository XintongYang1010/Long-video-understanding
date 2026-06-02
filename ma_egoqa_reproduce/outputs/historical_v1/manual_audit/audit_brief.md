# Historical Memory Manual Audit Brief v1

说明：以下 10 个 case 是按文本启发式排序出的人工优先审查对象。这里的 Potential final label 只是待检查方向，不是自动标签。

Case ID: HISTV1_025
Question: What misunderstanding did Alice have about the beverages at first?
Answer: She thought the drinks included sodas but later realized it was only water.
Provisional query_user: Alice (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 theory_of_mind 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 有少量词面相关信号 | history 检索信号看起来强于 current，适合检查历史记忆是否补证据 | self_history top caption 看起来相关 | external_history top caption 看起来相关 | self_current 可能不足，self_history 可能补充 | external_current 可能不足，external_history 可能补充
Top self current: Alice D3 11:39:30.00-11:40:00.00 30sec rank=1 score=4.8186 | Alright, let me detail what I'm doing in this video. We were getting ready to go out. I heard Lucia saying, "Yeah, it's really urgent," and Alice responding, "But it shouldn't be very urgent." I then hear Alice saying, "You went up, treasure." Katrina said, "Wow, he actually put it on." Briefly,...
Top self history: Alice D2 18:24:30.00-18:25:00.00 30sec rank=1 score=10.9321 | I start by observing a jar of seasoning on the table, while Tasha, who is looking at a tablet and occasionally glancing up, asks, "Do you have coconut water too?" I then glance towards the kitchen where Shure is. Alice asks "Is there anything above?" and Shure responds, "It's also on top." Shif...
Top external current: Lucia D3 11:40:00.00-11:40:30.00 30sec rank=1 score=7.1465 | The video starts with me taking a step forward after hearing Shure say, "If I knew, I wouldn't have had coffee for this activity." Shure then says, "Wipe my tongue first." I then comb my hair, and Katrina says, "Wipe your tongue." I turn my head to look at Shure, who is walking back and forth. J...
Top external history: Shure D1 17:34:00.00-17:34:30.00 30sec rank=1 score=12.3060 | I am navigating through the supermarket aisle, starting in front of a shelf stocked with various beverages, including soda water. Initially, I reach out and grab approximately a dozen bottles of soda water off the shelf while Jake and Shure were having a conversation about Soda water. As I proc...
What human should check: 确认 Alice 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：重点检查是否为 self_history_needed

---

Case ID: HISTV1_028
Question: Why did Jake joke about putting all the puzzle pieces in randomly while everyone else took the task seriously?
Answer: Because Jake wanted to test how mismatched the pieces would look.
Provisional query_user: Jake (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 theory_of_mind 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：jake, joke, pieces, putting, puzzle | history 检索信号看起来强于 current，适合检查历史记忆是否补证据 | self_history top caption 看起来相关 | external_history top caption 看起来相关
Top self current: Jake D1 12:29:30.00-12:30:00.00 30sec rank=1 score=7.6219 | the video starts with me joining others around a table with a red and white checkered tablecloth as we attempt to assemble a puzzle. I initially pick up the puzzle box, then place it down. I repeat the action of picking up the box and putting it down several times. We are all looking at and handl...
Top self history: Jake D1 12:29:00.00-12:29:30.00 30sec rank=1 score=17.0790 | I am here, looking for puzzle pieces with everyone. Tasha says, "Hmm." I'm still searching for the right piece of the puzzle. Katrina mentions, "It doesn't fit, just a tiny bit off." I am sitting down in my place. I remain standing. I am working on putting together the puzzle. Jake asks, "Hmm, w...
Top external current: Shure D1 12:34:30.00-12:35:00.00 30sec rank=1 score=12.6140 | I observe Alice making observations about the assembly of an object, stating, "It's not a perfect fit. Some are just approximate. Hmm, this time it's quite different." Throughout this time, the group is seriously focused on the assembly task at hand. I quietly watch the others, and Jake suggest...
Top external history: Tasha D1 12:29:30.00-12:30:00.00 30sec rank=1 score=18.4528 | I am sitting at a table with others, attempting to piece together a wooden puzzle. Initially, I place a piece on the plate, then voice my concern that it seems inappropriate, prompting Jake to joke that I'm clueless. I remove the pieces, swapping them for another piece. Jake then suggests that...
What human should check: 确认 Jake 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：可能是 self_current_sufficient，也可能需要历史校验

---

Case ID: HISTV1_011
Question: What did Tasha propose when organizing the gift exchange task?
Answer: To taste the drinks after sorting the cups.
Provisional query_user: Tasha (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 task_coordination 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：exchange, gift, organizing, tasha | self_history top caption 看起来相关 | external_history top caption 看起来相关 | self_current 可能不足，self_history 可能补充
Top self current: Tasha D3 11:40:00.00-11:50:00.00 10min rank=1 score=0.9041 | I'm sitting at a table with Jake, Lucia, Shure, Alice, Tasha and Katrina, engaged in a water-tasting challenge, attempting to distinguish between different brands of mineral water. I start by tasting a glass, identifying it as mineral water. Frustrated by the subtle differences, I sigh, declarin...
Top self history: Tasha D1 19:27:00.00-19:27:30.00 30sec rank=1 score=13.0645 | Sitting on the bed, I begin by expressing how I feel full and hungry quickly, considering eating less, but more frequently as a solution. I am browsing my phone while stating this. Lucia then asks, "What is this?". Focusing on my phone, I start typing into an input box as Alice says "Sand ball....
Top external current: Alice D3 11:30:00.00-11:40:00.00 10min rank=1 score=9.7828 | it's Day 3, and I was mostly just chilling in the living room, completely engrossed in my phone, browsing Xiaohongshu. I started by looking at WeChat but quickly moved to Xiaohongshu and spent what felt like forever scrolling through various posts, comments, and even a tutorial on making a thous...
Top external history: Jake D2 13:48:30.00-13:49:00.00 30sec rank=1 score=17.1595 | The video begins with me overhearing Alice mentioning someone brought something, followed by Katrina stating that there isn't a gift exchange segment planned. I then observed my colleagues communicating with Katrina about potentially having a gift exchange. I chimed in with "We do, uh... Tonight...
What human should check: 确认 Tasha 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：重点检查是否为 self_history_needed

---

Case ID: HISTV1_027
Question: Why did Jake try to direct someone to the 'first floor Huawei' during their shopping trip?
Answer: Because that was where the group had agreed to regroup.
Provisional query_user: Jake (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 theory_of_mind 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：huawei, jake, shopping | self_history top caption 看起来相关 | external_history top caption 看起来相关
Top self current: Jake D5 15:58:30.00-15:59:00.00 30sec rank=1 score=9.6174 | Alright, I'm starting off in what appears to be a supermarket aisle, and I'm talking to Tasha on the phone, having finally gotten through to her. The conversation continues, and then I transition to calling Nicous. I ask where he is, and someone named Lucia interjects, and Tasha also questions he...
Top self history: Jake D5 15:53:00.00-15:53:30.00 30sec rank=1 score=17.0272 | The video starts as I'm walking through what appears to be a mall area. I notice a small shop with "Huawei" written on it, and I'm surprised to see it's a clothing store. Jake makes comments about how mysterious and rural the place is. Lucia and Tasha are with me. I point out the Huawei sign to...
Top external current: Tasha D5 15:56:00.00-15:56:30.00 30sec rank=1 score=11.8987 | I began by walking behind Lucia within what appears to be a large, warehouse-style supermarket. I then moved forward, subsequently picking up a bag. I overheard Lucia remarking that the second floor of the supermarket resembled a warehouse. I then proceeded to activate my phone, at which point...
Top external history: Lucia D5 15:53:00.00-15:53:30.00 30sec rank=1 score=21.3551 | The video begins with me turning my head to observe a store, identifying it as "Huawei". I take a closer look, noting the "Huawei" sign despite it selling general merchandise. We continue walking, during which Tasha points forward and gestures with her hand, then informs us that the shopping ca...
What human should check: 确认 Jake 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：可能是 self_current_sufficient，也可能需要历史校验

---

Case ID: HISTV1_030
Question: Why did Jake ask if preserving ice cubes could be eaten?
Answer: Because he was genuinely confused about their purpose.
Provisional query_user: Jake (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 theory_of_mind 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：ask, cubes, ice, preserving | self_history top caption 看起来相关 | external_history top caption 看起来相关
Top self current: Jake D1 18:19:30.00-18:20:00.00 30sec rank=1 score=6.3588 | I am standing in the checkout line at what appears to be a grocery store, as indicated by the "Check Out" sign above the registers. I say, "Yes, yes. I'll take this in and print it out for you," while facing the checkout counter. I then add, "Haha, I'm taking this gear into the cinema. Feels like...
Top self history: Jake D1 17:33:30.00-17:34:00.00 30sec rank=1 score=14.4737 | The video begins with me pushing a shopping cart through a grocery store, my hands visible on the handle. We're in an aisle with refrigerated displays lining the left side and a stack of eggs on the right. Alice mentions that "that one can be eaten raw," seemingly referring to a type of egg. Tas...
Top external current: Tasha D1 18:19:30.00-18:20:00.00 30sec rank=1 score=16.0273 | The video begins with me standing at the checkout counter, interacting with others as we unload groceries from our shopping carts. I make a waving gesture, stating, "Yeah, yeah, it is. I'll bring this in and make a copy for you." Alice then requests, "Play something lighter," and in response, I...
Top external history: Lucia D1 11:54:30.00-11:55:00.00 30sec rank=1 score=21.6062 | I am sharing my cooking experience with the group. I comment on the texture of the meat, stating, "If the meat is too soft, I can't cut it well," and clarify, "Like if it's already thawed." Shure replies, "Okay, got it." I then adjust my glasses. Jake initiates a discussion about grilling, aski...
What human should check: 确认 Jake 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：可能是 self_current_sufficient，也可能需要历史校验

---

Case ID: HISTV1_026
Question: Why was Katrina initially confused about what was happening with the ball during the game?
Answer: She assumed it would require manual intervention to release the ball.
Provisional query_user: Katrina (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 theory_of_mind 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 有少量词面相关信号 | self_history top caption 看起来相关 | external_history top caption 看起来相关 | self_current 可能不足，self_history 可能补充
Top self current: Katrina D3 16:30:00.00-16:40:00.00 10min rank=1 score=12.9874 | I headed out to the sale store because I needed to get face towels and sanitary napkins. A couple of people said they'd come along. Once we got there, at a store called "HotMax," the group debated what kind of sanitary napkins to get and if there were any special deals. I was also hoping to f...
Top self history: Katrina D3 16:00:00.00-16:10:00.00 10min rank=1 score=15.9421 | I started off by walking through a brightly lit arcade, marveling at all the games. I was with Katrina, Violet, Shure, and Jake, and we were on the hunt for something fun to play. At first, I was intrigued by the claw machines, even asking Katrina about the cost and wondering if winning was e...
Top external current: Lucia D3 16:20:00.00-16:30:00.00 10min rank=1 score=11.7728 | I started off completely absorbed in a round of Initial D Zero, fiercely gripping the steering wheel and making constant adjustments to stay on the virtual track, determined to win. I played hard, but after the race, the game ended, and I took a moment to adjust my glasses. Afterwards, there wa...
Top external history: Tasha D3 16:17:30.00-16:18:00.00 30sec rank=1 score=19.6498 | Okay, you go ahead. I then saw that the bowling missed by a hair. Jake then said, "But I might... oh, if it hits both sides, it should go down." I looked at Jake as he repeated, "If it hits both sides, it should go down." Then he was gesturing with his hands to emphasize. I continued to watch J...
What human should check: 确认 Katrina 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：重点检查是否为 self_history_needed

---

Case ID: HISTV1_019
Question: When Katrina helped finalize the list on the whiteboard, who seemed most actively giving her suggestions?
Answer: Lucia and Tasha
Provisional query_user: Katrina (inferred_weak)
Why this case may be useful: query_user 是 inferred_weak，适合优先人工确认 self-first 起点 | 类别 social_interaction 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：giving, katrina, whiteboard, who | self_history top caption 看起来相关 | external_history top caption 看起来相关 | self_current 可能不足，self_history 可能补充
Top self current: (empty)
Top self history: Katrina D1 13:37:00.00-13:37:30.00 30sec rank=1 score=13.0078 | I am actively recording, capturing a conversation where Jake jokes that someone, presumably Lucia, didn't remember something, to which Lucia replies that she only heard KFC initially. I continue recording, Lucia refutes her earlier statement, claiming she just remembered. I observe JakeShure...
Top external current: Tasha D1 13:55:00.00-13:55:30.00 30sec rank=1 score=10.4183 | I am holding a tablet and watching Katrina as she writes on a whiteboard. Initially, Jake and Lucia discuss the presence of lures, determining there shouldn't be any but only utensils, pots, and pans. Subsequently, Tasha lists items, including milk, light cream, eggs, starch, salt, saline, suga...
Top external history: Jake D1 13:10:00.00-13:20:00.00 10min rank=1 score=14.1266 | Initially, I was operating the rack, tightening screws and trying to get everything secured while others, including Lucia, Jake, and Tasha, were giving comments and opinions about the equipment. It seemed like there was some difficulty getting a particular component to fit, and at one point, I e...
What human should check: 确认 Katrina 是否真的是 query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：重点检查是否为 self_history_needed

---

Case ID: HISTV1_013
Question: Which of the following happened first? A) Lucia oversaw reimbursements and dismantled a cabinet before departure planning B) Alice coordinated takeout ordering and kitchen prep after organizing groceries, C) Shure managed recording setups while the group planned meals and budgets, D) Katrina led themed room planning and wrapped up the session, E) Jake finalized plans for the evening BBQ and post-dinner activities,
Answer: E
Provisional query_user: UNKNOWN (unknown)
Why this case may be useful: query_user 仍是 UNKNOWN，需要先判断是否适合 self-first | 类别 temporal_reasoning 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：alice, and, dismantled, for, jake, led | history 检索信号看起来强于 current，适合检查历史记忆是否补证据
Top self current: (empty)
Top self history: (empty)
Top external current: (empty)
Top external history: (empty)
What human should check: 先判断是否能从问题/上下文确认 human_query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：先确认 query_user；若无法确认，可能是 not_self_first 或 reject_unclear

---

Case ID: HISTV1_015
Question: Which of the following happened last? A) Tasha led evening cleanup and planned next day's activities, B) Lucia curated dance videos and finished sketching a cat for art, C) Katrina rehearsed music with Shure and mentored Nicous on guitar D) Alice finalized event roles and schedules for the evening program, E) Jake wrapped picnic setup and concluded filming of the gathering,
Answer: C
Provisional query_user: UNKNOWN (unknown)
Why this case may be useful: query_user 仍是 UNKNOWN，需要先判断是否适合 self-first | 类别 temporal_reasoning 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：alice, and, for, jake, last, led | history 检索信号看起来强于 current，适合检查历史记忆是否补证据
Top self current: (empty)
Top self history: (empty)
Top external current: (empty)
Top external history: (empty)
What human should check: 先判断是否能从问题/上下文确认 human_query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：先确认 query_user；若无法确认，可能是 not_self_first 或 reject_unclear

---

Case ID: HISTV1_017
Question: When Jake was walking with Nicous and Tasha carrying a package behind the houses, and Alice was listening to sad love songs and finishing a star, what was Shure doing?
Answer: Cleaning up the trash and packing it into a box while chatting with Katrina.
Provisional query_user: UNKNOWN (unknown)
Why this case may be useful: query_user 仍是 UNKNOWN，需要先判断是否适合 self-first | 类别 temporal_reasoning 是本轮优先审查类型 | 问题中有明确人物名 | history top caption 与问题有词面重合：alice, and, doing, jake, love, nicous | history 检索信号看起来强于 current，适合检查历史记忆是否补证据
Top self current: (empty)
Top self history: (empty)
Top external current: (empty)
Top external history: (empty)
What human should check: 先判断是否能从问题/上下文确认 human_query_user；比较 self/external 的 current 与 history caption 是否能支持答案所需证据；不要把 BM25 分数或排序当作最终标签
Potential final label: 人工待定：先确认 query_user；若无法确认，可能是 not_self_first 或 reject_unclear

---
