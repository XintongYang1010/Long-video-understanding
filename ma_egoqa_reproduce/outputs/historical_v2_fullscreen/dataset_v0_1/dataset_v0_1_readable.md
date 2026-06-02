# Dataset V0.1 Demo Cases

These are caption-only, auto-screened candidate cases for evidence-scope comparison. They require human spot-check before use as labels.

## DV01_DEMO_001

Question: Why did Shure laugh when Katrina mentioned her voice messages?

Answer: Shure remembered he had similar embarrassing experiences.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=12.0918 source=Shure D7 11:41:30.00-11:42:00.00 30sec] I begin by organizing a wire in my hand, then I shake my head as I address Katrina, asking, "What did you do last night?". I continue to observe Katrina as he says, "I don't know, oh my, what did I say last night?". I am watching him and then laugh. I ask, "Hahaha, did you send those voice messages?", I laugh again happily as Katrina takes a few steps. I continue chatting with him and say, "And now you don't dare to open them, right? Hahahahahaha, don't laugh". I smile again, glancing at the object in my hand, and proceed to place my glasses and power bank on the sofa, still observing Katrina. I say, "Don't la... [current_only rank=2 score=7.1784 source=Shure D7 11:42:00.00-11:42:30.00 30sec] I am in what appears to be an apartment or a home, and I begin speaking, gesturing with my right hand, saying, "Katrina: It's really too..." I then turn around sli...

Historical evidence: [history_only rank=1 score=19.1031 source=Katrina D4 16:43:30.00-16:44:00.00 30sec] I'm sitting at a table in what appears to be a living room and kitchen area. Shure is standing and chatting with me, sharing about his freshman and sophomore years, specifically mentioning that he "couldn't control my drinking." He continues to discuss times when he was "drinking with my roommates" and would "get too drunk." I respond with "Hmm" as I listen. He elaborates that he would "Just send a bunch of weird voice messages," continuing to explain his experiences. He walks around the living room as he speaks. Gesturing with his left hand, he says, "The next day, I wouldn't want to meet again," but then add... [history_only rank=2 score=17.2833 source=Shure D5 18:41:00.00-18:41:30.00 30sec] I began by turning around, then adjusting my glasses. I then walked forward towards a wooden gate, during which I sang, "Shure: Sakira, how should I describe yo...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=12.0918 source=Shure D7 11:41:30.00-11:42:00.00 30sec] I begin by organizing a wire in my hand, then I shake my head as I address Katrina, asking, "What did you do last night?". I continue to observe Katrina as he says, "I don't know, oh my, what did I say last night?". I am watching him and then laugh. I ask, "Hahaha, did you send those voice messages?", I laugh again happily as Katrina takes a few steps. I continue chatting with him and say, "And now you don't dare to open them, right? Hahahahahaha, don't laugh". I smile again, glancing at the object in my hand, and proceed to place my glasses and power bank on the sofa, still observing Katrina. I say, "Don't la... [current_only rank=2 score=7.1784 source=Shure D7 11:42:00.00-11:42:30.00 30sec] I am in what appears to be an apartment or a home, and I begin speaking, gesturing with my right hand, saying, "Katrina: It's really too..." I then turn around slightly, fiddling with a stand that I'm holding a bracket for. I tinker with the bracket in my hands. "Katrina: Have you guys had breakfast?" I ask, while still adjusting the bracket. Then I hear someone saying, "Shure: Didn't we ask...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_002

Question: What decision-making role did Jake take in the group task involving glasses?

Answer: Coordinator who managed roles and responsibilities.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=4.8813 source=Katrina D5 12:50:00.00-13:00:00.00 10min] I was settled on a couch, working on a PowerPoint presentation on my laptop, when the usual chaos of the living room started swirling around me. Initially, I was trying to work but had trouble with an online search, even needing to adjust my settings, and was getting snippets of a lively conversation between Jake, Katrina, Alice, Lucia, and Tasha. Jake, in particular, was on a roll, cracking jokes about fermented mung bean milk, the use of idiot data for AI training, digestive tablets, and the implications of a pregnant woman hitting someone, which lead to my own search about that very topic. There was a lot o... [current_only rank=2 score=4.6916 source=Jake D5 13:00:00.00-13:10:00.00 10min] starting at 1:00 PM, I found myself in a room and picked up a pair of glasses that were tangled with a black cord. As I walked out of the room, I overheard someone...

Historical evidence: [history_only rank=1 score=17.1330 source=Katrina D2 12:05:30.00-12:06:00.00 30sec] I initiate a turn, and then proceed to descend a set of stairs while holding a sheet of stickers. Throughout my descent, I overhear a conversation where Tasha describes someone as the "core strategist of the team," elaborating on their commanding and operational management roles. Alice then inquires about the person's appearance, to which Tasha responds that "this is their AD," further explaining that they are currently selecting for the AD role. Upon reaching the bottom of the stairs, I enter a room where others are gathered and push my glasses up the bridge of my nose. The discussion continues, with Tasha re... [history_only rank=2 score=14.1909 source=Jake D3 15:23:30.00-15:24:00.00 30sec] The video starts as I ascend a set of stairs outdoors, accompanied by a group of people, when Lucia exclaims about the heat. As we reach the top of the stairs, w...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=4.8813 source=Katrina D5 12:50:00.00-13:00:00.00 10min] I was settled on a couch, working on a PowerPoint presentation on my laptop, when the usual chaos of the living room started swirling around me. Initially, I was trying to work but had trouble with an online search, even needing to adjust my settings, and was getting snippets of a lively conversation between Jake, Katrina, Alice, Lucia, and Tasha. Jake, in particular, was on a roll, cracking jokes about fermented mung bean milk, the use of idiot data for AI training, digestive tablets, and the implications of a pregnant woman hitting someone, which lead to my own search about that very topic. There was a lot o... [current_only rank=2 score=4.6916 source=Jake D5 13:00:00.00-13:10:00.00 10min] starting at 1:00 PM, I found myself in a room and picked up a pair of glasses that were tangled with a black cord. As I walked out of the room, I overheard someone, who I think was Jake, mention the number "1,500." I headed straight for the stairwell and started descending the stairs. I remember hearing people making some noise as I walked down. At the bottom, I turned into the living room on...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_003

Question: When Katrina discussed her breakup, how did the group mostly respond?

Answer: They offered different speculations about her ex’s intentions.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=8.2742 source=Tasha D6 21:09:30.00-21:10:00.00 30sec] The video begins with my point of view as I stand in a room with several friends. Katrina is speaking, questioning why someone always wants to take pictures of her and another friend, even though they often take couple photos, and wonders why he insists on taking pictures with them. I am mostly silent, observing the conversation and listening attentively to Katrina's remarks. Nicous makes a comment about a good student. Then, Shure suggests opening champagne, and Jake agrees. Nicous mentions the invitation of someone, leading to a discussion about relationships and friendships. I remain mostly observant. Katri... [current_only rank=2 score=7.2090 source=Tasha D6 21:00:00.00-21:10:00.00 10min] I was really trying to buckle down and get this homework done at my workspace. I started by searching through files on my computer, checking my iPad occasionally, a...

Historical evidence: [history_only rank=1 score=24.9177 source=Shure D2 22:32:00.00-22:32:30.00 30sec] The video begins with me sitting with a group of people as Katrina shares a story about her breakup and how her ex-boyfriend attempted to make up for it but got the wrong makeup product, specifically the wrong shade. Choiszt jokingly commented about getting the wrong shade, and Katrina clarified it was shade 82, adding that there are two series of shade 82, one a lipstick and the other a lip gloss. Nicous is surprised at the mistake, pointing out how different lipstick and lip gloss are. I glanced down at my phone while Katrina describes her ex-boyfriend as unreliable and inattentive. I then stood up, taking o... [history_only rank=2 score=22.2438 source=Shure D2 22:20:00.00-22:20:30.00 30sec] In this egocentric video, I am engaged in a conversation with Katrina. Initially, I am holding a cigarette in my left hand while affirming, "Shure: That's true."...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=8.2742 source=Tasha D6 21:09:30.00-21:10:00.00 30sec] The video begins with my point of view as I stand in a room with several friends. Katrina is speaking, questioning why someone always wants to take pictures of her and another friend, even though they often take couple photos, and wonders why he insists on taking pictures with them. I am mostly silent, observing the conversation and listening attentively to Katrina's remarks. Nicous makes a comment about a good student. Then, Shure suggests opening champagne, and Jake agrees. Nicous mentions the invitation of someone, leading to a discussion about relationships and friendships. I remain mostly observant. Katri... [current_only rank=2 score=7.2090 source=Tasha D6 21:00:00.00-21:10:00.00 10min] I was really trying to buckle down and get this homework done at my workspace. I started by searching through files on my computer, checking my iPad occasionally, and even had my gloves nearby. I was copying information from one document, closing it, and then opening a different program. There was a loading screen for a bit, and then I was back to typing, really trying to improve the document....

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_004

Question: Who incorrectly assumed everyone would want an auction group of four items?

Answer: Others, in a majority vote during the debate

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=9.0135 source=Shure D6 16:49:00.00-16:49:30.00 30sec] The video opens with a wide view of the room where an auction is happening. Everyone seems to be in good spirits and smiling. I gesture with my right hand and say "I have four." Someone responds, "Ah, he just wants to think about four, right?" I then continue gesturing with my right hand and say, "So, four at a time, twice." I watch as the auction participants engage. Alice laughs, and someone says, "Damn, no one's bidding anymore." Tasha laughs loudly. I tap the table and say, "Soul four times, three times," gesturing again with my right hand. Katrina seems very happy and exclaims, "I'm dying of laughter, tha... [current_only rank=2 score=7.7997 source=Shure D6 16:46:30.00-16:47:00.00 30sec] the video begins as I'm in the middle of an event, specifically an auction. I say, "One postcard and...", implying I was discussing the item being auctioned. Then I...

Historical evidence: [history_only rank=1 score=19.3051 source=Tasha D1 12:05:30.00-12:06:00.00 30sec] I am participating in a meeting where the topic of discussion revolves around an auction. Katrina initiates the conversation by asking, "Or who's auctioning?" Shure responds, "Both ours and theirs," clarifying that the auction involves items from both our group and another party. Katrina follows up with, "The things they brought. Yes, and ours," emphasizing the inclusion of items brought by the external group and items from our own collection. Lucia adds, "And our products," specifying that our own products will also be part of the auction. Katrina then estimates, "It should take more than an hour then," sugge... [history_only rank=2 score=16.6036 source=Jake D2 11:57:30.00-11:58:00.00 30sec] I observe a conversation taking place in a room with four other individuals, Alice, Jake, Katrina, and Lucia, gathered around a table covered with a red and white...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=9.0135 source=Shure D6 16:49:00.00-16:49:30.00 30sec] The video opens with a wide view of the room where an auction is happening. Everyone seems to be in good spirits and smiling. I gesture with my right hand and say "I have four." Someone responds, "Ah, he just wants to think about four, right?" I then continue gesturing with my right hand and say, "So, four at a time, twice." I watch as the auction participants engage. Alice laughs, and someone says, "Damn, no one's bidding anymore." Tasha laughs loudly. I tap the table and say, "Soul four times, three times," gesturing again with my right hand. Katrina seems very happy and exclaims, "I'm dying of laughter, tha... [current_only rank=2 score=7.7997 source=Shure D6 16:46:30.00-16:47:00.00 30sec] the video begins as I'm in the middle of an event, specifically an auction. I say, "One postcard and...", implying I was discussing the item being auctioned. Then I hosted the event. I stood up, addressing the people and then said, "Katrina: One bookmark." After Katrina's response, I repeat, "One bookmark." I continued to fidget with the items on the table and I pick them up. Then I commented...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_005

Question: Why didn't Katrina seem fully involved in the conversation about renting costumes during 18:23?

Answer: Katrina expressed doubts about everyone's participation being practical.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=14.1713 source=Lucia D4 18:23:30.00-18:24:00.00 30sec] the video begins with me meticulously trimming the bottom of a cardboard cutout using blue scissors. It appears I am attempting to perfect its shape. Suddenly, Alice places a cardboard dog cutout near the table where I am working. I then hear Katrina initiate a conversation, stating, "Mainly because we haven't rented the costumes yet, right?" Tasha responds, "Hmm, good thing you haven't left yet." I declare my intention to refine the dog's eyes. Katrina replies, "It's alright, we made the decision because we couldn't rent." Then I correctly place the circles on the dog's cardboard face. Tasha then interjects,... [current_only rank=2 score=13.6579 source=Tasha D4 18:23:30.00-18:24:00.00 30sec] The video begins with me watching Katrina as she walks around; I hear her say, "Mainly because we haven't rented the right clothes, right?" I reply with, "Well, it...

Historical evidence: [history_only rank=1 score=22.2542 source=Jake D4 13:20:00.00-13:30:00.00 10min] I was sitting around a table with Tasha, Lucia, Jake, Nicous, Katrina, and later Shure, enjoying a dumpling feast. Tasha was showing me options on her phone for renting some costumes, mentioning a top that was 40 yuan before correcting herself and finding a full set for 25 yuan, we also discussed a budget for renting items and I mentioned asking for 1000 was too much. There was a slight cola mishap when Lucia spilled her drink, which I quickly cleaned up with some toilet paper. Lucia offered me the cola she spilled, and I said I'd have some after I finished my coffee, prompting a bit of light-hearted banter wi... [history_only rank=2 score=21.3171 source=Tasha D4 13:20:00.00-13:30:00.00 10min] I was sitting at the dining table surrounded by friends. Our meal was in full swing with a variety of dishes, from dumplings to eggs, and Chinese toon, alongside o...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=14.1713 source=Lucia D4 18:23:30.00-18:24:00.00 30sec] the video begins with me meticulously trimming the bottom of a cardboard cutout using blue scissors. It appears I am attempting to perfect its shape. Suddenly, Alice places a cardboard dog cutout near the table where I am working. I then hear Katrina initiate a conversation, stating, "Mainly because we haven't rented the costumes yet, right?" Tasha responds, "Hmm, good thing you haven't left yet." I declare my intention to refine the dog's eyes. Katrina replies, "It's alright, we made the decision because we couldn't rent." Then I correctly place the circles on the dog's cardboard face. Tasha then interjects,... [current_only rank=2 score=13.6579 source=Tasha D4 18:23:30.00-18:24:00.00 30sec] The video begins with me watching Katrina as she walks around; I hear her say, "Mainly because we haven't rented the right clothes, right?" I reply with, "Well, it's good that it hasn't been rented yet." Katrina responds to me, stating, "It's okay, we made the decision because we didn't rent them." I then reach for my power bank and phone, subsequently unlocking my phone and navigating to the...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_006

Question: Why did Jake smile and respond to Nicous by referring to a 'national guardian'?

Answer: He believed Nicous was talking about a donation program.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=10.6819 source=Jake D6 22:16:00.00-22:16:30.00 30sec] The video begins with Nicous addressing me as "big brother," and I notice Shure and Nicous's girlfriends engaged in a conversation nearby. Nicous offers to light a cigarette for Shure, and I briefly glance at Shure before Nicous makes a comment referencing a Chinese aircraft carrier, eliciting laughter. Shure is engaged in a conversation with Nicous, and I observe Shure looking in my direction. I subsequently turn to my left, and I vocalize "What's an aircraft carrier?". I glanced at Nicous and then again at Shure. I chuckled. Shure responds with "No idea." I take two steps backward, and Nicous suggests that i... [current_only rank=2 score=5.9314 source=Jake D6 22:15:30.00-22:16:00.00 30sec] As the video begins, I am walking forward, overhearing a conversation between ShureNicous and Shure's girlfriend where Shure says "He doesn't smoke, he doesn't smoke...

Historical evidence: [history_only rank=1 score=15.5212 source=Jake D6 19:32:30.00-19:33:00.00 30sec] I am walking forward while listening to Lucia, who is commenting on a video, stating, "The last one, the last one is me," and "From that angle." I turn around and take a step back, and Lucia continues, "My face looks a bit chubby," followed by, "But the earlier ones are all good." I then glance to the right before looking at Nicous and Katrina. I smile while talking to Nicous. Jake says, "Haven't even started and I'm asked to clear the field," and then, "Don't forget those three songs, oh my gosh hahaha." Lucia responds with, "It's okay, it's okay, it's fine." I turn back and forth and then to the right. Katri... [history_only rank=2 score=15.4044 source=Jake D5 18:49:30.00-18:50:00.00 30sec] I'm standing outside on a deck area during what appears to be dusk, engaging in a conversation with Lucia, who is just off-camera, asking, "Is this it? Did he stick...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=10.6819 source=Jake D6 22:16:00.00-22:16:30.00 30sec] The video begins with Nicous addressing me as "big brother," and I notice Shure and Nicous's girlfriends engaged in a conversation nearby. Nicous offers to light a cigarette for Shure, and I briefly glance at Shure before Nicous makes a comment referencing a Chinese aircraft carrier, eliciting laughter. Shure is engaged in a conversation with Nicous, and I observe Shure looking in my direction. I subsequently turn to my left, and I vocalize "What's an aircraft carrier?". I glanced at Nicous and then again at Shure. I chuckled. Shure responds with "No idea." I take two steps backward, and Nicous suggests that i... [current_only rank=2 score=5.9314 source=Jake D6 22:15:30.00-22:16:00.00 30sec] As the video begins, I am walking forward, overhearing a conversation between ShureNicous and Shure's girlfriend where Shure says "He doesn't smoke, he doesn't smoke," and she responds with "Whatever, whatever, whatever," and then, "Anything is fine, anything is fine." Nicous then chimes in, saying, "Take it out, take it out," and Shure's girlfriend adds, "I'm okay with anything." I turn left,...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_007

Question: What supported the collaborative mood around the cake task?

Answer: Team celebrations and applause at key milestones.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=7.1615 source=Lucia D3 22:06:00.00-22:06:30.00 30sec] the video shows Tasha sprinkling chocolate powder onto the cake with a slotted spoon. Jake comments, "Adding the finishing touches" and then, "Impressive, impressive, okay." I, Lucia, look at Tasha as Tasha asks Jake, "What do you really want to say?" I laugh and say, "Haha, these two..." and continue chatting with them. I then say, "These two words have completely different meanings," followed by Jake saying "Ha." I advised, "You better think it through." I am in a happy mood, and I look at the cocoa powder. Katrina is using chopsticks to eat the meat on the plate. I continue to look at the cake while Katrina... [current_only rank=2 score=5.4429 source=Jake D3 22:10:00.00-22:10:30.00 30sec] Oh crap, another strawberry has fallen off the cake, and it looks like another is about to! Alice asks what I see, and I reply that the strawberries are collapsing o...

Historical evidence: [history_only rank=1 score=18.9686 source=Lucia D1 21:48:00.00-21:48:30.00 30sec] I ate the cake entirely. I hear Alice asking, "Can you open it from the outside?" and then stating, "Let me see if it can lock" and later "Hmm, it can't be locked, it's a bit broken." I observe Jake saying, "This probably won't work. I think we should give up." Tasha picks up strawberries that are on the ground. I throw a paper cup into the trash can. I hear Katrina say, "Look at the rainbow in my eyes." Lucia then says, "Haha, didn't you bake it?" and corrects herself by saying, "We all baked it together," following up with, "The key thing is your guidance and experience." Tasha uses a piece of paper to wipe... [history_only rank=2 score=16.1579 source=Katrina D1 12:29:00.00-12:29:30.00 30sec] The video begins with a collaborative effort in progress, as several individuals, including myself, are gathered around a table covered with a red and white chec...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=7.1615 source=Lucia D3 22:06:00.00-22:06:30.00 30sec] the video shows Tasha sprinkling chocolate powder onto the cake with a slotted spoon. Jake comments, "Adding the finishing touches" and then, "Impressive, impressive, okay." I, Lucia, look at Tasha as Tasha asks Jake, "What do you really want to say?" I laugh and say, "Haha, these two..." and continue chatting with them. I then say, "These two words have completely different meanings," followed by Jake saying "Ha." I advised, "You better think it through." I am in a happy mood, and I look at the cocoa powder. Katrina is using chopsticks to eat the meat on the plate. I continue to look at the cake while Katrina... [current_only rank=2 score=5.4429 source=Jake D3 22:10:00.00-22:10:30.00 30sec] Oh crap, another strawberry has fallen off the cake, and it looks like another is about to! Alice asks what I see, and I reply that the strawberries are collapsing off the cake like skyscrapers. Lucia tells me to be careful as I try to support the cake, and I put a fallen strawberry aside. I exclaim "Emergency repair!" to which Jake laughs. I laugh as well, repeating "Emergency repair!" and Luc...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_008

Question: How did Jake contribute to the group's tasks during the time Shure prepared coffee?

Answer: Carrying and rearranging tables alongside Nicous

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=5.9963 source=Alice D2 15:30:00.00-15:40:00.00 10min] I started my day by admiring the flowers on the table before Jake showed me something. Then, I checked the timer on my phone, looked around, and caught my reflection in the mirror, taking a moment to appreciate how I looked. After that, I walked slowly towards the kitchen, noticing Katrina practicing a dance routine on her tablet. I grabbed a bottle of mineral water from the shelf and took it to the table, watching Katrina practice and commenting on the dance's difficulty. We all gathered around the tablet, watching the dance video together. After finishing my water, I realized we had other things to do and me... [current_only rank=2 score=5.5737 source=Katrina D2 15:26:00.00-15:26:30.00 30sec] Okay, so the video begins with me dancing in what appears to be a living room, while Alice observes me. I continue dancing. Subsequently, I approach and push hand...

Historical evidence: [history_only rank=1 score=14.5590 source=Jake D2 13:55:00.00-13:55:30.00 30sec] The video begins with me glancing in the direction of Nicous while hearing comments about latte art and a coffee machine's supposed faults. I then turn to see Nicous adjusting my action camera. Shure asks if they can still do latte art at home, and I pause to consider the question as Nicous continues working. Other voices continue discussing the coffee machine and latte art. Turning towards Shure, I hear comments about the cost of coffee shop machines. I then walk towards the kitchen, open the curtains, and make an adjustment to them. As someone else pulls the curtains further, I glance at the action camera, b... [history_only rank=2 score=12.9796 source=Katrina D2 13:53:00.00-13:53:30.00 30sec] Okay, let me describe what I saw and did in the video. I hear Katrina speaking about a past interview and procedures as I walk towards Shure. I raise my hand dur...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=5.9963 source=Alice D2 15:30:00.00-15:40:00.00 10min] I started my day by admiring the flowers on the table before Jake showed me something. Then, I checked the timer on my phone, looked around, and caught my reflection in the mirror, taking a moment to appreciate how I looked. After that, I walked slowly towards the kitchen, noticing Katrina practicing a dance routine on her tablet. I grabbed a bottle of mineral water from the shelf and took it to the table, watching Katrina practice and commenting on the dance's difficulty. We all gathered around the tablet, watching the dance video together. After finishing my water, I realized we had other things to do and me... [current_only rank=2 score=5.5737 source=Katrina D2 15:26:00.00-15:26:30.00 30sec] Okay, so the video begins with me dancing in what appears to be a living room, while Alice observes me. I continue dancing. Subsequently, I approach and push hands, presumably with Jake, seeking permission to make coffee, saying "Shure: OK, so can I go make coffee now, Jake?" Jake replies "Jake: You can make coffee now" and then confirms "Jake: Go ahead." I respond with "Shure: Okay" and Cho...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_009

Question: Why did they decide on having transition shots in the choreography?

Answer: To ensure smooth flow between different members’ solos

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=6.8851 source=Katrina D2 13:43:30.00-13:44:00.00 30sec] I initially overheard Jake mentioning "trainees," and then I immediately averted my gaze to my phone, holding it horizontally. I then overhear Jake complimenting someone's hair and suggesting they do a solo dance. I glanced up briefly at Nicous before returning my attention to my phone screen, while Alice is also occupied with her own phone. After Katrina told everyone to start dancing, I momentarily looked up. I heard Tasha's exclamation of "Oh, oh, oh," and saw Nicous getting up. I then noticed Lucia waving her hand and engaging in a discussion with Tasha about potentially inserting a solo section for someon... [current_only rank=2 score=6.1581 source=Jake D2 13:43:30.00-13:44:00.00 30sec] The egocentric video begins at 13:43:30 on DAY2. I am at a table with several colleagues. The table is covered with containers of food, drinks, and other items. Th...

Historical evidence: [history_only rank=1 score=15.7940 source=Katrina D2 11:40:00.00-11:50:00.00 10min] Sitting on the floor, I started watching videos on my tablet, a mix of dance performances and random content. I remember questioning aloud about a "solo event" at one point. There was a lot of back-and-forth between full-screen and the video selection screen as I browsed. While watching, I overheard conversations between Tasha, Alice, and Lucia, mostly about specialist appointment costs and a power bank issue, with Tasha mentioning something about "500 yuan per person". Later, the topic shifted to concerts and tickets, with Alice lamenting her failure to get any, even with help. I even jokingly suggested Jake... [history_only rank=2 score=10.8292 source=Katrina D1 17:30:00.00-17:30:30.00 30sec] Alright, in this video, I'm navigating a supermarket. I begin by walking straight down an aisle while thinking to myself, "As for the grape juice, should I jus...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=6.8851 source=Katrina D2 13:43:30.00-13:44:00.00 30sec] I initially overheard Jake mentioning "trainees," and then I immediately averted my gaze to my phone, holding it horizontally. I then overhear Jake complimenting someone's hair and suggesting they do a solo dance. I glanced up briefly at Nicous before returning my attention to my phone screen, while Alice is also occupied with her own phone. After Katrina told everyone to start dancing, I momentarily looked up. I heard Tasha's exclamation of "Oh, oh, oh," and saw Nicous getting up. I then noticed Lucia waving her hand and engaging in a discussion with Tasha about potentially inserting a solo section for someon... [current_only rank=2 score=6.1581 source=Jake D2 13:43:30.00-13:44:00.00 30sec] The egocentric video begins at 13:43:30 on DAY2. I am at a table with several colleagues. The table is covered with containers of food, drinks, and other items. The scene starts with me hearing Jake commenting, "A trainee, a trainee," and then complimenting someone’s hair, stating, "Your hair looks pretty good." I reply in the communication; others agree with, "No problem." Tasha then exclaim...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.

## DV01_DEMO_010

Question: How was coordination handled in dividing cleanup responsibilities?

Answer: Jake contacted the housekeeper to address the issue.

Why selected: Tier A case with readable current+historical evidence gain over weaker current-only evidence.

Current-only evidence: [current_only rank=1 score=6.5164 source=Lucia D2 10:40:00.00-10:50:00.00 10min] I'm sitting at the table with everyone, listening to their conversations as I struggle to use the Meituan app on my phone. Jake's showing someone how to use the cameras, and Alice is discussing the consistency of egg pancakes, while I try to find a specific address – Lucia: 50, No. 53. It's not going well, and my network keeps cutting out. I even scraped my foot and need to buy iodine solution, so I sit down to order it, but the app is just not cooperating. I hear them talking about cleaning services and whether the place was cleaned before we arrived, and I chime in to ask if we can request someone to come cl... [current_only rank=2 score=5.4395 source=Lucia D2 10:47:30.00-10:48:00.00 30sec] As the video begins, I'm seated at a table with a checkered tablecloth, saying "This village feels..." while simultaneously opening the Meituan app on my phone. I l...

Historical evidence: [history_only rank=1 score=11.8798 source=Lucia D1 13:01:30.00-13:02:00.00 30sec] The video begins with me noticing that the software interface has unexpectedly changed to "network Arrow" after unplugging it, causing me to feel puzzled. I then decided to seek Jake's assistance, prompting me to grab my phone to show him the issue. I demonstrate the incorrect interface on my phone screen, replicating the unexpected change. Shure interjects, advising me to deal with certain characters or elements. I express my confusion about their meaning, after that Jake examines my phone, sliding the screen to further analyze the problem. Following this, Jake suggests restarting the device as a solution. De... [history_only rank=2 score=11.0978 source=Tasha D1 18:13:30.00-18:14:00.00 30sec] Okay, everyone is waiting for me to pay the bill, so I'm at the cashier and Shure is asking if Alipay works and Jake confirmed that it does, and Shure mentioned t...

Current+historical evidence: CURRENT EVIDENCE: [current_only rank=1 score=6.5164 source=Lucia D2 10:40:00.00-10:50:00.00 10min] I'm sitting at the table with everyone, listening to their conversations as I struggle to use the Meituan app on my phone. Jake's showing someone how to use the cameras, and Alice is discussing the consistency of egg pancakes, while I try to find a specific address – Lucia: 50, No. 53. It's not going well, and my network keeps cutting out. I even scraped my foot and need to buy iodine solution, so I sit down to order it, but the app is just not cooperating. I hear them talking about cleaning services and whether the place was cleaned before we arrived, and I chime in to ask if we can request someone to come cl... [current_only rank=2 score=5.4395 source=Lucia D2 10:47:30.00-10:48:00.00 30sec] As the video begins, I'm seated at a table with a checkered tablecloth, saying "This village feels..." while simultaneously opening the Meituan app on my phone. I look up towards Jake, who's standing nearby as Lucia asks if the Meituan issue is related to her VPN. Katrina then mentions something about "outsiders". Jake then asks if the app is stuck, and Lucia offers to take a look and reassure...

Expected result: current_plus_history_better

Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.

Potential issue: Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required.
