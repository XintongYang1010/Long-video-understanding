# True A+B Candidate Search Summary Round 2

This is a low-bandwidth visual evidence generation step. It does not run VLM/LLM/CLIP, does not run official CASTLE QA, and does not download full videos.

- number of candidate windows selected: 10
- number of windows with generated visual evidence: 10
- number of contact sheets generated: 20
- frame extraction success: 519/519
- number of candidate cases listed: 10
- number of possible true_complementary_A_plus_B cases: all generated cases remain needs_human_review; 0 locked true cases

## Strongest 3 candidate cases
- R2_001 DAY1_122000000: What is the person referring to, and how is it connected to the group or screen context? (weak_or_unclear)
- R2_002 DAY1_124000000: Who handled the object, and where did the object end up? (weak_or_unclear)
- R2_003 DAY2_121500000: What local action happened, and where in the room did it happen? (weak_or_unclear)

## Weak/unclear cases
- R2_001 DAY1_122000000: needs_human_review=yes
- R2_002 DAY1_124000000: needs_human_review=yes
- R2_003 DAY2_121500000: needs_human_review=yes
- R2_004 DAY2_142000000: needs_human_review=yes
- R2_005 DAY2_164000000: needs_human_review=yes
- R2_006 DAY2_171500000: needs_human_review=yes
- R2_007 DAY3_122500000: needs_human_review=yes
- R2_008 DAY3_131500000: needs_human_review=yes
- R2_009 DAY4_122000000: needs_human_review=yes
- R2_010 DAY4_132500000: needs_human_review=yes

## Recommendation
Manually review the generated round-2 contact sheets. Only label true_complementary_A_plus_B if A-only and B-only are both no/unclear and A+B is yes. If no such cases appear, continue targeted search rather than building an automatic selector.
