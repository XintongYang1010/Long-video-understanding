# True A+B Candidate Search Summary

This is a low-bandwidth visual evidence generation step. It does not run VLM/LLM inference, does not run official CASTLE QA, and does not download full videos.

- number of candidate windows selected: 9
- number of windows with generated visual evidence: 9
- number of contact sheets generated: 18
- frame extraction success: 303/303
- number of candidate cases listed: 9
- number of possible true_complementary_A_plus_B cases: 6 reviewable candidates, 0 locked true cases

## Strongest 3 candidate cases
- C001 DAY1_100500000: Can we tell both that someone is presenting and what the group/screen context is? (self_insufficient_other_sufficient)
- C002 DAY3_174500000: Can we connect tabletop activity with kitchen activity in the same mixed window? (query_dependent)
- C004 DAY1_102000000: What is the presenter referring to, and who is addressing the seated group? (weak_or_unclear)

## Weak/unclear cases
- C003 DAY4_120500000: redundant_control; needs_human_review=yes
- C004 DAY1_102000000: weak_or_unclear; needs_human_review=yes
- C005 DAY2_141000000: weak_or_unclear; needs_human_review=yes
- C006 DAY2_183500000: weak_or_unclear; needs_human_review=yes
- C007 DAY3_121500000: weak_or_unclear; needs_human_review=yes
- C008 DAY4_100500000: weak_or_unclear; needs_human_review=yes
- C009 DAY4_182500000: weak_or_unclear; needs_human_review=yes

## Recommendation
Review generated contact sheets manually before making any true A+B claims. If extraction failed or no sheets were generated, rerun this script on the NYU/conda ffmpeg environment where remote seeking previously worked.
