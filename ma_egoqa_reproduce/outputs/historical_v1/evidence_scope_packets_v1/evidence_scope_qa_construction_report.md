# Evidence Scope QA Construction Report v1

This is a caption-only evidence-scope packet construction draft. It does not use video, VLM inference, LLM API calls, or final human labels.

## Counts

- Total candidate cases reviewed: 30
- Constructed high-confidence/medium-confidence QA subset size: 1
- current-only likely_answerable: 12
- history-only likely_answerable: 7
- current+historical likely_answerable: 14
- Cases where current+historical appears better than current-only: 4 yes; 2 unclear-but-promising

## Answerability Distributions

- current-only: {'unclear': 6, 'partially_answerable': 11, 'not_answerable': 1, 'likely_answerable': 12}
- history-only: {'not_answerable': 3, 'partially_answerable': 12, 'unclear': 8, 'likely_answerable': 7}
- current+historical: {'unclear': 4, 'partially_answerable': 11, 'not_answerable': 1, 'likely_answerable': 14}
- gain labels: {'no': 24, 'yes': 4, 'unclear-but-promising': 2}
- likely buckets: {'not_self_first_or_unclear': 21, 'current_sufficient': 6, 'history_adds_useful_context': 1, 'current_plus_history_needed': 1, 'history_irrelevant': 1}
- keep recommendations: {'no': 22, 'unclear': 7, 'yes': 1}

## Best Demo Candidates

- HISTV1_026 (medium): Why was Katrina initially confused about what was happening with the ball during the game? | best_scope=current_plus_historical

## Likely Rejects

- HISTV1_001: bucket=not_self_first_or_unclear; keep=no; reason=current=unclear (global/statistical/temporal question with partial lexical overlap; caption coverage may be too local) | history=not_answerable (global/statistical/temporal question and this evidence scope does not expose enough answer-related coverage) | c...
- HISTV1_002: bucket=not_self_first_or_unclear; keep=no; reason=current=unclear (global/statistical/temporal question with partial lexical overlap; caption coverage may be too local) | history=partially_answerable (global/statistical/temporal question; captions mention answer-related terms but may not cover enough event...
- HISTV1_003: bucket=not_self_first_or_unclear; keep=no; reason=current=partially_answerable (global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order) | history=not_answerable (global/statistical/temporal question and this evidence s...
- HISTV1_004: bucket=not_self_first_or_unclear; keep=no; reason=current=partially_answerable (global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order) | history=unclear (global/statistical/temporal question with partial lexical overl...
- HISTV1_005: bucket=not_self_first_or_unclear; keep=no; reason=current=not_answerable (evidence does not contain enough answer-relevant caption content) | history=not_answerable (evidence does not contain enough answer-relevant caption content) | current+history=not_answerable (evidence does not contain enough answer-r...
- HISTV1_006: bucket=not_self_first_or_unclear; keep=no; reason=current=partially_answerable (global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order) | history=partially_answerable (global/statistical/temporal question; captions men...
- HISTV1_008: bucket=not_self_first_or_unclear; keep=no; reason=current=likely_answerable (evidence covers key answer terms and strongly overlaps with the question context) | history=likely_answerable (evidence covers key answer terms and strongly overlaps with the question context) | current+history=likely_answerable (...
- HISTV1_009: bucket=not_self_first_or_unclear; keep=no; reason=current=partially_answerable (evidence covers some answer keywords but may not fully support the answer) | history=unclear (evidence has lexical overlap but does not clearly support the answer) | current+history=likely_answerable (evidence covers key answer...
- HISTV1_010: bucket=not_self_first_or_unclear; keep=no; reason=current=likely_answerable (evidence covers most answer keywords and overlaps with the question context) | history=likely_answerable (evidence covers key answer terms and strongly overlaps with the question context) | current+history=likely_answerable (evide...
- HISTV1_012: bucket=not_self_first_or_unclear; keep=no; reason=current=likely_answerable (evidence covers key answer terms and strongly overlaps with the question context) | history=likely_answerable (evidence covers key answer terms and strongly overlaps with the question context) | current+history=likely_answerable (...
- HISTV1_013: bucket=not_self_first_or_unclear; keep=no; reason=current=partially_answerable (global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order) | history=partially_answerable (global/statistical/temporal question; captions men...
- HISTV1_014: bucket=not_self_first_or_unclear; keep=no; reason=current=partially_answerable (global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order) | history=partially_answerable (global/statistical/temporal question; captions men...

## What This Draft Suggests

- We built a caption-only evidence-scope QA packet set.
- The draft suggests whether adding historical memory may improve evidence coverage for selected cases.
- The current result is useful for deciding which cases deserve manual packet inspection and later model-input construction.

## What This Draft Cannot Claim

- It does not show that the benchmark is complete.
- It does not provide final labels.
- It does not show model accuracy improved.
- It does not prove historical memory is useful.
- It does not solve self-first routing.
- It does not claim answer accuracy for any MA-EgoQA item.
