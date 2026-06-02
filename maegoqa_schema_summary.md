# MA-EgoQA Schema Summary

Local MA-EgoQA files were found, so this optional schema summary was generated without downloading a dataset.

- QA file: `ma_egoqa_reproduce/MA-EgoQA/data/MA-EgoQA.json`
- QA items: 1741
- Top-level keys in sample: answer, category, contexts, options, question, subcategory
- Context spans per item: min 1, median 1.0, max 5

## Question Schema

- `question`: natural-language QA prompt.
- `category`: broad reasoning category.
- `subcategory`: more specific reasoning pattern.
- `options`: answer candidates.
- `answer`: correct option text.
- `contexts`: timestamp-window keys mapped to visible/participating agents.

## Category Counts

| Category | Count |
| --- | ---: |
| task_coordination | 484 |
| social_interaction | 376 |
| environmental_interaction | 359 |
| temporal_reasoning | 287 |
| theory_of_mind | 235 |

## Subcategory Counts

| Subcategory | Count |
| --- | ---: |
| single_span | 703 |
| environmental_interaction | 359 |
| theory_of_mind | 235 |
| comparison | 162 |
| multi_span | 157 |
| concurrency | 125 |

## Agents

- JAKE: 1459 context mentions
- ALICE: 1304 context mentions
- LUCIA: 1262 context mentions
- TASHA: 1201 context mentions
- SHURE: 1180 context mentions
- KATRINA: 1147 context mentions

## Captions/Transcripts

- 10min: 42 JSON files
- 1day: 1 JSON files
- 1hour: 42 JSON files
- 30sec: 42 JSON files

Sample caption file: `ma_egoqa_reproduce/MA-EgoQA/data/caption/10min/10m_captions_A1_JAKE_DAY1.json`: dict with 45 top-level entries

## Evidence Labels

The local QA schema exposes context windows and agents, but no explicit per-modality evidence-route labels were found in the sample schema. Captions are available at multiple temporal granularities and can seed memory/history baselines.
