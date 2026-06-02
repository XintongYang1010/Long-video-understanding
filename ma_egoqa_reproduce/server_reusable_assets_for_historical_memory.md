# Server Reusable Assets for Historical Memory Feasibility

Date: 2026-05-28

Scope: read-only inventory for EgoLife / MA-EgoQA / historical-memory feasibility. No new data was downloaded, no CLIP/VLM/LLM was run, and no original files were modified.

## 1. Relevant Directories

| Path | Size | File count | Relevance |
|---|---:|---:|---|
| `/scratch/xy3257` | 4.9G | not fully counted as a single root | Parent scratch area. Contains `ma_egoqa_reproduce`, `castle_hpc`, and `castle_poc`; also unrelated env/cache dirs. |
| `/scratch/xy3257/ma_egoqa_reproduce` | 690M | 24,716 | Main relevant workspace. Contains MA-EgoQA repo, data, captions, BM25/shared-memory artifacts, env/cache. |
| `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA` | 130M | 323 | Directly relevant to EgoLife / MA-EgoQA. Best source for current feasibility work. |
| `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data` | 125M | 267 | Directly relevant data: original MA-EgoQA JSON, caption JSONs, BM25 index files, shared-memory JSON. |
| `/scratch/xy3257/castle_hpc` | 934M | 21,380 | CASTLE-related only. Reusable mostly for remote-frame extraction / contact-sheet code patterns, not for EgoLife evidence. |
| `/scratch/xy3257/castle_poc` | 2.0G | 38,903 | CASTLE POC archive with incremental QA and true-A+B candidate artifacts. Useful as design precedent only; not current mainline. |

Explicitly requested but not found as top-level directories:

- `/scratch/xy3257/Egolife_analyse`
- `/scratch/xy3257/egolife*`
- `/scratch/xy3257/maegoqa*`
- `/scratch/xy3257/castle_self_first_project_archive`

## 2. EgoLife Reusable Data

Found EgoLife-derived caption data through MA-EgoQA:

| Asset | Path | Purpose | Directly usable for historical memory pool? |
|---|---|---|---|
| 30-second per-agent captions | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/caption/30sec/*.json` | Fine-grained per-agent current/historical text evidence. 42 files, one per 6 agents x 7 days. Example file has 828 clip-key entries for A1/Jake Day1. | Yes. Best text-level pool for self current, self history, external current, and external history. |
| 1-hour per-agent captions | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/caption/1hour/*.json` | Coarser per-agent summaries. 42 files. | Yes, as compressed historical memory; less precise than 30s. |
| 10-minute per-agent captions | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/caption/10min/*.json` | Mid-level per-agent summaries. 42 files. | Yes. Useful default historical-memory granularity. |
| 1-day per-agent summaries | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/caption/1day/captions.json` | Day-level agent summaries for all 42 agent-day streams. | Yes, for very coarse memory; not enough alone for evidence audit. |
| 10-minute shared memory | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/10min_shared_memory.json` | Cross-agent merged memory by `day`, `start`, `end`, `caption`. | Yes, but it is already aggregated across agents, so it should not be used where source isolation is required. |
| BM25 indexes | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/10min_bm25.pkl`, `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/30sec_bm25.pkl` | Existing text retrieval indexes over shared memory and 30s captions. | Reusable for retrieval baselines, but source-filtering must be audited. |
| MA-EgoQA assets images | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/assets/*.png` | README/paper figures only. | No. Not evidence data. |

Requested EgoLife files not found:

- `manifest_A*_DAY1_DAY7.csv`
- `session_summary_A*_DAY1_DAY7.csv`
- `participant_metadata_coverage_summary.csv`
- separate transcript inventory
- separate text inventory
- extracted EgoLife frames
- packet galleries
- downloaded EgoLife video clips

Important note: the caption JSON keys include `.mp4` clip names such as `DAY1_A1_JAKE_11094208.mp4`, but no actual EgoLife `.mp4` files were found under the scanned project paths. These are references, not local clips.

## 3. MA-EgoQA Reusable Data

| File | Path | Field overview | Contains Q/A/agents/context/timestamp? | Can construct QA candidates? |
|---|---|---|---|---|
| Original MA-EgoQA questions | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/MA-EgoQA.json` | Array length 1,741. Fields: `category`, `subcategory`, `question`, `options`, `answer`, `contexts`. Categories: environmental_interaction 359, social_interaction 376, task_coordination 484, temporal_reasoning 287, theory_of_mind 235. | Question: yes. Answer/options: yes. Agents: yes in many context values. Context/timestamp: yes, but mixed format. `contexts` values include arrays of agent names and also timestamped strings like `DAY1 13:16:55 - 13:16:57, Katrina: ...`. No explicit `query_user` label found. | Yes. Strongest source for selecting candidate QA cases, but needs derived labels for querying user and source-access tiers. |
| MA-EgoQA with retrieved BM25 contexts | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/MA-EgoQA_bm25.json` | Same 1,741 questions plus `bm25`. Each `bm25` item has `id` and `caption`, e.g. `1-1210-1220_0` with `DAY1 12:10:00 - 12:20:00` text. | Question/answer: yes. Retrieved context/timestamps: yes. Agents: sometimes in caption text, not guaranteed structured. | Yes for candidate discovery, but do not treat BM25 retrieval as proof of feasibility. |
| 10-minute shared memory | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/10min_shared_memory.json` | Array entries with `day`, `start`, `end`, `caption`; `caption` contains event objects with `name`, `action`, `location`, `detail`. | Agents: yes via `name`. Timestamps: yes via `day/start/end`. Q/A: no. | Yes as memory evidence pool; not as original questions. |
| Caption files | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/caption/{30sec,1hour,10min,1day}` | Agent-day caption dictionaries. 30s entries map clip IDs to first-person text; 1day maps agent-day IDs to long summaries. | Agents/timestamps: encoded in filenames/keys. Q/A: no. | Yes, especially for constructing source-isolated current/history evidence. |
| README | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/data/README.md` | Dataset description and citation. | No case-level annotation. | No, background only. |

Requested MA-EgoQA derived files not found:

- `selected_contexts.csv`
- `egolife_context_frame_plan.csv`
- `packet_manifest.csv`
- `frame_manifest.csv`
- `packet_gallery.html`
- `phase0_report.md`
- `phase0_mapping_report.md`
- `visual_review.md`

No separate MA-EgoQA full annotation file beyond `MA-EgoQA.json` was found in the scanned paths.

## 4. Reusable Code

### MA-EgoQA / EgoLife Code

| Script | Path | Input | Output | Reusable? | Needed changes |
|---|---|---|---|---|---|
| Shared memory construction | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/egomas/src/construct_shared_memory.py` | `data/caption/10min/*.json`; Gemini API prompt | Script code writes `data/10m_shared_memory.json`; existing reusable artifact is `data/10min_shared_memory.json`. | Partly. It shows how 10m captions are merged into cross-agent memory. | Do not use as-is for source-isolated feasibility because it calls Gemini and merges agents. Adapt only `_build_tasks` style parsing; disable API generation. |
| BM25 indexing/retrieval | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/egomas/src/index_bm25.py` | 10min shared memory JSON or 30sec caption directory | `10min_bm25.pkl`, `30sec_bm25.pkl`, `MA-EgoQA_bm25.json` | Yes for text retrieval utilities. | Add explicit source filters by querying user, agent, day, current window vs historical window; avoid treating retrieved output as evaluated result. |
| Person-filtered retrieval helper | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/egomas/src/retrieval_helpers.py` | BM25 retriever, person name, query | Formatted retrieved context string | Yes. Closest existing source-filter primitive. | Extend to four access tiers: self current, self history, external current, external history. Need timestamp-window exclusion/inclusion. |
| Inference scripts | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/egomas/src/inference_egomas.py`, `inference_egomas_singleproc.py` | MA-EgoQA data plus retrievers/model config | Model answers/eval outputs | No for this inventory task. Potential later use only. | Do not run for feasibility inventory. If reused later, separate packet generation from VLM/LLM inference. |
| Parsing / formatting utils | `/scratch/xy3257/ma_egoqa_reproduce/MA-EgoQA/egomas/utils/*.py` | JSON/text/prompt helper inputs | formatted context, prompt text, eval utilities | Partly. | Keep only source-neutral IO/formatting. Avoid prompt/eval code until packet definitions are fixed. |

No dedicated EgoLife manifest parsing, frame extraction, gallery building, recurrence-check, CLIP selector, or packet generator scripts were found in the MA-EgoQA workspace.

### CASTLE Code Adaptable As Patterns Only

| Script / asset | Path | Input | Output | Reusable? | Needed changes |
|---|---|---|---|---|---|
| Remote low-bandwidth frame extraction | `/scratch/xy3257/castle_hpc/castle_low_bandwidth_remote_frame_test.py` | CASTLE interval/window CSVs, HF remote video URLs | remote frames, logs, CSV selection, contact sheet | Pattern reusable for future EgoLife frame packet generation if remote URLs are available. | Replace CASTLE interval schema/entities/HF repo paths with EgoLife clip metadata. Do not assume CASTLE clocks/entities. |
| Event-relevant view selection | `/scratch/xy3257/castle_hpc/castle_event_relevant_view_selection.py` | CASTLE interval/window CSVs; optional CLIP | overview/event frames, view similarity CSV, contact sheets | Pattern reusable for multi-view packet construction. | Remove or disable CLIP for inventory; adapt source isolation around EgoLife agents and MA-EgoQA contexts. |
| CASTLE incremental QA POC | `/scratch/xy3257/castle_poc/castle_incremental_qa_poc.py` | CASTLE interval table | selected windows, annotation templates, frame sheets | Conceptually reusable. It encodes incremental evidence/access-tier thinking. | Port the access-tier logic to MA-EgoQA captions. Do not reuse CASTLE evidence or labels as EgoLife results. |
| True A+B candidate extraction | `/scratch/xy3257/castle_poc/castle_extract_true_AB_candidates.py`, `castle_extract_true_AB_candidates_round2.py` | CASTLE candidate CSVs, interval/window tables | candidate CSVs, contact sheets, summaries | Useful as case-set workflow precedent. | Replace with MA-EgoQA question/context/caption fields. Need query_user and current/history labels. |
| CASTLE candidate CSVs | `/scratch/xy3257/castle_poc/candidate_true_AB_cases.csv`, `candidate_true_AB_cases_round2.csv` | manually proposed CASTLE cases | 9 and 10 listed candidate cases respectively | Not data-reusable for EgoLife, but schema is useful. | Use schema columns such as `proposed_querying_user`, `A_only_sufficient`, `B_only_sufficient`, `A_plus_B_required`, `needs_human_review` for new EgoLife case audit. |
| CASTLE contact sheets | `/scratch/xy3257/castle_poc/candidate_true_AB_contact_sheets` | CASTLE remote frames | 276 images/contact-sheet frames | Not evidence-reusable. | Only reuse layout conventions if building EgoLife packet galleries later. |

CASTLE warning: scripts existing on disk do not mean the EgoLife/MA-EgoQA historical-memory experiment is done. They are only reusable implementation patterns.

## 5. Missing Pieces

| Item | Status |
|---|---|
| Original MA-EgoQA full annotation | Partially present as `MA-EgoQA.json`; no richer full annotation file found. |
| `query_user` label | Missing. MA-EgoQA contexts include agents, but there is no explicit querying-user field. This is the main blocker for self-first source access. |
| Captions / transcripts | Captions present at 30s/1h/10m/1day. Separate transcripts not found. |
| Historical memory index | Partly present: `10min_bm25.pkl`, `30sec_bm25.pkl`, `10min_shared_memory.json`. Missing a source-isolated historical-memory index with self/external and current/history partitions. |
| Human audit labels | Missing for EgoLife/MA-EgoQA historical-memory cases. CASTLE candidate CSVs have review-style columns, but not applicable as EgoLife labels. |
| VLM input packet generator | Missing for EgoLife/MA-EgoQA. No `packet_manifest.csv`, `frame_manifest.csv`, or `packet_gallery.html` found. |
| EgoLife frames / clips | Missing locally in scanned paths. Caption keys reference clip names, but actual EgoLife videos/frames were not found. |
| Context mapping reports | Missing: no `phase0_report.md`, `phase0_mapping_report.md`, `visual_review.md`, or `egolife_context_frame_plan.csv`. |

## 6. Recommended Next Minimal Task

The smallest executable next task is to build a small, read-only MA-EgoQA candidate-case table from existing JSON/caption assets, without running models or downloading video.

Concretely:

1. Start from `MA-EgoQA.json` and select a small set of questions whose `contexts` include clear day/time and multiple agents.
2. For each selected question, manually assign or infer a provisional `query_user` from the question/context only, and mark it as `needs_human_review=yes`.
3. For each case, derive four text-only evidence pools from existing captions:
   - self current: same `query_user`, same context window
   - self historical: same `query_user`, earlier windows/days
   - external current: other agents, same context window
   - external historical: other agents, earlier windows/days
4. Store only a candidate/audit manifest, not model outputs. Suggested columns can reuse the CASTLE schema pattern: `case_id`, `question`, `answer`, `query_user`, `context_window`, `self_current_available`, `self_history_available`, `external_current_available`, `external_history_available`, `needs_human_review`, `notes`.

This stays within the current research goal: checking whether self-first incremental source access remains feasible after adding historical memory on EgoLife / MA-EgoQA.
