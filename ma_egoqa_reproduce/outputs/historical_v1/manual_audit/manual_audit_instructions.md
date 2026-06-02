# manual_historical_case_audit_v1 标注说明

本轮只做人审准备，不判断答案正确性。CSV 里的检索 caption 和 BM25 分数只是帮助你定位证据的线索，不是最终 label。

## 基本原则

- 每行是一个候选 case。
- `provisional_query_user` 是脚本弱推断的候选用户，不一定正确。
- 所有人工字段初始为空；请只在人工读过问题、上下文和 caption 后填写。
- 不要把 BM25 score、rank 或 top1 caption 当成真实标签。它们只能提示“先看哪里”。
- 不要 claim answer accuracy；本表标的是 source routing / historical memory 是否有证据支持。

## 人工字段怎么填

`human_query_user`：你人工判断的查询用户/第一人称主体。无法判断就留空或写 `unclear`。

`query_user_valid`：`yes` 表示 provisional query_user 与你判断一致；`no` 表示不一致；`unclear` 表示无法从材料判断。

`is_self_first_suitable`：这个问题是否适合先查 query_user 自己的视角。若问题是全局统计、多人整体比较、没有明确第一人称主体，或一开始就必须依赖其他人视角，通常填 `no`。

`self_current_sufficient`：在确认 human_query_user 后，只看该用户 current/context window 内的 caption，是否足以支持回答所需的关键证据。若 current caption 只部分相关、缺少关键事件、时间不对，或必须依赖之前记忆/他人视角，填 `no`。证据不够清楚填 `unclear`。

`self_history_sufficient`：该用户在 context window 之前的历史 caption，是否提供了 current 缺失的关键证据。只有当 self history 能独立或明确补足关键事实时填 `yes`；如果只是词面相关但不能支持判断，填 `no` 或 `unclear`。

`external_current_sufficient`：非 query_user 的 current/context window caption 是否足以支持关键证据。若需要当前其他人的视角才能回答，且 caption 清楚支持，填 `yes`。

`external_history_sufficient`：非 query_user 的历史 caption 是否足以支持关键证据。若 evidence 来自别人过去看到/说过/做过的内容，且能支持问题，填 `yes`。

`historical_memory_helpful`：如果 self_history 或 external_history 相比 current caption 明显补充了必要证据，填 `yes`。如果 current 已足够、history 只是冗余或无关，填 `no`。无法判断填 `unclear`。

`final_case_type` 只能填以下之一：

- `self_current_sufficient`
- `self_history_needed`
- `external_current_needed`
- `external_history_needed`
- `not_self_first`
- `reject_unclear`

`oracle_source_agent`：你认为最关键证据来自哪个 agent。多个 agent 可用分号分隔。

`oracle_time_window`：关键证据的大致时间窗口，例如 `DAY3 18:10-18:20`。

`oracle_caption`：最能支持人工判断的 caption 摘要或原文片段。

`audit_notes`：写下为什么这样标，尤其是 query_user 不清、caption 不足、时间窗口不匹配等情况。

`keep_for_demo`：适合放进论文/报告 demo 的填 `yes`；不适合填 `no`；暂不确定填 `unclear`。

## 什么时候标 reject_unclear

- 无法确认 query_user，且问题是否 self-first 也说不清。
- 检索 caption 只有词面相关，无法支持答案所需事实。
- 时间窗口明显不匹配，无法判断 current/history 的边界。
- 问题本身依赖未提供的视频细节，caption 不足以审。
- 答案可能对但当前证据链不清楚；不要为了保留 case 硬标。

## 什么时候不适合 self-first

- 问题问的是全体成员统计或全局比较，例如“who used X the most”，且没有明确 query_user。
- 问题天然需要多人的外部视角或旁观者视角，query_user 自己不可能优先提供主要证据。
- `human_query_user` 无法确定，或 provisional query_user 只是名字出现在问题里但不是实际主体。
- 答案需要聚合多个 agent 的经历，而不是先从某个 self stream 开始扩展。
