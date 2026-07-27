# BeliefKV 与 SGLang 0.5.2rc1 集成说明

## 1. 固定版本

BeliefKV 当前只支持：

```text
SGLang tag:    v0.5.2rc1
SGLang commit: 18f91eb639084825717c0e3c3c7273492812ab71
```

`beliefkv check-sglang SOURCE_ROOT` 会检查 `version.py`、git HEAD（源码目录有
git 信息时）、上游 AST 接口和 BeliefKV patch marker。检查失败时应停止实验，
不能在相近版本上强行运行。

## 2. Patch 的职责

`patches/sglang-0.5.2rc1-beliefkv.patch` 只增加窄接口：

- HTTP/generation input 到 `Req` 的 `beliefkv_metadata` 传播；
- request 始终进入 SGLang waiting queue，BeliefKV 在每个 prefill epoch 编译短期 admission
  ticket，并在 `PrefillAdder` 前后做局部校验；
- abort 时清理 BeliefKV visible side state、当前 epoch ticket 和未完成事务；
- scheduler safe point 驱动 ACK、控制器和迁移 backend；
- SGLang 原生 queue policy 后为 tagged request 提供 causal/ticket candidate view；无 ticket 的
  request 本轮跳过但仍留在原生 waiting queue；
- Radix split/insert/delete/lock 与 HiCache residency 变化触发 observer；
- server flags `--enable-beliefkv` 和 `--beliefkv-config`。

Patch 不把 BeliefKV 策略复制到 SGLang；策略仍位于本仓库。SGLang 继续拥有
allocator、Radix topology、KV tensor 和 DMA queue。

## 3. 请求 metadata

请求 JSON 可带：

```json
{
  "beliefkv_metadata": {
    "root_workflow_id": "wf-42",
    "invocation_id": "coder-call-3",
    "context_id": "coder-session",
    "context_epoch": 7,
    "agent_definition_id": "coder",
    "agent_instance_id": "coder-1",
    "parent_invocation_id": "planner-call-2",
    "parent_context_id": "planner-session",
    "relation_type": "call",
    "context_mode": "resume",
    "execution_mode": "foreground",
    "return_target_id": "planner-call-2",
    "join_id": null
  }
}
```

没有该字段的请求绕过 BeliefKV ticket gate。`relation_type` 只能是
`root/call/spawn/message/handoff`，context 和 execution mode 也会严格校验。

仅靠 request metadata 能恢复 invocation/context 关系，但工具开始/结束、join
和独立消息总线事件仍应由 agent runtime 通过 `RuntimeEvent` hook 上报。通用的
线程安全批处理接口位于 `beliefkv/runtime/agent_runtime_adapter.py`；当前没有
跨进程网络 collector，接具体 agent framework 时需要实现一个薄 adapter。

## 4. 运行时审计

配置中的 `runtime_audit_path` 默认为 `null`，此时不打开文件，也不增加调度器
I/O。设置为 JSONL 路径后，每条记录包含 `run_id + sequence + monotonic ts`，可
验证以下链路：

```text
invocation_created
  -> request_visible_pending
  -> admission_ticket_epoch_started
  -> ticket selected or skipped
  -> request_started
  -> request_finished
```

CALL/SPAWN/MESSAGE/HANDOFF 还会记录 `causal_relation_linked`，迁移会记录
`transfer_dispatched/transfer_acknowledged`。日志只包含 identity、token/byte
计数和状态，不保存 prompt、observation 或生成文本。重复的 admission 拒绝按
状态去重，避免调度循环造成日志风暴。该日志用于正确性与实验审计，不应在正式
性能测量中开启，除非各 baseline 采用等价的观测开销。

## 5. Scheduler 事务顺序

每个 safe point 的固定顺序是：

```text
drain HiCache ACK
  -> 同步 dirty Radix tree
  -> 上报 allocator/HBM
  -> admission/transfer planning
  -> 提交至多一个迁移 command
  -> SGLang 原生 queue policy
  -> begin_prefill_epoch 编译 causal/active-set ticket
  -> ticket gate + prefix rematch + PrefillAdder
  -> end_prefill_epoch 提交实际 selection accounting
```

ACK 必须先于 tree sync。否则同步完成的 `COMMIT_CPU` 或 `DROP` 会让控制面先看到
物理新状态，随后又根据 ACK 重复执行状态转换。

## 6. HiCache 限制

- 只迁移 sealed Radix node extent；
- D2H 遵守 leaf/prefix closure；
- H2D 必须从浅到深选择完整 CPU ancestor closure，禁止由 `load_back()` 隐式
  加载未计费祖先；
- CUDA DMA 一旦提交不假设可抢占，cancel 只停止后续 shadow chunk；
- native HiCache write/load 期间分别镜像为 `MIRRORING/PREFETCHING`；
- cache reset 先生成 `CANCELLED` ACK，再失效 allocation generation。

## 7. Predictor

先将真实 trace 归一化，再训练：

```bash
beliefkv normalize-clawtrace raw.jsonl runtime_events.jsonl
beliefkv train-predictor runtime_events.jsonl predictor.json
```

artifact 保存工具 survival curve、action context tree、LLM service bucket 和训练
摘要；原始 prompt 不进入模型。训练 trace 应按 project/session、时间和 workload
family 划分，不能随机拆散同一 workflow。

## 8. 真机验收清单

- 未带 metadata 的请求与 upstream 返回一致；
- patched-disabled 相对 upstream 的吞吐/延迟开销可测且足够小；
- active/shared/locked page 不会迁移；
- abort deferred、abort admitted、cache reset 和 host allocation failure 不泄漏；
- HBM pressure 下 admission 只在实际释放 ACK 后继续；
- D2H/H2D 字节与 HiCache allocator 计数一致；
- shadow slowdown 不超过配置预算；
- 长时间混合 workload 不出现 stale handle、location divergence 或死锁。

2026-07-15 已使用 Qwen2.5-0.5B-Instruct 完成单卡 CUDA smoke：未标注请求
成功旁路，root 与 spawn child 均在同一审计 `run_id` 下完成上述生命周期，且
child 建立了指向 root 的因果边。该结果验证集成机制，但不验证高 HBM pressure、
长时间稳定性或性能收益。完成清单中的压力与故障测试之前，不能称为已经完成
生产级 GPU 验证。
