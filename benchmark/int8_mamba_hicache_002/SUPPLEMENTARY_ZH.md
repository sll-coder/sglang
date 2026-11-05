# 002 中文 PR 补充材料

本材料对应独立 002 提交 `15a00b66a6b388c5c795d533e4ef21a6ee92ff5e`，基线 main `af98fcd545f90eaaf4b26d20bfb42e3ec1d91020`。不包含 001 的通用回滚改动或 003 的融合 codec。本次整理只修改文档和归档，没有重新运行代码回归，也没有创建或更新远端 PR。

## 阅读索引

- `regression_service/`：故障 before/after、CPU/GPU/Rust 回归、真实 GDN/KDA TP2 服务与固定批次控制。
- `accuracy_quantized_weights/`：NVFP4 GDN / AWQ KDA 的全量准确率、logprob、KDA 复算波动及协议。
- `accuracy_bf16_weights/`：未做权重量化的 GDN 全量精度对照；与旧 GDN 全部输入/参考 token 一致。
- `accuracy_inputs/`：冻结的 GSM8K 数据与两个模型的输入 token；BF16 GDN 与 NVFP4 GDN 共用同一份输入。
- `performance/`：用户指定 ratio=2、并发 8、72 个 BF16 状态大小的 L2 预算下，正式三轮性能结果、逐请求记录、双 rank 路径审计、输入和实验驱动。
- `followup/`：等 72 槽的实际 buffer 和锁页内存、72 个前缀的逐请求回迁记录，以及 1/4/16/32 次连续恢复的统计、正式逐题生成与评分、图表和复现配置。完整服务日志、状态快照和逐次传输审计保留在原始实验目录。
- `SHA256.json`：归档证据文件的大小与哈希。

归档保留精度汇总及服务重启对照，精度测试的完整逐请求输出/评分/搬运审计仍在原实验目录。性能部分包含全部正式逐请求计时和路径审计；原始完整 server.log、GPU telemetry 留在原目录。模型权重和运行环境不包含在归档中。驱动记录原机器绝对路径，复现时需要按本地模型、源码和依赖位置调整。

## 结果应如何解释

1. 性能比较两组都开启 L2，同样的逻辑 L1 8 槽、Mamba L2 字节预算以及 attention KV 容量。L1 配额和独立 host 预算通过实验驱动控制，不能描述为只改两个原生启动开关即可得到同样分配。
2. 每 rank Mamba L2 预算为 2,824,077,312 bytes，等于 72 个 BF16 状态；同预算 INT8 向下取整容纳 136 个，余量分别记录。两组整卡 GPU 分配量没有被设为相等，也没有声称 GPU 峰值下降。
3. 容量主场景 1K/N96 的 KV 需求为 98304 tokens，小于 KV L2 的 131072 tokens。状态容量差异可以在该设置下独立体现。1K/N160 同时超过 KV/Mamba 容量，作为整体压力结果，不将其全部收益单独归因于 Mamba。
4. 两组保留原文本计算和 MRoPE；只在启动时把未使用的视觉模块放 CPU，跳过默认图像 warmup，并使用 token-only 预热。非视觉权重全部在 GPU；32/32 固定串行输出与原完整模型一致。
5. 性能输出长度固定为 32 tokens（ignore_eos），以排除生成长度差异；这轮性能实验不替代精度评测。精度实验采用独立的 1024-token 预算和固定答案提取规则。
6. 定时窗口不强制驱逐指定前缀、不改写状态、不注入命中，不增加逐项传输同步。Mamba L2 命中由真实状态回迁确认，并检查两个 rank。每个 rank 的模型实际 prefill token 总数均与客户端推导值一致。
7. 三轮采用交错格式顺序。两个格式各 worker 的键序列和请求数量相同；并发下实际调度交错可能不同。表中吞吐变化是三轮配对比值均值，不是把所有请求合并后计算的比值；三轮范围不是置信区间。
8. 正式性能实验为 10 场景、6 个独立服务、每场景每服务 872 请求，总计 52,320 个计时请求，单窗口 90.59–219.99 秒。无请求失败/retraction，admission skip/load defer 均为零。
9. 停止的旧 512 MiB 实验及其预跑不混入正式结果；其曾观测到的约 4.2× 短窗口收益不作为本 PR 性能结论。
10. 当前没有这轮配置的 Nsight trace，不据此声称传输 kernel 或 codec 单独加速。scale side-state 仍沿用通用 blocking copy。
11. 等容量补测固定两组均为 72 个可用 L2 槽，实际填满并回迁全部前缀。实测 Mamba L2 主机内存下降 GDN 47.37%（三轮）、KDA 45.99%（一轮）；这与等预算的 72→136 槽是两项互补的对照。降幅只描述 Mamba L2 池。
12. 连续恢复每轮都续算新 token 并保存新的 checkpoint。各深度固定相同最终 8K 前缀，但分段长度不同；BF16 分段本身也有数值差异。以同深度 INT8/BF16、同格式 L2/L1 为主对照；32 题不能用于证明长期精度等价。

下面附完整实验报告快照，原始目录索引见末尾。


## 回归与真实服务完整报告


### 002 最新 main 适配和真实服务验证

完成日期：2026-11-03。

基线：官方 main `af98fcd545f90eaaf4b26d20bfb42e3ec1d91020`，本次开始时重新 fetch 得到。

工作树：`/mnt/sll/sglang-main-int8-hicache-002-20261103`。

独立分支：`feat/int8-mamba-hicache-002-20261103`。

本地提交：`15a00b66a6b388c5c795d533e4ef21a6ee92ff5e`；工作树干净。改动为 563 行增加、24 行删除。

#### 结果

002 已按当前 main 接口适配，完成 checkpoint 耗尽回归，以及 GDN/KDA 的真实 TP2 服务验证。两种模型都实际发生了压缩 checkpoint 的 GPU L1→host L2→GPU L1 回迁、驱逐、并发请求和重复恢复。

001 和 003 没有合入本分支，其原工作树仍干净。当前 INT8 codec 类与 main 完全相同，002 不需要融合 codec。001 的通用 controller load 和 load-back admission 方法也保持 main 原样；其既有回滚和 pin handoff 问题仍由独立的 001 处理。

改动共 11 个文件：7 个生产文件、4 个现有测试文件；没有新增算子、环境变量或服务接口。

#### 具体改动

1. **压缩池元数据**：增加 `MambaCheckpointDevicePool`，通过现有 `DevicePoolInfo` 公布 INT8 qdata、native scale、native conv、checkpoint 容量和实际字节数。host 不再误读 active BF16 pool，也不建立 BF16 镜像。
2. **组装及驱逐**：Mamba 和 Mamba+SWA host stack 在 INT8 模式选择 checkpoint pool 和 checkpoint allocator；回迁驱逐按 checkpoint 空间计算，保持运行请求的 active allocator 语义不变。
3. **恢复路径**：L2 回迁只恢复树节点的 checkpoint；随后设置 request deferred CoW，把该 checkpoint 反量化到独立 active slot。禁止原始 INT8 payload 直接写入 request 的 BF16 slot。
4. **传输等待**：全状态反量化前调用已有 `layer_router.wait_for_part(STATE_PART)`，等待相关层完成 H2D。
5. **耗尽及异常**：checkpoint 全被占用/pin 时返回“本次不缓存”；先预留 checkpoint，再申请替代 tracking slot；编码成功后才 donation。申请或编码失败时归还保留资源，保持 request tracking 所有权。
6. **支持范围**：在完成上述接线后开放受限配置；启动和 runtime storage attach 都拒绝未经覆盖的组合，L3 后端不能运行时绕过限制接入。

复用 main 的 `MambaPoolHost` 传输 qdata/scale/conv，没有照搬旧版专用 host mover。现有 side-state scale 搬运仍包含 blocking copy；本次没有声称优化该部分耗时。

#### checkpoint 耗尽的 before/after 回归

在未修复 main 上运行新增的 5 个 admission 方法：**3 failed、2 passed**。失败覆盖 finished/unfinished checkpoint 耗尽，以及编码异常后的资源泄漏。保留了原始 `exhaustion_before.log`。

修复后 5 个方法全部通过，覆盖：

| 条件 | 修复后的行为 |
|---|---|
| checkpoint 全被占用且不可驱逐，unfinished request | 返回 0，不 donation；tracking `[2,3]`、active 空闲数保持不变 |
| 同上，finished request | 返回 0、没有新 checkpoint，保留原状态供正常清理 |
| checkpoint 已预留，但 active slot 申请失败 | 归还 checkpoint，不改变 tracking |
| 真实 codec 源索引错误 | 抛出原异常，归还 checkpoint 和新 active slot，tracking 不被替换 |
| eviction 释放一个 checkpoint | 正常缓存；编码数据和 scale 与 eager 参考完全相同 |

测试使用真实分配器、checkpoint pool、storage、component 和 request-table donation；仅在 eviction 外部边界模拟“无可回收槽位”或释放一个真实 reservation。没有把受控 pin 条件描述为完整服务中的并发故障复现。

其他新增方法覆盖压缩池 descriptor/allocator、驱逐容量、错误模型 side state/packed draft 的拒绝、runtime L3 attach 拒绝、启动配置矩阵，以及真实 GPU cache 回迁和 deferred CoW 等待。

GPU 等待用例让 checkpoint 内容在传输 fence 回调中才变为有效值，随后调用真实 ModelRunner restore 和 eager codec；漏掉等待会恢复出错误的零状态。该用例同时验证 qdata、scale、conv 往返一致、checkpoint allocator 减少一槽，以及 request active pool 不被误作 checkpoint 目标。

#### 相关回归

| 检查 | 结果 |
|---|---|
| codec/admission、assembler、side state、transfer metadata、Mamba capacity、完整 server-args 文件 | **431 passed、1 skipped、281 subtests passed** |
| GPU codec、HiCache load-back 矩阵、timing、新 INT8 恢复用例 | **82 passed、19 skipped** |
| 原生 Rust Mamba TreeCore 相关项 | **48 passed、534 deselected** |
| 适用 pre-commit hooks、whitespace | **通过** |

CPU 的 1 项 skip 为 CUDA decode 项，已在 GPU 那轮执行。GPU 19 项为配置不适用。CPU/GPU codec 有重复覆盖，不累加为独立测试数量。没有运行完整远端 CI。

#### GDN/KDA 真实服务

硬件：2× RTX 5090，TP=2。运行实际模型：

- GDN：`/mnt/sll/models/Qwen3.8-27B-NVFP4`。
- KDA：`/mnt/sll/models/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit`。

每轮固定 active capacity=32、INT8 checkpoint capacity=16、max-running-requests=8、KV capacity=32768；HiCache 使用 kernel IO、page_first、write_through，host ratio=8。保留默认 overlap 和 decode CUDA Graph；prefill CUDA Graph 关闭。所有编解码都使用 main 的 eager 实现。

每轮流程：8 个目标前缀预热 → 8 次串行 L1 参考 → 32 个不同前缀 churn 使 checkpoint 淘汰 → 8 次目标 L2 恢复 → 64 次并发压力请求 → 8 次再次恢复。前缀长 1024；参考查询输入长 1032、输出 16 tokens。

分别运行带逐项同步校验和关闭额外校验的两种模式，共 4 轮、**512 个请求**：

| 模型/模式 | 成功请求 | 客户端并发峰值 | L1 参考 | 首次 L2 恢复 | 并发窗口 | 再次 L2 恢复 | 串行 L1/L2 输出一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GDN，逐项校验 | 128/128 | 8 | 8/8 | 8/8 | 56 L2 + 8 L1 | 8/8 | 8/8 |
| GDN，关闭额外校验 | 128/128 | 8 | 8/8 | 8/8 | 56 L2 + 8 L1 | 8/8 | 8/8 |
| KDA，逐项校验 | 128/128 | 8 | 8/8 | 8/8 | 56 L2 + 8 L1 | 8/8 | 8/8 |
| KDA，关闭额外校验 | 128/128 | 8 | 8/8 | 8/8 | 56 L2 + 8 L1 | 8/8 | 8/8 |

L1 请求报告 device=1024、host=0；L2 请求报告 device=0、host=1024。每轮两个 scheduler rank 都记录 **40 个 checkpoint backup slots、72 个 load slots**，以及正常的缓存驱逐。每轮服务日志无 Python traceback，所有请求完成长度正确、没有 retraction。

逐项校验轮在真正的搬运方法之后比较 qdata、scale、conv 的原始值，两张卡均完全一致。额外同步会影响调度，因此另跑关闭这些校验的轮次；现有 generic scale fallback 自身仍有同步，这里不称为完全无同步传输。

实际压缩 host payload 为 GDN **20,643,840 bytes/checkpoint/rank**、KDA **6,062,080 bytes/checkpoint/rank**，均包括 conv 和 scale。此处仅描述数据格式；没有做额外显存峰值或吞吐提升基准。

##### GDN 并发输出的批次对照

压力窗口中，把相同目标请求的 C=8 输出与串行参考比较，GDN 并非全部相同（逐项校验轮 4/16、普通轮 6/16）；KDA 为 16/16。不能据此把全部并发输出写成 bit-exact 正确性通过，也不能直接把差异归因于 L2。

为区分批次变化和搬运，额外启动 GDN TP2 服务，使用同一组 8 个输入，分别执行：串行 L1、一次 batched L1、churn 后的 batched L2、再次 batched L1。缓存报告确认各阶段的实际命中层级。

这轮 **72 个请求**全部成功：

- batched L1 vs batched L2：**8/8 output-token 序列相同**。
- 再次 batched L1 vs batched L2：**8/8 相同**。
- 串行 L1 vs batched L1：**3/8 相同**。

因此，批次变化的数值差异可以在 L1 内复现；该控制样本中，固定批次下的 L2 搬运没有新增输出差异。这个结果不证明所有 batch/模型/硬件都 bitwise 稳定，也不证明 INT8 相比 BF16 无量化误差。

##### KDA AWQ 的独立兼容层

当前 main 的 Kimi AWQ 加载仍需要前文已验证的 3 处模型兼容调整：shared-expert prefix、MoE prefix，以及 packed module mapping。本次通过实验目录里的单模型 import overlay 应用，配合独立 tokenizer compatibility 目录。

这些调整没有写入 002 工作树。补丁保存在 `kda_compatibility_only.patch`，有效模型代码为 `kimi_linear_awq_compat.py`，每轮 launch manifest 记录 overlay 路径。因此 KDA 结果是 **002 + 独立 AWQ 兼容层** 的验证，不能描述为未调整 main 的 AWQ 开箱即用。

#### 范围和后续

首版只开放 CUDA、单机 TP、BF16 active state、kernel/page_first/write_through/cache/extra_buffer_lazy。拒绝 L3、PD、ReplaySSM、speculative、unified-memory、host-only checkpoint 以及额外模型 side state；未宣称验证其他硬件或并行方式。

001 的通用失败回滚和 pin handoff 修复未包含，应继续独立合入并在最终上游基线上复验。003 的融合优化未包含，002 所有结果都来自原 codec。

当前代码和本地验证已可供独立评审；本次没有推送或创建 002 PR。没有为 002 声称吞吐、TTFT 或显存峰值改善。

#### 复查材料

- `002-main.patch`、`source_state.json`：独立源码改动、基线、分支和隔离检查。
- `validation.json`：总体结果、边界、JUnit 元数据。
- `exhaustion_before.log`：未修复 main 的失败结果。
- `cpu_regressions.log/xml`、`gpu_regressions.log/xml`、`rust_mamba.log/xml`：回归。
- `precommit_final.log`：最终 lint/格式检查。
- `SERVICE_RESULTS.json`、`GDN_BATCH_CONTROL.json`：真实服务结果。
- 各服务目录：launch、server info、inputs、responses、audit 和 server log。
- `run_service.py`、`audit/sitecustomize.py`：仅用于实验的驱动和校验钩子。


## 量化权重模型精度完整报告


### GDN/KDA checkpoint 精度对比：BF16、INT8 与 L2 回迁

源码：独立 002 提交 `15a00b66a6b388c5c795d533e4ef21a6ee92ff5e`；001、003 未包含。

#### 主要结果

- GDN：BF16 为 1277/1319，INT8 为 1276/1319；净少 1 题，由 7 道对→错和 6 道错→对组成。准确率差为 −0.076 个百分点，配对 95% 区间为 [−0.607, +0.455] 个百分点。
- GDN 的量化差异可测：平均 |Δlogprob|=0.0677119，参考答案 token 加权 ΔNLL=+0.00192228。准确率接近不等于量化无损。
- GDN 在 BF16、INT8 两种格式内，L1/L2 都是 1319/1319 输出序列完全一致，逐 token logprob 差为 0；本测试未观察到 L2 新增影响。
- KDA 四组准确率为 85.140%–86.050%，所有配对准确率差的 95% 区间均包含 0。默认配置已有复算波动，量化对照和 L1/L2 对照的 logprob 差处在相近量级；不能把这些差异直接解释成量化收益或搬运损伤。
- 64 题流程检查及完整阶段的逐请求路径校验全部通过，两张卡均实际执行了 L2 回迁。全部原始数据已完成核对，002 工作树保持干净。

#### 实验配置

- GDN：Qwen3.8-27B-NVFP4；KDA：Kimi-Linear-48B-A3B-Instruct-AWQ-4bit。模型权重量化保持各自原格式，只改变 checkpoint 格式和恢复来源。
- 两张 RTX 5090，TP2，每次只运行一个服务，batch/并发固定为 1；active state 为 BF16。
- 使用模型各自固定的原生 chat template、8 个训练集示例，关闭 Qwen thinking 提示开关；贪心解码，最多生成 1024 tokens。
- GSM8K 完整 1319 道测试题，固定随机顺序；64 题流程检查通过后，完整阶段补齐其余 1255 题。
- 任务生成与参考答案 teacher forcing 分别清理缓存、预热。每次只恢复一次 checkpoint，随后以 BF16 继续计算。
- checkpoint 切点按 256 tokens 对齐，GDN 为 1024/1280、KDA 为 768/1024；恢复后剩余提示词为 32–287 tokens。结论适用于这些固定切点和单次恢复。
- L2 通过生产备份完成与 GPU 驱逐接口触发，保留 host 数据；逐请求检查命中长度和两个 TP rank 的实际搬运。
- KDA 使用与此前相同的独立 AWQ 模型兼容层；默认 Marlin 设置在全部正式 KDA 组保持一致。

数据源：[GSM8K 官方仓库](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data)。数据版本、文件哈希和全部 token IDs 保存在 `protocol.json`、`inputs_*.jsonl`。

#### 任务准确率

| 模型 | checkpoint / 来源 | 正确数 | 准确率 | 95% Wilson 区间 | 截断数 | 无可解析答案数 |
|---|---|---:|---:|---|---:|---:|
| GDN | bf16_l1 | 1277/1319 | 96.816% | [95.724%, 97.636%] | 6 | 6 |
| GDN | bf16_l2 | 1277/1319 | 96.816% | [95.724%, 97.636%] | 6 | 6 |
| GDN | int8_l1 | 1276/1319 | 96.740% | [95.638%, 97.571%] | 6 | 6 |
| GDN | int8_l2 | 1276/1319 | 96.740% | [95.638%, 97.571%] | 6 | 6 |
| KDA | bf16_l1 | 1124/1319 | 85.216% | [83.198%, 87.029%] | 0 | 73 |
| KDA | bf16_l2 | 1135/1319 | 86.050% | [84.075%, 87.815%] | 2 | 73 |
| KDA | int8_l1 | 1133/1319 | 85.898% | [83.916%, 87.673%] | 1 | 66 |
| KDA | int8_l2 | 1123/1319 | 85.140% | [83.119%, 86.958%] | 0 | 75 |

主指标按固定 `#### <number>` 规则提取并规范化数值，所有题都计入。达到长度上限或答案格式错误是测量结果，没有因这些结果换题、提高某组预算或剔除样本。

#### 配对变化与 logprob

| 模型 | 候选 − 基线 | 准确率差（百分点） | 配对 bootstrap 95% | 对→错 / 错→对 | 输出序列完全相同 | mean\|Δlogprob\| | max\|Δlogprob\| | ΔNLL（token 加权） |
|---|---|---:|---|---:|---:|---:|---:|---:|
| GDN | int8_l1 − bf16_l1 | -0.076 | [-0.607, +0.455] | 7 / 6 | 287/1319 | 0.067711868 | 6.0202162 | +0.0019222778 |
| GDN | bf16_l2 − bf16_l1 | +0.000 | [+0.000, +0.000] | 0 / 0 | 1319/1319 | 0 | 0 | -0 |
| GDN | int8_l2 − int8_l1 | +0.000 | [+0.000, +0.000] | 0 / 0 | 1319/1319 | 0 | 0 | -0 |
| GDN | int8_l2 − bf16_l2 | -0.076 | [-0.607, +0.455] | 7 / 6 | 287/1319 | 0.067711868 | 6.0202162 | +0.0019222778 |
| KDA | int8_l1 − bf16_l1 | +0.682 | [-0.531, +1.895] | 27 / 36 | 735/1319 | 0.029402406 | 1.7741857 | +0.00022410833 |
| KDA | bf16_l2 − bf16_l1 | +0.834 | [-0.303, +1.971] | 24 / 35 | 770/1319 | 0.026968866 | 1.3607063 | +0.00014191779 |
| KDA | int8_l2 − int8_l1 | -0.758 | [-1.971, +0.379] | 36 / 26 | 733/1319 | 0.027528461 | 1.4584904 | -0.00023095373 |
| KDA | int8_l2 − bf16_l2 | -0.910 | [-2.123, +0.303] | 41 / 29 | 727/1319 | 0.029517313 | 1.4911065 | -0.00014876319 |

所有 logprob 比较都对齐相同的参考答案 token，并检查每个返回 token ID；不是比较两份不同生成文本的概率。单位为自然对数。ΔNLL>0 表示候选组对参考答案的平均打分更差。分位数、变化题号、最大差异题目和 McNemar 检验保存在 `RESULTS.json`。

#### 64 题流程检查与 L2 验证

正式的四个 smoke 服务均完成 64 题 × 两条路径 × 两类请求。输入长度、checkpoint 边界、缓存报告、两张卡的 H2D/驱逐记录和参考答案 token 对齐均通过。
BF16 temporal/conv 以及 INT8 qdata/scale/conv 的实际搬运在 smoke 阶段逐元素比较一致。完整阶段关闭额外逐项同步检查，保留路径审计；所有完整服务必须通过逐请求的 L1/L2 校验才会写 DONE。

#### KDA 的重复计算背景

默认 KDA 配置在相同输入、相同 L1 路径重复计算时已存在差异。8 个诊断输入、3 次重复的对照中：
- 默认设置，l1→l1 重复对照：输出完全相同 4/8；最大 |Δlogprob|=0.46664155。
- 默认设置，l1→l2 重复对照：输出完全相同 4/8；最大 |Δlogprob|=0.36788034。
- 默认设置，l2→l2 重复对照：输出完全相同 4/8；最大 |Δlogprob|=0.43759775。
- 关闭 atomic 的诊断设置，l1→l1 重复对照：输出完全相同 4/8；最大 |Δlogprob|=0.62273479。
- 关闭 atomic 的诊断设置，l1→l2 重复对照：输出完全相同 5/8；最大 |Δlogprob|=0.46403384。
- 关闭 atomic 的诊断设置，l2→l2 重复对照：输出完全相同 4/8；最大 |Δlogprob|=0.48197436。

不同预热运行产生的源状态哈希已经不同，而实际 BF16 source→active copy 和 host 往返均通过逐元素校验。关闭 atomic 的诊断没有解决复算波动，因此没有把该修改纳入正式协议。诊断服务和原始结果独立保存，不混入 1319 题主表。
因此，KDA 主表给出的是固定配置下的观测差异。不能把其 L1/L2 输出或 logprob 差异全部归因于搬运，也不能把准确率的少量上升直接解释为量化改善模型。完整服务重启时还单独复跑了最初 8 题，结果在各 full 目录的 restart_comparison_epoch*.json 中。

#### 如何解释结果

- INT8-L1 对 BF16-L1 用于评估 checkpoint 量化；BF16-L2 对 BF16-L1、INT8-L2 对 INT8-L1 用于检查 L2 之后的额外差异。
- INT8 的状态量化本身有数值误差；任务正确数相同不能称为量化无损。L2 字节一致也不代表任意批次、模型和硬件的输出都 bitwise 稳定。
- bootstrap 区间以题目为单位。如果所有题的正确性都一致，bootstrap 区间退化到 0，仅描述已观察到的数据，不证明不存在未观察到的退化；`RESULTS.json` 另记录零观测翻转时的单侧比例上界。
- 这里评估的是固定 few-shot 提示、固定 checkpoint 切点和 1024-token 预算下的 GSM8K，不外推到其他任务、较远/较近切点或多次量化恢复。

#### 文件

- `RESULTS.json`：完整统计、全部分位数、翻转题号和最大差异样本。
- `ANSWER_FLIPS.csv`：逐题正确性翻转、两组答案、截断/格式状态和 ΔNLL。
- `FINAL_VALIDATION.json`：最终记录数、唯一性、源码、输入哈希和路径检查。
- `protocol.json`、`inputs_gdn.jsonl`、`inputs_kda.jsonl`：冻结协议和输入。
- `SMOKE_GATE_PASSED.json`：64 题流程闸门。
- `runs/{smoke,full}_{gdn,kda}_{bf16,int8}/`：launch、server info、原始输出、逐 token logprob、路径审计和重启对照。
- `KDA_REPEATABILITY*.json`、`calibration_notes.json`：背景波动及未纳入正式设置的诊断。
- 生成驱动、校验钩子和统计程序全部位于实验目录；002 工作树保持独立且干净。


## BF16 权重模型精度完整报告


### BF16 权重模型的 INT8 checkpoint 精度对比

模型：`/mnt/sll/models/Qwen3.8-27B`。已检查 safetensors 中全部 1199 个张量为 BF16，没有权重量化配置。

使用与此前 NVFP4 权重实验完全相同的 GSM8K 1319 道题、8-shot 模板、顺序、参考答案 token 和切分位置。先验证 64 题，再评估其余 1255 题；全部计入结果。

TP2、并发 1；活动 Mamba state 为 BF16；checkpoint 为 BF16 或 INT8；KV cache 保持 FP8 E4M3。固定活动槽位 32、INT8 checkpoint 槽位 16、KV 容量 32768 tokens。mem_fraction_static=0.94。独立 002 源码 `15a00b66a6b388c5c795d533e4ef21a6ee92ff5e`，未加入 001/003。

每次生成和参考答案打分前独立 flush/warm；L2 通过生产驱逐与回迁路径触发。逐请求检查命中长度、设备/主机来源、无 retraction、双 rank 传输。64 题阶段额外逐位检查传输数据。

| checkpoint / 命中层 | 正确数 | 准确率 | 截断数 | 答案格式错误数 |
|---|---:|---:|---:|---:|
| bf16_l1 | 1281/1319 | 97.1190% | 8 | 8 |
| bf16_l2 | 1281/1319 | 97.1190% | 8 | 8 |
| int8_l1 | 1282/1319 | 97.1948% | 5 | 5 |
| int8_l2 | 1282/1319 | 97.1948% | 5 | 5 |

| 比较：后者减前者 | 准确率变化（百分点） | 配对 bootstrap 95% CI | 正确→错误 / 错误→正确 | 平均绝对 Δlogprob | token 加权 ΔNLL | 输出完全相同 |
|---|---:|---|---|---:|---:|---:|
| bf16_l1 → int8_l1 | +0.0758 | [-0.3791, +0.5307] | 4 / 5 | 0.01236808 | +0.00122037 | 859/1319 |
| bf16_l1 → bf16_l2 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 | 0.00000000 | -0.00000000 | 1319/1319 |
| int8_l1 → int8_l2 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 | 0.00000000 | -0.00000000 | 1319/1319 |
| bf16_l2 → int8_l2 | +0.0758 | [-0.3791, +0.5307] | 4 / 5 | 0.01236808 | +0.00122037 | 859/1319 |

#### 与此前 NVFP4 权重实验的对照

| 权重精度 | BF16 checkpoint L1 准确率 | INT8 checkpoint L1 准确率 | INT8 − BF16（百分点） | 平均绝对 Δlogprob | token 加权 ΔNLL |
|---|---:|---:|---:|---:|---:|
| NVFP4（历史） | 96.8158% | 96.7400% | -0.0758 | 0.06771187 | +0.00192228 |
| BF16（本轮） | 97.1190% | 97.1948% | +0.0758 | 0.01236808 | +0.00122037 |

两轮使用相同题目和 token，但权重格式会选择不同计算内核，静态显存预留比例也不同。因此可比较各自内部的 checkpoint 精度差异，不能把跨轮差异全部归因于 4bit 是否掩盖了 INT8 误差。一次数据集评测也不能证明量化无损。

#### 重启重复性与显存

完整阶段在独立服务重启后复测首 8 题，记录如下：

```json
{
  "smoke_gdn_bf16": {
    "checks": 0,
    "equal_outputs": 0,
    "max_abs_logprob": 0
  },
  "smoke_gdn_int8": {
    "checks": 0,
    "equal_outputs": 0,
    "max_abs_logprob": 0
  },
  "full_gdn_bf16": {
    "checks": 32,
    "equal_outputs": 16,
    "max_abs_logprob": 0.0
  },
  "full_gdn_int8": {
    "checks": 32,
    "equal_outputs": 16,
    "max_abs_logprob": 0.0
  }
}
```

每 2 秒采样的整卡 NVML 已用显存峰值（MiB；非 PyTorch 精确分配峰值，可能漏掉瞬时峰值）：

```json
{
  "between_runs": {
    "0": 625,
    "1": 6
  },
  "smoke_gdn_bf16": {
    "0": 31308,
    "1": 30691
  },
  "smoke_gdn_int8": {
    "0": 32000,
    "1": 31383
  },
  "full_gdn_bf16": {
    "0": 31455,
    "1": 30838
  },
  "full_gdn_int8": {
    "0": 31964,
    "1": 31676
  }
}
```

完整数值：`RESULTS.json`；逐题答案翻转：`ANSWER_FLIPS.csv`；审计：`FINAL_VALIDATION.json`；配置与来源：`protocol.json`。


## 正式性能实验完整报告


### INT8 Mamba state 接入 HiCache L2：等容量对比

模型：`/mnt/sll/models/Qwen3.8-27B`，BF16 权重；独立 002 `15a00b66a6b388c5c795d533e4ef21a6ee92ff5e`，原 eager codec，无 001/003。两组都开启 L2。

#### 对齐方式

每个 TP rank 的 L1 树节点缓存状态上限为 8 槽，活动状态物理池为 32 槽。使用同一实验侧 admission/load 配额，通过生产 LRU 驱逐实现；锁定状态无法回收时跳过缓存或拒绝本次回迁，并记录次数。配额只限制树节点中的缓存状态，不把请求运行状态算成额外缓存。

Mamba L2 预算为每 rank 2693 MiB，包括 temporal、conv、scale。按完整状态向下取整；实际 payload 与不能容纳完整槽位的预算余量分别记录。KV L1 为 65536 tokens，KV L2 为 131072 tokens。

```json
{
  "r0_bf16": {
    "l1_quota": 8,
    "mamba": {
      "budget_bytes": 2824077312,
      "allocated_bytes": 2824077312,
      "bytes_per_slot": 39223296,
      "slots": 72,
      "unused_budget_bytes": 0
    },
    "kv": {
      "tokens": 131072,
      "allocated_bytes": 2147483648
    }
  },
  "r0_int8": {
    "l1_quota": 8,
    "mamba": {
      "budget_bytes": 2824077312,
      "allocated_bytes": 2807562240,
      "bytes_per_slot": 20643840,
      "slots": 136,
      "unused_budget_bytes": 16515072
    },
    "kv": {
      "tokens": 131072,
      "allocated_bytes": 2147483648
    }
  },
  "r1_bf16": {
    "l1_quota": 8,
    "mamba": {
      "budget_bytes": 2824077312,
      "allocated_bytes": 2824077312,
      "bytes_per_slot": 39223296,
      "slots": 72,
      "unused_budget_bytes": 0
    },
    "kv": {
      "tokens": 131072,
      "allocated_bytes": 2147483648
    }
  },
  "r1_int8": {
    "l1_quota": 8,
    "mamba": {
      "budget_bytes": 2824077312,
      "allocated_bytes": 2807562240,
      "bytes_per_slot": 20643840,
      "slots": 136,
      "unused_budget_bytes": 16515072
    },
    "kv": {
      "tokens": 131072,
      "allocated_bytes": 2147483648
    }
  },
  "r2_bf16": {
    "l1_quota": 8,
    "mamba": {
      "budget_bytes": 2824077312,
      "allocated_bytes": 2824077312,
      "bytes_per_slot": 39223296,
      "slots": 72,
      "unused_budget_bytes": 0
    },
    "kv": {
      "tokens": 131072,
      "allocated_bytes": 2147483648
    }
  },
  "r2_int8": {
    "l1_quota": 8,
    "mamba": {
      "budget_bytes": 2824077312,
      "allocated_bytes": 2807562240,
      "bytes_per_slot": 20643840,
      "slots": 136,
      "unused_budget_bytes": 16515072
    },
    "kv": {
      "tokens": 131072,
      "allocated_bytes": 2147483648
    }
  }
}
```

#### 工作负载与计时

服务 --hicache-ratio=2、--max-running-requests=8。KV L2 实际为 KV L1 的 2 倍；Mamba L2 独立预算精确等于 72 个完整 BF16 状态的大小，同预算向下取整容纳 136 个 INT8 状态。

1K/N96 的 KV 需求为 98304 tokens，小于 131072-token KV L2，同时状态工作集超过 BF16 的 72 槽、落在 INT8 的 136 槽以内。1K/N160 同时超过 KV 与两种 Mamba 容量，只作为整体缓存溢出压力，不能把全部差异单独归因于 Mamba。2K/4K N20 两组状态都放得下，用于检查容量充裕时的开销。

正式并发为 8。容量主实验使用 1K 前缀和 48/96/160 个前缀的工作集；另保留 2K/4K 前缀、工作集 20 的等命中对照。每种配置各测固定轮转与固定种子的 Zipf 热点访问，共 10 个场景、每格式 3 次独立服务重启。64-token 不同查询后缀，固定生成 32 tokens（ignore_eos）；前缀在第一个 256-token page 内不同。

每个场景独立清空缓存、预热，再测量。两组及所有重复使用相同请求数量、前缀、每个 worker 的访问序列；并发下服务实际调度交错可能不同。请求数由独立 pilot 统一冻结，逐场景列出实际计时长度。启动、预热、原始数据序列化和额外校验不计入计时。

两组保持原模型配置、MRoPE 和文本计算路径，仅在实验启动时将未使用的视觉模块放到 CPU，为对齐后的 KV 池留出空间；非视觉权重全部保留在 GPU，计时中不进行文本权重 offload。为避免长前缀产生额外的中间 chunk checkpoint，两组均关闭 chunked prefill，prefill batch token 上限为 8192。prefill CUDA Graph 关闭，decode CUDA Graph 最大 batch 为 8。没有强制驱逐指定目标、篡改状态或注入命中；运行时没有增加逐项传输同步。

TTFT 为客户端流式请求至首次非空输出事件。吞吐按整段请求完成墙钟时间计算。重算量为前缀长度减实际复用 token 数；另记录模型实际执行的 prefill token 数。Mamba L2 恢复依据双 rank 生产回迁记录，不能只用 attention KV 的 host/device token 报告代替。

#### 结果

下表为三轮各自统计值的均值；完整原始每轮数据和比值范围见 RESULTS.json。吞吐变化为三轮配对比值的均值。

| 场景 | BF16 / INT8 L2 恢复率 | BF16 / INT8 状态未命中率 | BF16 / INT8 每请求前缀重算 tokens | BF16 / INT8 TTFT P50（ms） | BF16 / INT8 请求/s | 吞吐变化 |
|---|---|---|---|---|---|---|
| p1024_n48_cyclic_c8 | 100.0% / 100.0% | 0.0% / 0.0% | 0.0 / 0.0 | 227.7 / 194.6 | 9.41 / 9.18 | -2.4% |
| p1024_n48_zipf_c8 | 51.0% / 51.4% | 0.0% / 0.0% | 0.0 / 0.0 | 239.4 / 210.8 | 9.31 / 9.59 | +3.0% |
| p1024_n96_cyclic_c8 | 0.0% / 100.0% | 100.0% / 0.0% | 1024.0 / 0.0 | 1295.6 / 223.7 | 4.01 / 9.38 | +133.9% |
| p1024_n96_zipf_c8 | 48.7% / 55.9% | 6.7% / 0.0% | 68.1 / 0.0 | 255.3 / 211.7 | 8.62 / 9.61 | +11.5% |
| p1024_n160_cyclic_c8 | 0.0% / 0.0% | 100.0% / 100.0% | 1024.0 / 1024.0 | 1294.9 / 1301.4 | 4.02 / 3.98 | -1.2% |
| p1024_n160_zipf_c8 | 48.0% / 59.6% | 15.5% / 3.8% | 158.9 / 38.8 | 361.9 / 219.0 | 7.74 / 9.06 | +17.0% |
| p2048_n20_cyclic_c8 | 100.0% / 100.0% | 0.0% / 0.0% | 0.0 / 0.0 | 231.8 / 230.7 | 9.34 / 9.29 | -0.5% |
| p2048_n20_zipf_c8 | 29.7% / 29.4% | 0.0% / 0.0% | 0.0 / 0.0 | 222.0 / 207.2 | 9.37 / 9.37 | +0.1% |
| p4096_n20_cyclic_c8 | 100.0% / 100.0% | 0.0% / 0.0% | 0.0 / 0.0 | 241.2 / 244.8 | 9.16 / 9.05 | -1.2% |
| p4096_n20_zipf_c8 | 30.7% / 30.5% | 0.0% / 0.0% | 0.0 / 0.0 | 229.0 / 213.8 | 9.35 / 9.47 | +1.3% |

#### 验证与限制

L1 上限在每次插入/回迁后断言，启动时检查双 rank 实际 L2 槽位和字节数。所有计时请求必须生成完整 32 tokens、无错误和 retraction。计数、槽位、自然命中路径及 admission skip/load defer 保留原始日志。

该配额与独立预算配置属于实验驱动，不包含于生产 002，也不代表相同原生启动参数即可得到同样分配。BF16 与 INT8 仍使用各自生产恢复和编解码路径；不同的请求状态池关系、pin 行为和编码开销是实际方案的一部分。若配额跳过或回迁拒绝存在，应结合次数解释命中差异。

L2 预算也容纳 L1 数据的 host backup，并非 L1 槽位数加 L2 槽位数的完全独立容量。固定轮转可能放大容量边界的收益，因此同时保留热点访问和两边均溢出的结果。端到端差异不等同于传输 kernel 加速；没有对 decode 质量作新评估，精度结论参见之前独立 GSM8K 实验。

这里报告单机双 RTX 5090、固定输出和服务配置的结果，不外推任意并发、长输出、其他模型和硬件。原始 GPU 显存、功耗、时钟、温度采样保留在各 run 的 gpu.jsonl 中。


## 等 72 槽内存与连续恢复精度完整报告


### 002 补充实验：相同 72 槽内存与连续 checkpoint 精度

源码为独立 002 `15a00b66a6b388c5c795d533e4ef21a6ee92ff5e`，生产工作树保持干净，001/003 均未混入。所有模型沿用原 eager codec。

#### 主要结论

- **同样缓存 72 个前缀，GDN 的 Mamba L2 主机内存每卡从 2.630 GiB 降到 1.384 GiB，减少 47.37%。** 三轮的实际 buffer 字节和构建 host pool 时新增的锁页内存均与该数值一致，两组每轮均完成 72/72 个 L2 回迁。双卡合计节省约 2.49 GiB。KDA 单轮补充检查为 0.753 → 0.406 GiB/卡，减少 45.99%。这是 Mamba L2 池的节省。
- **GDN 的 INT8 数值差异随链式恢复深度增加。** 在同一深度对比 BF16 与 INT8，平均绝对 Δlogprob 从 1 次的 0.01342 增至 32 次的 0.02396，采样层状态相对误差从 1.99% 增至 2.44%；参考 token 的 top-1 一致率从 99.44% 降至 99.16%。32 次时两组均为 31/32 题正确，逐题对错一致，但生成序列仅 19/32 完全相同。BF16 在 32 次分段下与完整重算也有 0.01257 的平均绝对 Δlogprob，因此不能把跨深度变化全部归因于量化误差累积。
- **GDN 未观察到 L2 回迁引入额外差异。** 在每个深度和每种 checkpoint 格式内，L1/L2 均为 32/32 生成序列一致、逐 token logprob 差为 0、所采样 temporal state 差为 0。KDA 同格式 L2/L1 的平均绝对 Δlogprob 约 0.0265–0.0297，与同路径重复运行的 0.0257–0.0322 同量级，不能从现有噪声中分离出 L2 的独立影响。

#### 相同 72 槽的实测主机内存

两组 L1 缓存配额均为 8，active 物理池为 32，KV L1/L2 容量相同；INT8 使用独立的 16 槽物理 checkpoint 池。两组 Mamba host pool 均恰好 72 个可用槽位，启动时触及全部未使用 buffer 页，再实际填满并回迁 72 个不同 1K 前缀。

| 模型 | 独立配对次数 | BF16 host payload / rank | INT8 host payload / rank | 节省 / rank | 减少 |
|---|---:|---:|---:|---:|---:|
| gdn | 3 | 2.630127 GiB | 1.384277 GiB | 1.245850 GiB | 47.368% |
| kda | 1 | 0.752563 GiB | 0.406494 GiB | 0.346069 GiB | 45.985% |

Payload 从实际唯一 tensor storage 统计，包含 temporal、conv 和 scale；同时核对完整状态大小估算。每个 rank 的前后 VmRSS/VmLck、服务进程组 RSS/PSS/锁页数，以及 GPU 采样单独保存。上述百分比仅描述 Mamba L2 payload，不能解释为整进程 RAM 或 GPU 显存减少同样比例。GDN 做三轮配对；KDA 的单轮来自路径预跑，作为补充而非三轮内存基准。

#### 固定 8K 最终上下文的链式精度

每个模型使用固定 32 道 GSM8K 测试题，在原 8-shot 提示的 system 消息内加入固定训练集参考笔记，使 checkpoint 边界统一为 8192 tokens；所有深度/格式/层级使用完全相同的最终 prompt 与参考答案 token。不能直接与此前短提示下的 1319 题准确率比较。

深度为 1/4/16/32。每一轮恢复前一个状态，继续计算固定的下一段 token，再缓存新状态；最终评估再恢复最后一个状态。扩展过程中生成的 1 token 不进入下一次输入，确保比较输入一致。D−1 次中间恢复加最终 1 次，构成 D 次有效串联恢复。真实 codec store/load 次数、双 rank H2D/路径和 cached token 长度全部核验。

生成与参考答案打分分别清空缓存、独立构建链。L2 组在每次继续前通过生产 backup-completion/eviction 路径制造受控回迁，并等待传输；这是正确性实验，不计恢复性能。另做无缓存完整重算基线，用于区分 BF16 分段执行/复制本身的数值影响。

GDN 使用未量化 BF16 权重、1 次正式完整运行，另有独立 pilot 的重复对照；KDA 使用原 AWQ 4bit 及相同独立兼容层，正式运行 3 次以测量已知复算背景。两者均 TP2、客户端并发 1、原 overlap 设置和贪心生成，上限 1024 tokens，所有截断/格式失败均保留。

下表比较每个深度下的 INT8 与 BF16 checkpoint。数值为各重复统计的均值；KDA 同题重复不作为额外独立题目。状态指标只覆盖第一/中间/最后三个 state 层的全部 heads/channels，在最终恢复后、后续 forward 前采样，不是全状态误差上界。

| 模型 | 恢复路径 | 有效恢复次数 | BF16 / INT8 正确数（每次32题，跨重复均值） | 平均绝对 Δlogprob | 参考 token top-1 一致率 | sampled state 相对误差均值 | 同格式 L2/L1 平均绝对 Δlogprob |
|---|---|---:|---|---:|---:|---:|---|
| gdn | l1 | 1 | 31.00 / 32.00 | 0.01341501 | 99.437% | 0.01987141 | 0.00000000 / 0.00000000 |
| gdn | l1 | 4 | 31.00 / 31.00 | 0.01496647 | 99.331% | 0.02102064 | 0.00000000 / 0.00000000 |
| gdn | l1 | 16 | 31.00 / 32.00 | 0.02005887 | 99.261% | 0.02308269 | 0.00000000 / 0.00000000 |
| gdn | l1 | 32 | 31.00 / 31.00 | 0.02395741 | 99.156% | 0.02443693 | 0.00000000 / 0.00000000 |
| gdn | l2 | 1 | 31.00 / 32.00 | 0.01341501 | 99.437% | 0.01987141 | 0.00000000 / 0.00000000 |
| gdn | l2 | 4 | 31.00 / 31.00 | 0.01496647 | 99.331% | 0.02102064 | 0.00000000 / 0.00000000 |
| gdn | l2 | 16 | 31.00 / 32.00 | 0.02005887 | 99.261% | 0.02308269 | 0.00000000 / 0.00000000 |
| gdn | l2 | 32 | 31.00 / 31.00 | 0.02395741 | 99.156% | 0.02443693 | 0.00000000 / 0.00000000 |
| kda | l1 | 1 | 28.00 / 28.33 | 0.02954997 | 98.878% | 0.04030680 | 0.02726897 / 0.02790498 |
| kda | l1 | 4 | 28.00 / 27.67 | 0.03039181 | 98.975% | 0.04305491 | 0.02701746 / 0.02865948 |
| kda | l1 | 16 | 28.67 / 29.00 | 0.03290916 | 98.892% | 0.04797693 | 0.02726643 / 0.02882951 |
| kda | l1 | 32 | 29.00 / 28.00 | 0.03581238 | 98.767% | 0.05549102 | 0.02650047 / 0.02966223 |
| kda | l2 | 1 | 28.00 / 27.33 | 0.03054771 | 98.712% | 0.03996515 | 0.02726897 / 0.02790498 |
| kda | l2 | 4 | 28.67 / 28.33 | 0.03025513 | 99.100% | 0.04329027 | 0.02701746 / 0.02865948 |
| kda | l2 | 16 | 28.33 / 27.33 | 0.03440534 | 98.560% | 0.04846866 | 0.02726643 / 0.02882951 |
| kda | l2 | 32 | 27.33 / 27.33 | 0.03550386 | 98.851% | 0.05615459 | 0.02650047 / 0.02966223 |

#### 重复性与解释范围

同深度 INT8−BF16 是主要量化对照；同格式 L2−L1 用于检查搬运新增影响；BF16 chain−cold 用于检查分段/复用本身的误差。KDA 另保存相同格式、路径、深度在独立运行间的 logprob/top-1/state 噪声，避免把背景波动解释成误差累积或 L2 损伤。

32 题任务准确率只作辅助观察，不能证明量化无损、统计等价或所有长对话无退化。不能仅凭 state 相对误差平稳就断言全模型误差有理论上界；也不能把不同深度的变化全部归因于量化，因为分段长度随深度改变，BF16 对照同样受影响。

原始数据、每题翻转和全部重复对照见 RESULTS.json 与各服务 chain_results.jsonl / states / audit。之前的完整 1319 题精度测试，以及72→136槽的三轮性能实验保持独立；本轮不改写此前结论。

相关既有报告：

- `/mnt/sll/experiments/int8-checkpoint-accuracy-bf16weights-20261103/REPORT_ZH.md`
- `/mnt/sll/experiments/int8-checkpoint-accuracy-20261103/REPORT_ZH.md`
- `/mnt/sll/experiments/int8-l2-benefit-72slots-20261104/REPORT_ZH.md`


## 原始本地材料索引

- 回归与真实服务：`/mnt/sll/experiments/int8-hicache-002-port-20261103/`
- 量化权重精度：`/mnt/sll/experiments/int8-checkpoint-accuracy-20261103/`
- BF16 权重精度：`/mnt/sll/experiments/int8-checkpoint-accuracy-bf16weights-20261103/`
- 正式性能：`/mnt/sll/experiments/int8-l2-benefit-72slots-20261104/`
- 等 72 槽内存与连续恢复精度：`/mnt/sll/experiments/int8-l2-followup-20261105/`
- 被替换的旧性能实验与诊断：`/mnt/sll/experiments/int8-l2-benefit-20261104/`

这些是本机目录。对外提交 PR 时，需要上传补充归档并替换为维护者可访问的链接。
