# QAMOO 项目复现

已完成环境配置、官方算法入口验证、采样数据导出和断点续算实现。按用户选择，**当前正在运行接近原论文预算的长任务，尚未全部完成**。

## 当前结果与进度

- 实时进度：`output/project_reproduction/PROGRESS.md`。
- 当前各任务数值：`output/project_reproduction/results.csv`。`running` 行仅代表中间结果，不能作为最终结果。
- 机器可读状态：`output/project_reproduction/progress.json`。
- 三目标随机对照已完成 25,000,000 个样本，HV=35194.132536。
- 新量子模拟样本的 1,000 条预览：`output/project_reproduction/qaoa_sample_preview.jsonl`。
- 此前完成的 9 次小预算模拟记录于 `OFFICIAL_REPRODUCTION.md`；它们与本次完整预算任务分开存储。

## 完整预算队列

| 实验 | 问题 | 每次预算 | 重复 |
|---|---|---|---|
| 官方 QAOA / Aer MPS，χ=50 | 42 节点、3 目标，p=3/4/5/6 | 5000 电路 × 5000 shots | 每个 p 五次 |
| 官方 QAOA / Aer MPS，χ=20 | 42 节点、4 目标，p=6 | 20000 电路 × 5000 shots | 五次 |
| 官方均匀随机采样 | 3 / 4 目标 | 2500 万 / 1 亿样本 | 各一次 |
| 官方 Goemans–Williamson | 3 / 4 目标 | 5000 权重 × 5000 随机切分 | 各一次 |
| 官方随机 ε-约束 | 3 / 4 目标 | 459629 / 742081 次求解 | 各一次 |
| 官方加权和 | 3 / 4 目标 | 各 400000 次求解 | 各一次 |

完整计划为 `expanded_plan.json`，共 33 个任务，其中 QAOA 合计 10 亿 shots。
这覆盖论文主要三目标 MPS 比较设置和四目标 MPS 设置；没有额外排入图 4 的全部键维数网格。
四个进程并行，每个 MPS/Gurobi 求解单线程。首次 p=6、χ=50 测速 10000 shots 用时约 31.79 秒，单次 2500 万 shots 的粗略外推约 22 小时；整个队列按天运行，实际速度随参数和系统负载变化。

## 运行、停止与恢复

项目根目录 PowerShell：

```powershell
# 在后台启动或从检查点恢复。一个项目只允许一个控制进程。
.\run_project.ps1 -Workers 4

# 查看当前状态。
.\.runtime\python310\python.exe project_reproduction.py status

# 当前批次完成后停止，保留所有已提交检查点。
.\run_project.ps1 -Stop

# 再次执行启动命令即可续算。
```

也可使用 `run_project.ps1 -Foreground` 前台运行。正在执行的任务依赖本机进程持续运行；关机后再次启动即可从最后完成的检查点恢复。
控制进程 PID 记录于 `output/project_reproduction/controller.json`，日志在同目录和 `logs/` 中。
重复启动会被独占锁拒绝，避免两个进程写同一个实验。任务失败会记录为 `failed`，不会伪装成完成；修复原因后重新启动可以续算。

## 数据格式与断点

每个任务目录为 `output/project_reproduction/runs/<任务ID>/`：

- `config.json`：固定预算、深度、键维数、种子和批大小。
- `batch_*.npz`：每批真实测量位串、起止电路/求解编号、批次种子和当时 HV。`raw_packed_bits` 每行六字节。
- `checkpoint_0.npz` / `checkpoint_1.npz`：交替保存完整 Pareto 前沿、代表性位串、原始位置及累计时间，不在每个历史批次重复保存完整前沿。
- `status.json`：本任务进度。
- `complete.json`：仅在全部预算完成后创建。

原始位串解码为节点 0–41 顺序：

```python
bits = numpy.unpackbits(raw_packed_bits, axis=1)[:, -42:][:, ::-1]
```

如需标准 JSONL：

```powershell
.\.runtime\python310\python.exe export_project_samples.py --run qaoa_3o_p6_chi50_r0 --output output/project_reproduction/my_samples.jsonl --limit 1000
```

`--front-only` 导出当前 Pareto 前沿，`--limit 0` 可导出全部现有样本。默认限制条数，避免将完整十亿样本展开为庞大的文本。
测量样本的三个或四个目标均按原图跨切分边权之和计算；编号从 0 开始。

断点提交顺序是先原子保存测量批次，再原子提交轮换前沿检查点。若在两步之间中断，则重放未提交批次。批种子由运行种子和起始编号确定，续算不重复累计已完成样本。

## 与作者代码的对应关系

- 原始源代码位于 `qamoo-main/qamoo/`，没有修改。
- 大预算模拟使用作者归档的 QPY 电路和完整参数绑定，重新调用 Qiskit Aer 生成测量；未把作者历史测量结果当成本次实验。
- 电路和绑定的来源及 SHA-256 记录在 `output/project_reproduction/inputs/*/manifest.json`。
- 经典算法直接调用作者 `random`、`weighted_sum`、`random_eps_constraint`、`goemans_williamson` 函数；本地适配仅切分权重序列、重定向输出、显式分批种子及指标后处理。
- Windows 无可用的 pygmo pip 包，因此显式兼容层仅替换作者的超体积 API。当前大预算任务采用 moocore 0.3.2 精确超体积和非支配筛选；早期少量批次使用过经核验的 pymoo/本地精确实现。
- 为支持确定性断点，经典随机数与量子测量均按批次派生种子；这不是作者原始运行的同一随机序列。QPY、电路参数、软件版本、批大小、种子均固定后才保证重放。
- 实际本机采样、算法和后处理耗时分别记录，不能把论文假定的 10000 shots/s 当成本机实际耗时。

## 验证记录

`output/project_reproduction/pipeline_validation.json` 记录：

1. 三目标和四目标非支配筛选与独立实现一致；三维 HV 与精确矩形并集实现一致，四维与 pymoo 一致。
2. 四个官方经典算法在三/四目标上共八个小规模入口验证通过。
3. 从部分检查点续算，测量位串与未中断运行逐位一致。
4. QAOA 旧检查点迁移、轮换检查点、六字节位串解码及指标重算验证通过。

此前还用作者归档的 1000 个测试样本核对了 10 个 HV 检查点，最大绝对误差约 5.09e-11。
Python 3.10.11 环境已通过 `pip check`，依赖快照为 `output/project_reproduction/requirements-lock.txt`。

## 尚未执行的部分

- IBM 真机实验：需要 IBM Quantum 访问条件，未提交任何付费硬件作业。
- 27 节点 JuliQAOA 参数训练：本次复用作者已训练角度，未重新训练。
- DCM / DPA-a：算法本体属于作者链接的另一个仓库，并涉及 CPLEX/Gurobi 环境；当前只保留其公开结果作为参考，没有声称重新执行。
- 完整 χ 网格和其他扩展实验没有列入当前 33 任务队列。

因此应将本交付称为“官方主要模拟与经典基线的本地扩大预算复现”，不能称为原论文全部实验已经独立复现，也不能在队列完成前将中间曲线视为最终量子优势证据。
