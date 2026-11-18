# 官方算法本地复现

**最新阶段：已启动扩大到论文采样预算的任务，尚未全部完成。** 请查看
`PROJECT_REPRODUCTION.md` 和 `output/project_reproduction/PROGRESS.md`。
下文保留的是此前已完成的小预算验证记录。

2026-10-20 已从用户下载的 `qamoo-main.zip` 提取作者源代码和输入数据，并完成真实 Aer MPS 模拟。
这部分与早期的 `analysis.ipynb` 源数据重绘不同：本次生成了新的测量样本。

## 已完成的实验

- 作者原始 42 节点、三目标 MO-MAXCUT 实例，问题编号 0。
- 直接调用作者 `prepare_qaoa_circuits`、`transpile_qaoa_circuits_parametrized`。
- 使用作者在 27 节点问题上预训练的 JuliQAOA 角度，未重新训练角度。
- Qiskit 1.3.2 / Aer 0.16.0 / IBM Runtime 0.36.0 / Optimization 0.6.1。
- 本项目内独立 Python 3.10.11 便携环境，不改动系统 Python。
- MPS 键维数 chi=20，单线程；p=1、3、6，各使用种子 42、43、44。
- 每次 10 个电路 × 100 shots，共 9 次运行、9,000 个新的 42 位测量样本。
- 权重向量固定为作者三目标权重文件中的前 10 个，各设置共享；种子控制量子测量采样。
- 每个深度固定同一个已保存电路，三组种子共享电路和参数绑定；p=1、3、6 的种子 42 重放均逐位匹配原运行。

| 深度 p | 三次运行平均 HV | 样本标准差 | 平均非支配目标向量数 |
|---|---:|---:|---:|
| 1 | 20687.845390 | 146.325128 | 37.33 |
| 3 | 37764.397748 | 367.862384 | 110.33 |
| 6 | 40403.972900 | 207.581042 | 171.33 |

标准差为三个固定权重向量集合上的测量重复的描述统计，不是置信区间。
HV 使用作者的下界参考点。当前预算下较深电路的 HV 更高；尚未执行同预算经典基线，不能据此宣称量子优势。
这仅是小预算流程复现。论文每个设置采用 5,000 电路 × 5,000 shots，并重复五次；本次未运行该完整预算、真实硬件、参数训练或商业求解器。

## 输出

以下路径相对于项目根目录：

- `output/official_reproduction/mps_42q_3obj_fixed_circuits/summary.csv`：每次运行的 HV、非支配向量数、实际准备及采样耗时。此运行复用了保存电路，准备耗时为文件载入准备，不代表重新构建耗时。
- 同目录 `samples.jsonl`：逐条记录深度、种子、电路序号、42 位测量向量及三个目标值。
- 各 `p*_seed*` 子目录：`.npy` 测量样本与目标值、非支配样本、`.qpy` 电路、参数绑定、权重向量、HV 进度。
- `output/official_reproduction/problem_edges.csv`：所有已提取实例的加权边长表；每行包含问题、目标编号、节点和权重。
- `output/official_reproduction/problem_graphs.json`：42 个目标图的结构化输入。
- 同目录 `source_manifest.json`：输入 ZIP 和 144 个提取文件的 SHA-256。ZIP 未提供可验证的 Git commit，因此 commit 字段保留 null。
- 同目录 `requirements-lock.txt`：本次 Python 3.10 环境的依赖快照。

`bits_node_0_to_41[k]` 表示节点 k 的切分取值，对应 Qiskit 测量字符串反转后的顺序。
目标 j 为该目标图所有跨切分边的权重之和，目标值可以为负。
`circuit_index`、`shot_index`、目标编号及节点编号均从 0 开始；同一电路的所有 shots 连续保存。
MPS 为理想电路的近似模拟，包含键维数截断；本次没有注入硬件噪声。

## 重新运行

在项目根目录的 PowerShell 中执行：

```powershell
python prepare_official_data.py
.\.runtime\python310\python.exe validate_official_hv.py
.\.runtime\python310\python.exe run_official_mps.py --depths 1 3 6 --seeds 42 43 44 --tag another_run
```

若要重放本次固定电路，并获得逐位一致的种子采样：

```powershell
.\.runtime\python310\python.exe run_official_mps.py --depths 1 3 6 --seeds 42 43 44 --templates-from output/official_reproduction/mps_42q_3obj_fixed_circuits --tag replay_run
```

作者电路重新构建时，部分可交换门的排列可发生变化，MPS 截断近似可能受排列影响。
因此仅固定采样种子不足以保证从头重建后的结果逐位一致；精确重放应同时固定 QPY 电路、参数绑定、软件版本与采样种子。
`mps_42q_3obj_3seeds` 保留首次运行；`seed_reproducibility_check` 保留从头重建电路的诊断结果，不作为本次固定电路对照实验。

`--tag` 必须使用新目录名，避免覆盖历史运行。可调整 `--circuits`、`--shots`、`--chi`，扩大预算会增加时间和内存。
若在另一台电脑重建环境，使用 Python 3.10，安装 `output/official_reproduction/requirements-lock.txt` 中的依赖；无需安装商业求解器及 pygmo 才能运行本脚本。

## 指标与来源验证

Windows 环境下，本脚本用本地精确三维矩形并集算法计算 HV，未调用作者依赖的 pygmo。
以容斥公式独立核对重叠矩形计算，并使用压缩包内作者 p=1 测试样本核对 10 个 HV 检查点，最大绝对误差 5.09e-11。
该验证仅使用历史样本核对指标，历史样本没有混入新模拟输出。详细记录见同目录 `hv_validation.json`。
电路源码未修改；本地适配仅包括结果目录、显式测量种子和指标后处理。

官方来源：https://github.com/stefan-woerner/qamoo
