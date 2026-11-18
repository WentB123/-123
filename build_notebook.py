"""Build and execute the source-data reanalysis notebook."""
import json
import os
from pathlib import Path
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
# Keep runtime caches inside the project, including in restricted environments.
os.environ["IPYTHONDIR"] = str(ROOT / ".jupyter" / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter" / "runtime")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".jupyter" / "matplotlib")
for key in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md("""
# 量子近似多目标优化：公开数据复现与分析

**范围：论文公开源数据的图表与数值复现；不包含量子硬件实验重跑。**

文献：Kotil et al., *Quantum approximate multi-objective optimization*,
Nature Computational Science 5, 1168–1177 (2025).
DOI: 10.1038/s43588-025-00873-y。

本项目下载出版社图 2–5 及扩展图 1 的源数据，运行本 notebook 完成二次分析。
作者原始绘图 notebook 另存于 `data/authors_create_figures.ipynb`，仅作来源核对，未执行。
本项目没有重跑量子硬件、42 量子比特模拟或商业优化求解器，也没有下载全部 2.48 GB 的 Zenodo 归档。

## 研究问题与方法

1. 比较三目标情况下 MPS、ibm_fez 和经典方法的超体积（HV）收敛。
2. 分析电路深度 p、MPS 键维数 chi 对最终非支配解数量及收敛的影响。
3. 比较四目标情况下的模拟、经典方法及**模型预测**硬件曲线。
4. 补充最终 HV、相对差距、五次重复的标准差、首次达到参考 HV 的 99.99% 的采样时刻。

HV 越大越好；绘图指标 `HV_ref - HV(t) + 1` 越小越好，纵轴倒置。
HV_ref 是论文报告的最佳参考值，不将其称为连续权重问题的严格全局最优值。
MPS 的时间轴是按 10,000 shots/s 折算的采样时间，不是本机模拟耗时；
硬件时间也不代表包含编译、排队等开销的端到端耗时。
五次重复仅支持描述统计，阴影是最小值至最大值，不是置信区间。
时间点是同一运行的重复观测，不能当作独立样本进行显著性检验。

## 数据来源

- 论文与 Source Data：https://www.nature.com/articles/s43588-025-00873-y
- 完整归档 DOI：10.5281/zenodo.16878920
- 作者代码：https://github.com/stefan-woerner/qamoo
- 各下载文件的精确 URL 和 SHA-256 校验值见 `data/sources.json`。
""")

code("""
from pathlib import Path
import hashlib
import json
import re
import sys
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd()
assert (ROOT / "data/sources.json").exists(), "Run from the project root."
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 110,
    "savefig.dpi": 200,
})
RUNS = [f"hv_{i}" for i in range(5)]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6F6F6F", "#AA8B00"]
manifest = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
for item in manifest:
    assert hashlib.sha256((DATA / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    if item["file"].endswith(".zip"):
        archive_path = DATA / item["file"]
        with ZipFile(archive_path) as archive:
            assert archive.testzip() is None, archive_path.name
            for member in archive.infolist():
                if not member.is_dir() and "__MACOSX" not in member.filename:
                    extracted = (DATA / archive_path.stem / member.filename).resolve()
                    assert extracted.is_relative_to(DATA.resolve())
                    assert extracted.read_bytes() == archive.read(member), str(extracted)
tables = {}
for folder in sorted(DATA.glob("source_figure_*")):
    if folder.is_dir():
        for path in sorted(folder.glob("*.csv")):
            tables[path.stem] = pd.read_csv(path)
print(f"Verified {len(manifest)} downloads; loaded {len(tables)} source CSVs.")
print(f"Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, matplotlib {matplotlib.__version__}")
""")

md("""
## 数据质量检查

保留原始文件不变，校验缺失值、时间排序、重复记录、HV 单调性及重复均值。
图 3 中 p=2 的硬件结果未提供，保留缺失，不填零。
图 2、图 5 部分 `hv_lb/hv_ub` 是全曲线范围，不能作逐时刻范围；
本分析始终从 `hv_0` 至 `hv_4` 逐行重算。
图 5 的 `mps_num_gates_*` 列虽然以 `hv_` 命名，实际存储的是
`HV_ref - HV + 1`，用对应 MPS 均值和保真度时间缩放交叉核验后直接绘制。
这些预测曲线的前三点来自时间缩放；20,000 秒末点使用作者代码中的幂律
插值或外推，不是新的观测值，不用于实测性能结论。
""")

code("""
audit = []
for name, df in tables.items():
    assert not df.duplicated().any(), name
    if "t" in df:
        assert df.notna().all().all(), name
        assert np.isfinite(df.to_numpy()).all(), name
        assert (df.t > 0).all() and (np.diff(df.t) > 0).all(), name
        assert df.hv_max.nunique() == 1, name
        values = df[RUNS].to_numpy() if set(RUNS) <= set(df.columns) else df.iloc[:, 2:3].to_numpy()
        projection = "num_gates" in name
        direction = -1 if projection else 1
        assert (direction * np.diff(values, axis=0) >= -1e-5).all(), name
        assert (values >= 0).all(), name
        if not projection:
            assert (values <= df.hv_max.iloc[0] + 1e-5).all(), name
        bad_bounds = 0
        if set(RUNS) <= set(df.columns):
            assert np.allclose(values.mean(axis=1), df.hv_mean, atol=1e-7, rtol=0), name
            correct = np.isclose(values.min(axis=1), df.hv_lb, atol=1e-7, rtol=0)
            correct &= np.isclose(values.max(axis=1), df.hv_ub, atol=1e-7, rtol=0)
            bad_bounds = int((~correct).sum())
        audit.append(dict(file=name, rows=len(df), missing=0, range_mismatch_rows=bad_bounds))
    else:
        assert name == "figure_3"
        mask = df.isna()
        assert mask.to_numpy().sum() == 5
        assert df.loc[df.p == 2, [f"ibm_fez_{i}" for i in range(5)]].isna().all().all()
        audit.append(dict(file=name, rows=len(df), missing=5, range_mismatch_rows=0))

mps4 = tables["figure_5_data_mps"]
for gates in [5000, 7500, 10000, 15000, 30000]:
    df = tables[f"figure_5_data_mps_num_gates_{gates}"]
    fidelity = 0.0371 ** (5000 / gates)
    # The authors use three scaled observations plus one power-law model endpoint.
    scaled_t = mps4.t.to_numpy() / fidelity
    gap = mps4.hv_max.to_numpy() + 1 - mps4.hv_mean.to_numpy()
    assert len(df) == 4
    assert np.allclose(df.t.iloc[:3], scaled_t[:3], atol=0.01, rtol=0)
    assert np.allclose(df.iloc[:3, 2], gap[:3], atol=1e-6, rtol=0)
    exponent = np.log(gap[4] / gap[3]) / np.log(scaled_t[4] / scaled_t[3])
    endpoint = gap[3] * (20000 / scaled_t[3]) ** exponent
    assert df.t.iloc[-1] == 20000
    assert np.isclose(df.iloc[-1, 2], endpoint, atol=1e-6, rtol=0)

audit = pd.DataFrame(audit)
audit.to_csv(OUT / "data_quality.csv", index=False, encoding="utf-8-sig")
display(audit.loc[(audit.range_mismatch_rows > 0) | (audit.missing > 0)])
print("All validation assertions passed. Ranges are recomputed; missing hardware results stay missing.")
""")

code("""
def curve(name):
    df = tables[name]
    ref = float(df.hv_max.iloc[0])
    values = df[RUNS].to_numpy() if set(RUNS) <= set(df.columns) else df.iloc[:, 2:3].to_numpy()
    if "num_gates" in name:
        values = ref + 1 - values
    return df.t.to_numpy(), values, ref

def plot_curve(ax, name, label, color, linestyle="-", shade=True):
    t, values, ref = curve(name)
    ax.plot(t, ref + 1 - values.mean(axis=1), label=label, color=color, ls=linestyle, lw=1.8)
    if shade and values.shape[1] > 1:
        ax.fill_between(t, ref + 1 - values.max(axis=1), ref + 1 - values.min(axis=1),
                        color=color, alpha=0.15, linewidth=0)

def hv_axes(ax, title, top):
    ax.set(xscale="log", yscale="log", title=title,
           xlabel="Reported / assumed sampling time (s)",
           ylabel="HV reference - HV(t) + 1", ylim=(top, 0.75))
    ax.axhline(1, color="#555555", ls=":", lw=1)

def save_figure(fig, name):
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Runnable regression checks for parsing and repeated-run aggregation.
t, v, ref = curve("figure_2_data_ibm_fez_p_3")
assert v.shape == (100, 5) and t[-1] == 2500
assert np.isclose(v[0].mean(), 42136.66150929575)
_, predicted, ref4 = curve("figure_5_data_mps_num_gates_5000")
assert np.isclose(predicted[0, 0], mps4.hv_mean.iloc[0])
""")

md("""
## 原论文图 2：三目标 HV 收敛

各面板固定电路深度 p，比较 chi=50 的 MPS 与真实硬件。
经典方法为单次运行，不能估算它们的运行间方差。
曲线只覆盖源数据提供的时间范围，不外推。
""")

code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for ax, p in zip(axes.flat, range(3, 7)):
    plot_curve(ax, f"figure_2_data_mps_chi_50_p_{p}", "MPS, chi=50 (assumed time)", COLORS[0])
    plot_curve(ax, f"figure_2_data_ibm_fez_p_{p}", "ibm_fez (measured samples)", COLORS[1], "--")
    for key, label, color, style in [
        ("eps", "epsilon-CM", COLORS[2], "-."),
        ("dcm", "DCM", COLORS[3], ":"),
        ("dpa", "DPA-a", COLORS[4], (0, (5, 2))),
    ]:
        plot_curve(ax, "figure_2_data_" + key, label, color, style)
    hv_axes(ax, f"Three objectives | p = {p}", 25000)
axes[0, 0].legend(fontsize=8, loc="lower right")
fig.suptitle("HV convergence: mean and pointwise min-max across 5 runs", fontsize=14)
save_figure(fig, "01_hv_convergence_3obj")
""")

md("""
## 原论文图 3：非支配解数量

左图包含较低键维数 chi=1 的结果；右图放大高质量区间。
这里的非支配解数量不是 HV，也不单独证明算法更优。
经典 DPA-a/DCM 使用缩放并截断的权重，解数量差异不能直接解释为优势大小。
""")

code("""
points = tables["figure_3"]
point_rows = []
for method in ["ibm_fez", "chi_1", "chi_5", "chi_10", "chi_20", "chi_30", "chi_50"]:
    cols = [f"{method}_{i}" for i in range(5)]
    for _, row in points.iterrows():
        x = row[cols].dropna().to_numpy(dtype=float)
        if len(x):
            point_rows.append(dict(method=method, p=int(row.p), n=len(x),
                                   mean=x.mean(), sd=x.std(ddof=1), minimum=x.min(), maximum=x.max()))
point_stats = pd.DataFrame(point_rows)
point_stats.to_csv(OUT / "pareto_point_summary.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
palette = {method: color for method, color in zip(
    ["ibm_fez", "chi_1", "chi_5", "chi_10", "chi_20", "chi_30", "chi_50"],
    ["#D55E00", "#777777", "#AA8B00", "#0072B2", "#009E73", "#CC79A7", "#222222"])}
for ax in axes:
    for method, group in point_stats.groupby("method", sort=False):
        if ax is axes[1] and method == "chi_1":
            continue
        ax.plot(group.p, group["mean"], marker="o", ms=4, color=palette[method],
                ls="--" if method == "ibm_fez" else "-", label=method.replace("_", " "))
        ax.fill_between(group.p, group.minimum, group.maximum, color=palette[method], alpha=0.12)
    ax.axhline(2063, color="#8B1A1A", ls=":", label="DPA-a / DCM: 2063")
    ax.axhline(2054, color="#7251A2", ls="-.", label="epsilon-CM: 2054")
    ax.set(xlabel="QAOA depth p", ylabel="Number of non-dominated points", xticks=range(2, 7))
axes[0].set(title="Full range (including chi=1)", ylim=(0, 2920))
axes[1].set(title="Detail: higher-quality settings", ylim=(1700, 2820))
axes[0].legend(fontsize=8, ncol=2, loc="center right")
fig.suptitle("Pareto-set size: five-run mean and min-max", fontsize=14)
save_figure(fig, "02_pareto_point_counts")
display(point_stats[point_stats.method == "ibm_fez"])
""")

md("""
## 原论文图 4：MPS 参数敏感性

在相同 2,500 秒折算采样预算下，比较六种键维数与六种电路深度。
不增加模拟规模，而直接复现作者已经公开的重复实验数据。
""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
for ax, chi in zip(axes.flat, [1, 5, 10, 20, 30, 50]):
    for p in range(1, 7):
        plot_curve(ax, f"figure_4_data_chi_{chi}_p_{p}", f"p={p}", COLORS[p-1])
    hv_axes(ax, f"MPS bond dimension chi = {chi}", 25000)
axes[0, 0].legend(fontsize=8, ncol=2, loc="upper right")
fig.suptitle("Sensitivity to MPS bond dimension and QAOA depth", fontsize=14)
save_figure(fig, "03_mps_parameter_sensitivity")
""")

md("""
## 原论文图 5：四目标问题与硬件性能预测

实线 MPS 是模拟数据；其他硬件曲线是通过保真度假设缩放时间的预测，
**不是四目标问题在这些设备上的实测结果**。
保真度关系沿用论文 `0.0371 ** (5000 / gate_capacity)`，
此处不将论文发布时的硬件路线图当作现状。
""")

code("""
fig, ax = plt.subplots(figsize=(12, 6.8), layout="constrained")
plot_curve(ax, "figure_5_data_mps", "MPS chi=20, p=6 (assumed time)", "#0072B2")
for gates, color in zip([5000, 7500, 10000, 15000, 30000],
                       ["#B8A600", "#CC79A7", "#D55E00", "#009E73", "#555555"]):
    fidelity = 0.0371 ** (5000 / gates)
    plot_curve(ax, f"figure_5_data_mps_num_gates_{gates}",
               f"Modeled HW: {100*fidelity:.2f}% fidelity", color, "--", shade=False)
for key, label, color in [("eps", "epsilon-CM", "#7251A2"),
                          ("dcm", "DCM", "#333333"), ("dpa", "DPA-a", "#A32020")]:
    plot_curve(ax, "figure_5_data_" + key, label, color, "-.")
hv_axes(ax, "Four objectives: observed classical / simulated / predicted results", 1500000)
ax.legend(fontsize=8, loc="lower right", ncol=2)
save_figure(fig, "04_hv_comparison_4obj")
""")

md("""
## 补充定量分析

对每次重复先找首次达到 `HV >= 0.9999 * HV_ref` 的已记录时刻，再汇总。
若第一条记录已达标，只能说“截至首次观测已达标”，不是精确达标时间；
未达到阈值记为缺失而非零。有限采样网格上的首次达标时刻是上界，
不采用线性插值声称更精确的时间。

末次 HV 使用各方法实际提供的最后一个时间点，时间预算不完全相同，
因此另给三目标 2,500 秒预算下的最后已观测 HV。
对已结束经典算法沿用其最终结果，不对中间未记录区间插值。
标准差只在存在多次重复时计算。
硬件预测单列为 `model_prediction`，其阈值时刻只是预测网格的指标，
包含幂律末点，不是实测达标时间。
""")

code("""
summary_rows, crossing_rows = [], []
selected = [name for name in tables if name.startswith(("figure_2_", "figure_4_", "figure_5_"))]
for name in selected:
    t, values, ref = curve(name)
    final = values[-1]
    kind = "model_prediction" if "num_gates" in name else "published_experiment_or_simulation"
    row = dict(source=name, data_kind=kind, n_runs=values.shape[1], last_time_s=t[-1],
               hv_reference=ref, final_hv_mean=final.mean(),
               final_hv_sd=final.std(ddof=1) if len(final) > 1 else np.nan,
               final_hv_min=final.min(), final_hv_max=final.max(),
               relative_gap_pct=100*(ref-final.mean())/ref)
    if name.startswith("figure_2_"):
        eligible = np.flatnonzero(t <= 2500)
        row["hv_mean_last_observed_at_2500s"] = values[eligible[-1]].mean() if len(eligible) else np.nan
        row["observation_time_at_2500s"] = t[eligible[-1]] if len(eligible) else np.nan
    for i in range(values.shape[1]):
        hits = np.flatnonzero(values[:, i] >= ref * 0.9999)
        crossing_rows.append(dict(source=name, data_kind=kind, run=i, target_ratio=0.9999,
                                  first_recorded_time_s=t[hits[0]] if len(hits) else np.nan,
                                  reached=bool(len(hits)),
                                  reached_by_first_observation=bool(len(hits) and hits[0] == 0),
                                  last_observation_s=t[-1]))
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
crossings = pd.DataFrame(crossing_rows)
summary.to_csv(OUT / "hv_summary.csv", index=False, encoding="utf-8-sig")
crossings.to_csv(OUT / "time_to_9999pct.csv", index=False, encoding="utf-8-sig")
display(summary[summary.source.str.startswith("figure_2_")].round(6))
display(crossings[crossings.source.str.startswith("figure_2_")].groupby("source").agg(
    reached_runs=("reached", "sum"), total_runs=("run", "count"),
    mean_first_recorded_time_s=("first_recorded_time_s", "mean"),
    min_first_recorded_time_s=("first_recorded_time_s", "min"),
    max_first_recorded_time_s=("first_recorded_time_s", "max"),
    reached_by_first_observation=("reached_by_first_observation", "sum")))
""")

code("""
hardware = summary[summary.source.str.contains("figure_2_data_ibm_fez")]
first = point_stats[(point_stats.method == "ibm_fez") & (point_stats.p == 3)].iloc[0]
last = point_stats[(point_stats.method == "ibm_fez") & (point_stats.p == 6)].iloc[0]
repaired = audit[audit.range_mismatch_rows > 0]
conclusions = [
    f"共校验 {len(manifest)} 个下载文件、读取 {len(tables)} 个源 CSV；其中 {len(repaired)} 个 CSV 的区间列与逐时刻重复范围不一致，绘图已从重复数据重算。",
    f"真实硬件的平均非支配解数量由 p=3 的 {first['mean']:.1f} 增至 p=6 的 {last['mean']:.1f}，增幅 {(last['mean']/first['mean']-1)*100:.2f}%。",
]
for _, row in hardware.iterrows():
    p = row.source.rsplit("_", 1)[1]
    conclusions.append(f"ibm_fez p={p}：2,500 秒末次平均 HV={row.final_hv_mean:.6f}，相对参考 HV 差距 {row.relative_gap_pct:.6f}%。")
conclusions += [
    "较大的非支配解集合不一定带来同比例的 HV 增益；权重截断也影响经典方法的解数量。",
    "MPS 与硬件曲线采用论文的采样计时口径，不能据此声称已经验证端到端量子加速。",
    "四目标硬件曲线属于模型预测；仅有少量固定问题实例，不能外推到所有多目标优化问题。",
    "本次完成公开源数据层面的结果复现，未重新运行 42 量子比特实验、MPS 模拟、27 节点参数训练或商业求解器。",
]
text = "# 分析结论与提交状态\\n\\n" + "\\n\\n".join("- " + line for line in conclusions)
(OUT / "analysis_summary.md").write_text(text, encoding="utf-8")
display(Markdown(text))
""")

md("""
## 复现与交付

- 按顺序运行所有单元，无需量子硬件账号或商业求解器。
- `output/01` 至 `04` 开头的 PNG 为可上传图表。
- `data_quality.csv` 为质量检查记录；`hv_summary.csv` 为 HV 汇总；
  `pareto_point_summary.csv` 为解数量汇总；`time_to_9999pct.csv` 为逐运行阈值记录。
- 项目附带已执行 notebook、HTML 阅读版和原始源数据。
- 此交付为公开源数据复现，不代表独立重跑原论文全部实验。
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
path = ROOT / "analysis.ipynb"
nbf.write(notebook, path)
# Install a project-local kernelspec so execution never picks another Python.
import sys
kernel_dir = ROOT / ".jupyter" / "kernels" / "qamoo-local"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "QAMOO local", "language": "python",
}), encoding="utf-8")
manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
client = NotebookClient(
    notebook, timeout=180, kernel_name="qamoo-local",
    resources={"metadata": {"path": str(ROOT)}},
)
client.create_kernel_manager()
client.km.kernel_spec_manager = manager
try:
    client.execute()
finally:
    nbf.write(notebook, path)
assert all(c.get("execution_count") is not None for c in notebook.cells if c.cell_type == "code")
assert not any(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
html, _ = HTMLExporter().from_notebook_node(notebook)
(OUT / "analysis.html").write_text(html, encoding="utf-8")
with ZipFile(OUT / "analysis_draft_bundle.zip", "w", ZIP_DEFLATED) as bundle:
    for file in [path, ROOT / "download_data.py", ROOT / "build_notebook.py", ROOT / "README.md",
                 ROOT / "requirements.txt", ROOT / "requirements-lock.txt", *DATA.rglob("*"), *OUT.glob("*")]:
        if file.is_file() and "__MACOSX" not in file.parts and file.suffix != ".zip":
            bundle.write(file, file.relative_to(ROOT))
    # Include the original verified archives for checksum verification.
    for file in DATA.glob("*.zip"):
        bundle.write(file, file.relative_to(ROOT))
print(f"Executed {sum(c.cell_type == 'code' for c in cells)} code cells successfully.")
print(f"Notebook: {path}")
print(f"Figures: {len(list(OUT.glob('*.png')))}")
