"""Build and execute the portable, data-embedded assignment notebook."""
from __future__ import annotations

import ast
import base64
import json
import os
from pathlib import Path
import sys
import textwrap
import time

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/supervised_learning"
SUBMIT = ROOT / "output/assignment_submission"
SUBMIT.mkdir(parents=True, exist_ok=True)
source = (ROOT / "supervised_learning.py").read_text(encoding="utf-8")
syntax = ast.parse(source)
functions = {node.name: ast.get_source_segment(source, node) for node in syntax.body
             if isinstance(node, ast.FunctionDef)}
verification_source = (ROOT / "verify_supervised_results.py").read_text(encoding="utf-8")
verification_node = next(node for node in ast.parse(verification_source).body
                         if isinstance(node, ast.FunctionDef) and node.name == "verify")
verify_code = ast.get_source_segment(verification_source, verification_node)
verify_code = verify_code.replace(
    'assert (OUT / "report.html").read_text(encoding="utf-8").count("data:image/png;base64,") == 4',
    'assert all((OUT / (name + ".png")).is_file() for name in figure_names)')
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    if hidden:
        cell.metadata["jupyter"] = {"source_hidden": True}
    cells.append(cell)


md("""
# 基于实验数据的优化质量预测
## 监督学习作业 notebook

本项目完成三个作业要求：从已收集数据中定义监督学习任务；实际训练至少三种机器学习模型；综合运用评价指标解释模型性能。

**任务**：给定电路深度 p、模拟精度参数 chi 和采样时间 t，预测固定三目标优化问题的结果质量。
本 notebook 内嵌经过校验的原始数据压缩包，可作为单个 `.ipynb` 导入 AI Studio；从上到下运行全部单元即可得到数据、训练输出、测试指标和四张评价图。

**运行说明**：使用 Python 3.10 或以上版本的 CPU 环境。所需依赖为 numpy、pandas、scipy、matplotlib、scikit-learn、joblib 和 IPython；缺失时第一个代码单元会安装。训练通常需要几分钟，以实际输出时间为准。所有数据文件只在 notebook 当前工作目录的 `data` 与 `output/supervised_learning` 下生成。已有同名原始数据会先验证一致性。
""")

code("""
from __future__ import annotations
import base64
import hashlib
import html
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

required = {'numpy': 'numpy>=1.24,<3', 'pandas': 'pandas>=2.0,<3',
            'scipy': 'scipy>=1.10,<2', 'matplotlib': 'matplotlib>=3.9,<4',
            'sklearn': 'scikit-learn>=1.4,<1.7', 'joblib': 'joblib>=1.2',
            'IPython': 'ipython>=8'}
missing = [package for module, package in required.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

ROOT = Path.cwd().resolve()
OUT = ROOT / 'output' / 'supervised_learning'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display, Markdown, Image
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

FEATURES = ['depth_p', 'log10_bond_dimension', 'log10_time_s']
SEED = 42
NAMES = {'ridge': '多项式岭回归', 'svr': '支持向量回归（RBF）',
         'random_forest': '随机森林回归', 'baseline': '常数基线'}
DISPLAY = {'ridge': 'Polynomial Ridge', 'svr': 'RBF SVR',
           'random_forest': 'Random Forest', 'baseline': 'Mean baseline'}
COLORS = {'ridge': '#6C83A5', 'svr': '#B36B43',
          'random_forest': '#458C83', 'baseline': '#8A8A8A'}
print('Python:', platform.python_version(), '| scikit-learn:', sklearn.__version__)
print('随机种子:', SEED)
""")

md("""
## 1 建模背景与监督标签

多目标优化同时考虑多个目标，超体积 HV（hypervolume）用来概括所得到解集的质量。对本固定问题而言，HV 越接近论文参考值，解集质量越好。
不同电路深度、模拟精度和采样时间会影响实验结果。若能学到这些输入与结果质量之间的关系，就能在同一问题上估计未参与训练的参数设置的表现。

使用本项目已收集论文 *Quantum approximate multi-objective optimization* 的图4配套源数据：36组(p,chi)，每组5次重复，每次100个时间点，共 **18,000条实际观测**。数据来源 DOI：`10.1038/s43588-025-00873-y`。

- 输入：p、log10(chi)、log10(t)，这些值在运行实验前可确定。
- 原始标签：每次重复的 HV 观测，即 hv_0 至 hv_4。
- 训练标签：`y = log10(max(HV_reference - HV, 0) + 1)`。
- 预测值还原：`HV_pred = HV_reference + 1 - 10**y_pred`。

差距跨多个数量级，因此使用对数标签；同时保留原始HV尺度评价，检查变换是否掩盖绝对误差。hv_mean、hv_lb、hv_ub 和任何已观测HV均不进入输入。相同特征有5次重复观测，其差异是实验随机性。
""")

md("""
### 原始数据随 notebook 提供

下方单元内嵌原始发布的 ZIP 字节，仅用于让单文件 notebook 可以离线还原数据。它不会生成新数据或标签。解码后核验 SHA-256，再将36个源CSV写入数据目录；若目录已有文件，则逐字节验证。
""")
archive_bytes = (ROOT / "data/source_figure_4.zip").read_bytes()
manifest = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
archive_info = next(item for item in manifest if item["file"] == "source_figure_4.zip")
payload = base64.b64encode(archive_bytes).decode("ascii")
code(f"""
SOURCE_ARCHIVE_INFO = {archive_info!r}
SOURCE_ARCHIVE_B64 = {payload!r}
archive_bytes = base64.b64decode(SOURCE_ARCHIVE_B64)
assert hashlib.sha256(archive_bytes).hexdigest() == SOURCE_ARCHIVE_INFO['sha256']
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
archive_path = DATA / 'source_figure_4.zip'
if archive_path.exists():
    assert archive_path.read_bytes() == archive_bytes
else:
    archive_path.write_bytes(archive_bytes)
folder = DATA / 'source_figure_4'
folder.mkdir(exist_ok=True)
with ZipFile(io.BytesIO(archive_bytes)) as archive:
    members = [name for name in archive.namelist()
               if name.endswith('.csv') and '__MACOSX' not in name]
    assert len(members) == 36
    for member in members:
        destination = folder / Path(member).name
        raw = archive.read(member)
        if destination.exists():
            assert destination.read_bytes() == raw
        else:
            destination.write_bytes(raw)
manifest_path = DATA / 'sources.json'
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert any(item['file'] == SOURCE_ARCHIVE_INFO['file'] and
               item['sha256'] == SOURCE_ARCHIVE_INFO['sha256'] for item in existing)
else:
    manifest_path.write_text(json.dumps([SOURCE_ARCHIVE_INFO], ensure_ascii=False,
                                        indent=2), encoding='utf-8')
print('校验成功：', len(archive_bytes), '字节，36个原始CSV。')
print('SHA-256:', SOURCE_ARCHIVE_INFO['sha256'])
""", hidden=True)

md("""
## 2 数据提取 清洗与防止泄漏

将每个CSV的5列重复观测展开为逐样本记录，保留文件名、原始行号、重复编号，检查缺失值、有限数值、时间顺序、HV随时间非递减及参考值范围。保留图4全部观测。max仅消除HV与参考值相减时约1e-11的舍入误差。

以完整(p,chi)参数组合为分组单位，固定种子42划分 **27组训练（13,500条）和9组测试（4,500条）**。同一条实验曲线及其5次重复整体留在同一集合；训练集内部采用5折GroupKFold。这样避免把高度相似的相邻时间点随机分散到训练集和测试集。
""")
code("\n\n".join(functions[name] for name in ["write_json", "save_csv", "extract_dataset"]))
code("""
data, train_idx, test_idx, cv = extract_dataset()
display(data[['depth_p', 'bond_dimension_chi', 'time_s', 'replicate',
              'target_hv', 'target_log10_gap', 'configuration', 'split']].head(10))
display(data.groupby('split').agg(样本数=('sample_id', 'count'),
                                  参数组合数=('configuration', 'nunique')))
assert set(data.iloc[train_idx].configuration).isdisjoint(data.iloc[test_idx].configuration)
print('训练与测试参数组合交集为0；原始样本没有删减。')
""")

md("""
## 3 三类模型与超参数搜索

1. **多项式岭回归**：提供带正则化的平滑非线性模型，搜索多项式次数2/3和alpha=0.1/10。多项式展开与标准化放在Pipeline内，每折仅使用该折训练数据拟合。
2. **RBF支持向量回归**：通过核函数学习非线性关系，搜索C=10/100及gamma=scale/0.1，epsilon固定0.05；标准化也在Pipeline内。
3. **随机森林回归**：通过120棵树拟合参数和时间的非线性作用，搜索最大深度12/不限和叶节点最少5/20条观测。

另训练**常数基线**：只预测训练标签均值，用于判断复杂模型是否优于最基本的预测方法。
每个模型搜索4组参数，共20次分组CV拟合；三种算法共60次CV拟合，之后分别用完整训练集拟合最终模型。算法和参数选择只看训练CV的平均RMSE(log)，不查看测试误差。
""")
code("\n\n".join(functions[name] for name in ["candidates", "train_models"]))
code("""
training_started = time.perf_counter()
models, cv_table, selected = train_models(data, train_idx, test_idx, cv)
training_seconds = time.perf_counter() - training_started
display(cv_table[['model', 'cv_rmse_log', 'cv_rmse_fold_sd', 'search_and_refit_seconds']])
print('仅按训练集CV选出的模型：', NAMES[selected])
print('训练与搜索总耗时：', round(training_seconds, 2), '秒')
display(json.loads((OUT / 'best_parameters.json').read_text(encoding='utf-8')))
""")

md("""
## 4 综合评价方案

在独立测试集上同时报告下列指标，MAE和RMSE越小越好，R²越接近1越好。R²为负说明比该评价集的真实标签均值预测还差；基线仅使用训练均值，因此其测试R²不必为0。

- **MAE**：平均绝对误差，`mean(abs(y - y_pred))`。
- **RMSE**：均方根误差，`sqrt(mean((y - y_pred)**2))`，对较大偏差更敏感。
- **R²**：决定系数，`1 - sum((y-y_pred)**2) / sum((y-mean(y))**2)`。
- **分组交叉验证均值与折间标准差**：评价训练阶段对未见参数组合的适应能力。
- **分参数组误差**：检查整体指标是否掩盖局部失败。
- **95% bootstrap区间**：按9个完整测试参数组重采样2000次估计RMSE(log)区间；不把相关时间点当成独立实验。

对数目标和还原后的原始HV分别报告MAE、RMSE与R²；还原预测值不依据测试结果作截断。
""")
code(functions["evaluate_models"])
code("""
metrics, predictions, group_metrics = evaluate_models(data, train_idx, test_idx, models, cv_table)
test_metrics = metrics[metrics.split == 'test'].set_index('model')
display(Markdown('### 对数目标上的独立测试结果'))
display(test_metrics[['mae_log', 'rmse_log', 'r2_log',
                      'test_rmse_log_ci_low', 'test_rmse_log_ci_high']].round(6))
display(Markdown('### 原始HV尺度上的独立测试结果'))
display(test_metrics[['mae_hv', 'rmse_hv', 'r2_hv']].round(6))
display(Markdown('### 训练误差与测试误差对比'))
display(metrics.pivot(index='model', columns='split', values='rmse_log').round(6))
""")

md("""
## 5 图表证据与误差分析

图1比较全部模型的测试RMSE及按参数组合bootstrap得到的95%区间；图2检查由训练CV选出的模型的真实值与预测值；图3比较四种方法的残差偏差与长尾；图4显示全部9个测试参数组合在固定末次时刻2500秒的表现。
图4的观测误差线是5次重复的最小值至最大值，**不是置信区间**。所有测试样本均保留。图中采用英文标签便于不同运行环境显示，中文含义由下方图注说明。
""")
code(functions["make_figures"])
code("""
qa_directory = os.environ.get('FIGURE_QA_SCRIPTS')
figure_names = make_figures(data, metrics, predictions, selected,
                            Path(qa_directory) if qa_directory else None)
captions = ['图1 独立测试误差与95%参数组bootstrap区间',
            '图2 训练CV所选模型的全部4500条测试预测与观测',
            '图3 各模型测试残差的中位数 四分位距和长尾',
            '图4 末次时刻的全部9个测试参数组合 观测范围为5次重复的最小至最大值']
for name, caption in zip(figure_names, captions):
    display(Markdown('**' + caption + '**'))
    display(Image(filename=str(OUT / (name + '.png')), width=900))
""")
code("""
best_test_name = test_metrics.rmse_log.idxmin()
selected_row = test_metrics.loc[selected]
forest = test_metrics.loc['random_forest']
worst = group_metrics[group_metrics.model == selected].sort_values('rmse_hv', ascending=False).iloc[0]
display(Markdown(f'''
## 6 结果解释与结论

训练集分组交叉验证选出了 **{NAMES[selected]}**。其测试RMSE(log)为 **{selected_row.rmse_log:.4f}**，
相对常数基线降低 **{100*(1-selected_row.rmse_log/test_metrics.loc['baseline'].rmse_log):.2f}%**。

在本次固定测试集上，**{NAMES[best_test_name]}的数值表现最好**。
随机森林的测试MAE(log)为{forest.mae_log:.4f}、RMSE(log)为{forest.rmse_log:.4f}、R²(log)为{forest.r2_log:.4f}；
原始HV的MAE为{forest.mae_hv:.4f}、RMSE为{forest.rmse_hv:.4f}、R²为{forest.r2_hv:.4f}。
这与训练CV的选择存在差异，说明单次划分下的模型排名会波动；本项目保留原CV选择，不以测试结果重新调参。

**高的对数目标R²不保证原始HV尺度上同样准确**。{NAMES[selected]}在{worst.configuration}上的原始HV RMSE为
{worst.rmse_hv:.4f}，指数还原放大了大差距区域的偏差；其整体原始HV R²为{selected_row.r2_hv:.4f}。
因此，应同时阅读两种尺度、分组误差和残差图，不能只用单个较高分数判断模型。

本任务数据来自同一个固定的42节点三目标问题，能评价未见参数组合的预测，不能据此外推到新图实例或真实量子硬件。
MPS采样时间是论文的折算口径，实际机器学习训练耗时另外记录。18000个时间与重复观测并非18000个独立实验。
'''))
display(group_metrics.sort_values(['model', 'rmse_hv'], ascending=[True, False]).round(6))
""")

md("""
## 7 独立核验与保存结果

最后重新读取导出的样本和模型，逐条核对原始标签，检查分组隔离，重新预测测试集并复算全部评价指标。
验证成功才完成此次运行。可提交本已执行notebook及配套PDF说明文档；本地文件准备完成不等于教学系统已经提交。
""")
code(verify_code)
code("""
verify()
run_manifest = {'status': 'completed', 'seed': SEED, 'samples': len(data),
                'train_samples': len(train_idx), 'test_samples': len(test_idx),
                'trained_algorithms': ['ridge', 'svr', 'random_forest'],
                'selected_by_training_cv': selected, 'training_seconds': training_seconds,
                'python': platform.python_version(), 'sklearn': sklearn.__version__}
write_json(OUT / 'notebook_run_manifest.json', run_manifest)
write_json(OUT / 'environment.json', {'python': platform.python_version(),
           'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
           'matplotlib': matplotlib.__version__, 'seed': SEED})
(OUT / 'requirements-lock.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
print('全部单元完成：数据提取、三类模型训练、独立测试和指标复算均已执行。')
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata["language_info"] = {"name": "python", "version": "3.10.11"}
notebook.metadata["assignment"] = {"source_archive_sha256": archive_info["sha256"],
                                  "includes_original_source_data": True}
path = ROOT / "supervised_assignment.ipynb"
nbf.write(notebook, path)
kernel_dir = ROOT / ".jupyter/kernels/supervised-python310"
kernel_dir.mkdir(parents=True, exist_ok=True)
python310 = ROOT / ".runtime/python310/python.exe"
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [str(python310), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Supervised learning Python 3.10", "language": "python",
}), encoding="utf-8")
manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
os.environ["FIGURE_QA_SCRIPTS"] = str(Path.home() / ".agents/skills/nature-figure/scripts")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter/runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".jupyter/ipython-supervised")
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(exist_ok=True)
client = NotebookClient(notebook, timeout=1200, kernel_name="supervised-python310",
                        resources={"metadata": {"path": str(ROOT)}})
client.create_kernel_manager()
client.km.kernel_spec_manager = manager
started = time.perf_counter()
try:
    client.execute()
finally:
    # Ship a standard kernel name, independent of this computer's local kernelspec.
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(notebook, path)
assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code"
               for output in cell.outputs)
execution = {"status": "PASS", "executed_code_cells": sum(c.cell_type == "code" for c in cells),
             "elapsed_seconds": time.perf_counter()-started,
             "errors": 0, "embedded_source_sha256": archive_info["sha256"]}
(SUBMIT / "notebook_execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
(SUBMIT / path.name).write_bytes(path.read_bytes())
exporter = HTMLExporter()
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
reading_copy, _ = exporter.from_notebook_node(notebook)
(SUBMIT / "supervised_assignment.html").write_text(reading_copy, encoding="utf-8")
print(json.dumps(execution, indent=2))
print("Executed notebook saved: supervised_assignment.ipynb", flush=True)
