"""Export the supplied paper and published tables as strict UTF-8 JSON/JSONL."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'machine_readable'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def number(value):
    if value.strip() == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite source value: {value}')
    return int(result) if result.is_integer() else result


def main():
    import pymupdf
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = list(ROOT.glob('*.pdf'))
    if len(pdfs) != 1:
        raise ValueError('Expected exactly one input paper PDF.')
    pdf = pdfs[0]
    with pymupdf.open(pdf) as doc:
        pages = [{'page': i + 1, 'text': page.get_text(sort=True)} for i, page in enumerate(doc)]
        paper = {
            'title': 'Quantum approximate multi-objective optimization',
            'doi': '10.1038/s43588-025-00873-y',
            'source_file': pdf.name,
            'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'page_count': len(pages), 'pages': pages,
            'extraction_note': '自动提取的阅读顺序文本；公式、分栏和图内标签可能失真。图像未转换为数值数据。',
        }
    write_json(OUT / 'paper.json', paper)
    (OUT / 'paper.txt').write_text('\n\n'.join(f"=== PAGE {p['page']} ===\n{p['text']}" for p in pages), encoding='utf-8')
    catalog, total, missing = [], 0, 0
    with (OUT / 'experiments.jsonl').open('w', encoding='utf-8') as stream:
        for path in sorted((ROOT / 'data').glob('source_figure_*/*.csv')):
            source = path.relative_to(ROOT).as_posix()
            name = path.stem
            projection = 'num_gates' in name
            figure = int(re.search(r'figure_(\d+)', name).group(1))
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                rows = list(reader)
            for index, row in enumerate(rows, 1):
                values = {key: number(value) for key, value in row.items()}
                missing += sum(v is None for v in values.values())
                record = {'source_file': source, 'source_row': index,
                          'figure': figure, 'dataset': name,
                          'data_kind': 'hardware_prediction' if projection else
                          ('hardware_measurement' if 'ibm_fez' in name else
                           'simulation' if ('mps' in name or 'chi_' in name) else
                           'mixed_methods' if figure == 3 else 'classical_result'),
                          'values': values}
                if projection:
                    # These source columns store transformed gaps, not HV values.
                    record['derived_hv'] = {k: values['hv_max'] + 1 - v
                                            for k, v in values.items()
                                            if k.startswith('hv_') and k != 'hv_max' and v is not None}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                total += 1
            catalog.append({'source_file': source, 'dataset': name, 'figure': figure,
                            'rows': len(rows), 'columns': columns,
                            'hv_columns_store': 'hv_reference_minus_hv_plus_1' if projection else
                            'hypervolume' if 't' in columns else 'not_applicable',
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(OUT / 'catalog.json', {
        'format_version': '1.0', 'encoding': 'UTF-8', 'record_count': total,
        'missing_value_count': missing, 'table_count': len(catalog), 'tables': catalog,
        'fields': {
            'source_row': '从 1 开始的数据行号，不含 CSV 表头。',
            'figure': '源文件编号；6 对应原论文 Extended Data Fig. 1。',
            'values': '保留原始列名的数值对象；缺失值为 JSON null。',
            't': '秒；沿用源数据口径。MPS 为假定采样时间，非本机耗时。',
            'hv_max': '论文参考超体积，不等同于已证明的连续权重全局最优值。',
            'hv_0..hv_4': '五次重复；预测文件中这些列实际存储变换后的差距。',
            'hv_mean/hv_lb/hv_ub': '源文件提供的均值/下界/上界；已知部分区间不一致，未改写。',
            'derived_hv': '仅预测文件提供：hv_max + 1 - 源差距值。',
            'p': 'QAOA 电路深度。',
            'figure_3 columns': '各方法的非支配解数量；方法后缀 0..4 表示重复编号。',
        },
        'limitations': ['非训练用标签数据集；这些记录是论文公开的汇总实验结果。',
                        '原始区间质量问题见 ../data_quality.csv；本导出保留源数值。',
                        '图 5 硬件预测包括外推末点，不可视为硬件实测。']})
    (OUT / 'README.md').write_text('''# 机器可读数据

- paper.txt：论文逐页纯文本，适合搜索和大模型阅读。
- paper.json：论文信息、PDF 校验值及每页文本。
- experiments.jsonl：每行一个 JSON 对象，包含来源、数据行号、实验类别及原始数值。
- catalog.json：文件目录、字段说明、单位和限制。

原始 CSV 已是机器可读数据；这里统一类型、来源与元数据。数值缺失转为 null，未插值或填零。
PDF 公式及图像的自动提取有局限，精确复现以原 PDF 和源 CSV 为准。

Python 读取示例（标准库，无需 pandas）：

```python
import json
from pathlib import Path
base = Path(__file__).resolve().parent  # 示例脚本保存于本目录时
paper = json.loads((base / 'paper.json').read_text(encoding='utf-8'))
with (base / 'experiments.jsonl').open(encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        # record['values'] 为数值字典
```

重新生成：在项目根目录运行 `python convert_machine_data.py`，需安装 PyMuPDF。
''', encoding='utf-8')
    # Reopen all exports to check strict JSON and exact record counts.
    for path in OUT.glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    with (OUT / 'experiments.jsonl').open(encoding='utf-8') as stream:
        assert sum(bool(json.loads(line)) for line in stream) == total
    print(f'Export verified: {len(pages)} pages, {len(catalog)} tables, {total} records, {missing} null values.')


if __name__ == '__main__':
    main()
