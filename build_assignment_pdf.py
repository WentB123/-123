"""Create the assignment's uploadable Chinese PDF from executed results."""
from pathlib import Path
import json
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "output/supervised_learning"
OUT = ROOT / "output/assignment_submission"
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")
pdfmetrics.registerFont(TTFont("AssignmentCN", str(FONT_PATH)))
F = "AssignmentCN"
styles = {
    "title": ParagraphStyle("Title", fontName=F, fontSize=21, leading=29, textColor=colors.black,
                            spaceAfter=10),
    "subtitle": ParagraphStyle("Subtitle", fontName=F, fontSize=12, leading=18, spaceAfter=16),
    "heading": ParagraphStyle("Heading", fontName=F, fontSize=13.5, leading=20,
                              spaceBefore=10, spaceAfter=8, keepWithNext=True),
    "body": ParagraphStyle("Body", fontName=F, fontSize=10, leading=16.3, spaceAfter=7,
                           wordWrap="CJK"),
    "small": ParagraphStyle("Small", fontName=F, fontSize=8.6, leading=13, spaceAfter=6,
                            wordWrap="CJK"),
    "table": ParagraphStyle("Table", fontName=F, fontSize=8.6, leading=12.8, wordWrap="CJK"),
    "caption": ParagraphStyle("Caption", fontName=F, fontSize=8.7, leading=13.2,
                              spaceBefore=5, spaceAfter=9, wordWrap="CJK"),
}


def p(text, kind="body"):
    markup = escape(str(text)).replace("R²", "R<super>2</super>")
    return Paragraph(markup.replace("\n", "<br/>"), styles[kind])


def table(rows, widths):
    values = [[p(cell, "table") for cell in row] for row in rows]
    result = Table(values, colWidths=[value * mm for value in widths], repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF3F6")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#708695")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.35, colors.HexColor("#D9E1E6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return result


def figure(name, width_mm=158):
    path = RESULTS / (name + ".png")
    with PILImage.open(path) as im:
        aspect = im.height / im.width
    return Image(str(path), width=width_mm*mm, height=width_mm*mm*aspect)


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont(F, 8)
    canvas.setFillColor(colors.HexColor("#65737F"))
    canvas.drawString(18*mm, 11*mm, "监督学习建模作业")
    canvas.drawRightString(A4[0]-18*mm, 11*mm, f"第 {document.page} 页")
    canvas.restoreState()


def main():
    run = json.loads((OUT / "notebook_execution.json").read_text(encoding="utf-8"))
    assert run["status"] == "PASS" and run["errors"] == 0
    verified = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "PASS"
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    test = metrics[metrics.split == "test"].set_index("model")
    cv = pd.read_csv(RESULTS / "cross_validation.csv").set_index("model")
    chosen = json.loads((RESULTS / "best_parameters.json").read_text(encoding="utf-8"))
    selected = chosen["selected_by_training_cv"]
    names = {"ridge": "多项式岭回归", "svr": "RBF支持向量回归", "random_forest": "随机森林回归", "baseline": "常数基线"}
    forest, svr = test.loc["random_forest"], test.loc["svr"]
    story = []
    add = story.append

    # Page 1: background and supervised task.
    add(p("基于实验数据的优化质量预测", "title"))
    add(p("监督学习建模作业说明", "subtitle"))
    add(p(f"本作业从已收集的公开实验数据中提取18,000条有监督样本，实际训练多项式岭回归、支持向量回归和随机森林，并使用独立测试、分组交叉验证、MAE、RMSE、R²及误差图评价性能。随机森林在本次固定测试集上的对数目标R²为{forest.r2_log:.4f}，原始HV的R²为{forest.r2_hv:.4f}。配套notebook已从首个代码单元运行至末个单元，无执行错误。"))
    add(p("一 建模背景与任务定义", "heading"))
    add(p("多目标优化需要同时改善多个目标。超体积HV可以概括所得解集的质量；在本固定问题中，HV越接近论文参考值，结果质量越好。电路深度、模拟精度及采样时间都会影响结果。由这些事先已知的输入预测结果质量，可用于比较同一问题上不同参数配置的预期表现。"))
    add(p("将该需求提炼为监督回归任务：输入电路深度p、模拟精度参数chi和采样时间t，预测HV相对参考值的对数差距。该任务从原实验记录重新定义，不是原论文提供的现成分类任务。"))
    add(table([
        ["要素", "本作业定义"],
        ["输入特征", "p、log10(chi)、log10(t)，共3个特征"],
        ["原始标签", "每次重复的观测HV，取源CSV的hv_0至hv_4"],
        ["训练目标", "y = log10(max(HV_reference - HV, 0) + 1)"],
        ["结果还原", "HV_pred = HV_reference + 1 - 10**y_pred"],
        ["参考值", "HV_reference = 43471.70365440；为该问题的论文参考值"],
    ], [28, 146]))
    add(p("二 数据来源与样本构造", "heading"))
    add(p("使用已收集论文Quantum approximate multi-objective optimization的图4配套公开模拟实验数据。36个CSV覆盖p取1至6、chi取1、5、10、20、30、50的全部36种组合；每组合5次重复，每次100个时刻，t从25秒至2500秒。共36×5×100=18,000条样本。原始ZIP内嵌在notebook中，解码后核验SHA-256，无需另行下载数据。"))
    add(p("保留图4全部样本，并为每条标签保存源文件、行号和重复编号。检查缺失值、数值有效性、时间顺序及HV范围；max只消除约1e-11的相减舍入误差。hv_mean、上下界及已观测HV不作为输入。", "small"))
    add(PageBreak())

    # Page 2: modelling and evaluation protocol.
    add(p("三 建模流程与训练方案", "heading"))
    add(p("流程为：原始数据校验 → 展开监督样本 → 按参数组划分训练与测试 → 训练集5折分组调参 → 用完整训练集拟合 → 独立测试 → 误差分析与模型保存。"))
    add(p("固定随机种子42，以完整(p,chi)组合划分：27组、13,500条用于训练；9组、4,500条用于测试。一个组合的所有重复与时刻始终处于同一集合。逐行随机划分会把高度相似的相邻时间点分散到两边，夸大泛化能力，因此采用分组划分。"))
    add(p("仅在训练集内部使用5折GroupKFold，每折仍按完整参数组合分组。每个模型搜索4组超参数，共20次CV拟合，三类算法合计60次CV拟合。选择依据为平均RMSE(log)，再用全部训练样本拟合。多项式变换和标准化均封装在Pipeline内，避免使用验证折的信息。"))
    add(table([
        ["模型", "建模作用", "搜索范围及本次结果"],
        ["多项式岭回归", "带正则化的平滑非线性模型", "次数2/3；alpha为0.1/10。选中次数3、alpha=0.1。"],
        ["RBF支持向量回归", "通过核函数学习非线性关系", "C为10/100；gamma为scale/0.1；epsilon固定0.05。选中C=10、gamma=0.1。"],
        ["随机森林回归", "用多棵树拟合非线性及交互关系", "120棵树；深度12/不限；最少叶样本5/20。选中深度12、最少叶样本5。"],
        ["常数基线", "判断模型是否学到有效规律", "仅输出训练标签均值，不使用测试标签。"],
    ], [33, 47, 94]))
    add(p("四 综合评价方法", "heading"))
    add(p("同时对训练标签y和还原后的原始HV计算MAE、RMSE、R²。MAE概括平均偏差，RMSE对较大误差更敏感，R²衡量相对于真实标签均值预测的改善。MAE和RMSE越小越好，R²越接近1越好；R²为负表示表现弱于该评价集均值预测。"))
    add(p("MAE = mean(abs(y - y_pred))\nRMSE = sqrt(mean((y - y_pred)**2))\nR² = 1 - sum((y - y_pred)**2) / sum((y - mean(y))**2)", "small"))
    add(p("另外报告CV折间标准差、训练与测试误差差异、每个测试参数组的误差和残差分布。95%区间按9个完整测试参数组bootstrap重采样2000次计算。区间只描述本次测试参数集的不确定性，不代表新问题实例或重新训练的波动；18000个相关观测不是18000个独立实验。"))
    add(PageBreak())

    # Page 3: metrics and conclusions.
    add(p("五 建模结果与指标解释", "heading"))
    add(p("表1为对数目标的独立测试结果。CV列完全来自训练集，测试列不参与调参。", "small"))
    rows = [["模型", "CV RMSE", "测试MAE", "测试RMSE", "测试R²"]]
    for name in ["ridge", "svr", "random_forest", "baseline"]:
        row = test.loc[name]
        rows.append([names[name], f"{cv.loc[name].cv_rmse_log:.4f}", f"{row.mae_log:.4f}",
                     f"{row.rmse_log:.4f}", f"{row.r2_log:.4f}"])
    add(table(rows, [54, 30, 30, 30, 30]))
    add(Spacer(1, 8))
    add(p("表2为还原至原始HV尺度后的独立测试结果。", "small"))
    rows = [["模型", "MAE HV", "RMSE HV", "R² HV"]]
    for name in ["ridge", "svr", "random_forest", "baseline"]:
        row = test.loc[name]
        rows.append([names[name], f"{row.mae_hv:.2f}", f"{row.rmse_hv:.2f}", f"{row.r2_hv:.4f}"])
    add(table(rows, [54, 40, 40, 40]))
    add(p("结果分析", "heading"))
    add(p(f"训练CV选出{names[selected]}，CV平均RMSE(log)为{cv.loc[selected].cv_rmse_log:.4f}，折间标准差为{cv.loc[selected].cv_rmse_fold_sd:.4f}。其测试RMSE(log)为{svr.rmse_log:.4f}，相对常数基线降低{100*(1-svr.rmse_log/test.loc['baseline'].rmse_log):.2f}%，对应整组bootstrap的95%区间为[{svr.test_rmse_log_ci_low:.4f}, {svr.test_rmse_log_ci_high:.4f}]。"))
    add(p(f"随机森林在固定测试集上的数值表现最好：RMSE(log)为{forest.rmse_log:.4f}，R²(log)为{forest.r2_log:.4f}；原始HV的MAE为{forest.mae_hv:.2f}，RMSE为{forest.rmse_hv:.2f}，R²为{forest.r2_hv:.4f}。这是测试集观察，不据此重新选择超参数或替换原先的CV选择，也不声称模型差异达到统计显著。"))
    add(p("两种尺度应联合解读。支持向量回归的对数目标R²虽为0.9704，原始HV的R²却只有0.0999；在chi=5、p=1这一测试组合上，原始HV的RMSE达到11845.38。指数还原会放大大差距区域的偏差，因而较高的对数目标分数不能解释为原始HV预测同样精确。"))
    add(p("无约束回归还会给出负的对数差距，导致HV略高于参考值：岭回归有755条，支持向量回归有770条，随机森林为0条。本作业保留这些输出，没有在查看测试结果后截断分数。若实际目标侧重原始HV，应事先使用该尺度调参，并另留新的独立测试集。", "small"))
    add(PageBreak())

    # Page 4: overall performance and calibration.
    add(p("六 评价图表", "heading"))
    add(figure("01_model_comparison"))
    add(p("图1  测试RMSE对比。点为整体RMSE(log)，误差线为按9个完整测试参数组重采样2000次得到的95%百分位区间。所有学习模型的测试误差均低于常数基线。", "caption"))
    add(figure("02_observed_vs_predicted"))
    add(p("图2  训练CV所选支持向量回归的全部4500条测试预测。横轴为真实对数差距，纵轴为预测值，虚线表示理想预测，颜色区分chi。远离虚线的区域显示局部偏差，相同输入的5次重复存在观测差异。", "caption"))
    add(PageBreak())

    # Page 5: residual diagnostics and terminal configurations.
    add(p("七 残差与分参数组分析", "heading"))
    add(figure("03_residual_distribution"))
    add(p("图3  测试残差为预测值减真实值。箱体为四分位距，中线为中位数，须为1.5倍四分位距内的范围，点为其余观测。随机森林多数残差较集中，但仍有长尾，不能仅看箱体宽度判断全部参数组合均准确。", "caption"))
    add(figure("04_terminal_configuration_errors"))
    add(p("图4  全部9个测试组合在2500秒时的预测。灰点为5次重复的对数差距均值，误差线为最小值至最大值，并非置信区间；菱形为CV所选支持向量回归预测。chi5/p1及部分接近收敛的组合存在可见偏差。", "caption"))
    add(PageBreak())

    # Page 6: limitations, reproducibility and submission mapping.
    add(p("八 总结与适用范围", "heading"))
    add(p("本作业将实验结果转化为一个输入明确、标签有来源的小型监督回归任务，实际训练三类模型并设置基线。通过分组划分和分组交叉验证，评价的是同一固定问题中未见参数组合的表现。结合多指标、两种尺度和误差图，可以识别只看平均分数时容易忽略的局部失败。"))
    add(p("数据来自同一个42节点三目标问题；结论不能直接推广到新图实例、其他目标数或真实量子硬件。MPS的时间是论文采用的折算采样时间，不是本机训练耗时。模型预测给定设置的结果质量，不直接输出优化解集，也不证明量子优势。"))
    add(p("九 可复现性与已运行证据", "heading"))
    add(p(f"配套supervised_assignment.ipynb已通过真实Jupyter内核从头执行，共{run['executed_code_cells']}个代码单元，错误数为0。运行包含原始数据解码及SHA-256核验、18,000条样本提取、三类模型的60次CV拟合及最终拟合、独立测试、图表生成和独立复算。notebook_execution.json记录执行状态。"))
    add(p("独立核验逐条比对全部18,000条标签与原CSV，检查训练/测试参数组和特征组合没有交集，重新加载保存模型，复现每个模型的4500条测试预测，再重新计算各项评价指标。验证结果为PASS。"))
    add(table([
        ["作业要求", "本次完成内容", "对应材料"],
        ["提炼有监督小任务", "从36个CSV构造三特征回归任务，18,000条标签有明确来源", "notebook第1至2节；监督样本CSV"],
        ["至少三种模型", "多项式岭回归、RBF支持向量回归、随机森林；另设基线", "notebook第3节；调参表与模型文件"],
        ["综合性能评价", "MAE、RMSE、R²、分组CV、整组bootstrap、残差及参数组分析", "notebook第4至6节；本说明第3至5页"],
        ["运行并提交", "notebook已执行，PDF与完整附件已生成", "ipynb、PDF、完整提交ZIP"],
    ], [34, 91, 49]))
    add(p("十 运行与提交材料", "heading"))
    add(p("将supervised_assignment.ipynb导入AI Studio，使用Python 3.10及以上CPU环境，从上到下运行全部单元。原始ZIP已内嵌，通常无需上传其他数据。运行后结果保存在output/supervised_learning中。提交说明文档使用本PDF；如平台支持附加文件，可一并提交完整ZIP。", "small"))
    add(p("参考资料与数据来源", "heading"))
    add(p("Quantum approximate multi-objective optimization. DOI: 10.1038/s43588-025-00873-y。数据为该论文图4配套公开源CSV，原始ZIP及其SHA-256收录于notebook与附件。机器学习模型、超参数与评价数值均来自本次独立建模，不作为原论文结论引用。", "small"))

    path = OUT / "监督学习建模作业说明.pdf"
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                                 topMargin=17*mm, bottomMargin=18*mm, title="基于实验数据的优化质量预测",
                                 author="", subject="监督学习作业的建模背景过程与结果")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print("PDF created:", path.name)


if __name__ == "__main__":
    main()
