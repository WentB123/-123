"""Build a one-sided A4 report and a five-minute narrated animation storyboard."""
from pathlib import Path
import csv
import json
import os
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
os.environ['MPLCONFIGDIR'] = str(ROOT/'output/course_delivery/qa/matplotlib')
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

DATA=ROOT/'output/course_reproduction_20261103'
OUT=ROOT/'output/course_delivery'
OUT.mkdir(exist_ok=True)
QA=OUT/'qa';QA.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
SKILL=Path('C:/Users/WentB/.agents/skills/nature-figure')
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    config=json.loads((DATA/'config.json').read_text(encoding='utf-8'))
    curves=list(csv.DictReader((DATA/'curves.csv').open(encoding='utf-8-sig')))
    gain=(summary[3]['hypervolume_mean']/summary[0]['hypervolume_mean']-1)*100
    hvgain=(summary[3]['hypervolume_mean']/summary[2]['hypervolume_mean']-1)*100
    refhv=43471.70365440166
    contract={'claim':'在这一固定实例、固定10个权重和每次10000样本预算下，p=6的平均HV高于均匀随机和p=3；不推断量子加速。',
              'archetype':'quantitative grid, single panel','backend':'python','dimensions_mm':[105,62],
              'n':'five measurement seeds per method; one instance and one fixed weight subset',
              'spread':'sample SD in table; min-max in convergence bands, neither is a confidence interval',
              'source':'all 200 rows of curves.csv and all 20 independent sampling runs',
              'projection_for_animation':'first seed only, every twentieth shot (500/10000) for visual clarity, no filtering by performance; full samples used for HV',
              'comparison':'uniform random, same number of samples; no assertion of equal computation time'}
    dump(QA/'figure_contract.json',contract)
    font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['SimHei','DejaVu Sans'],
                         'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'legend.frameon':False})
    fig,ax=plt.subplots(figsize=(105/25.4,62/25.4))
    fig.subplots_adjust(left=.18,bottom=.23,right=.96,top=.76)
    palette=['#7c8593','#a6b9d0','#4775a4','#0e807f']
    labels=['均匀随机','QAOA p=1','QAOA p=3','QAOA p=6']
    for idx,p in enumerate([0,1,3,6]):
        v=np.array([[float(r['hypervolume']) for r in curves if int(r['p'])==p and int(r['seed'])==seed] for seed in config['seeds']])
        x=np.arange(1,11)
        ax.fill_between(x,v.min(0)/1000,v.max(0)/1000,color=palette[idx],alpha=.12,linewidth=0)
        ax.plot(x,v.mean(0)/1000,color=palette[idx],lw=1.5 if p==6 else 1.1,label=labels[idx],ls='-' if p in [0,6] else '--')
    ax.set(xlabel='累计样本数（千条）',ylabel='超体积 HV（千）',xlim=(1,10),ylim=(17,44))
    ax.set_xticks([1,4,7,10]);ax.set_yticks([20,30,40])
    ax.grid(axis='y',color='#e3e7ea',linewidth=.5)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),borderaxespad=0,ncol=2,fontsize=7,handlelength=1.7,labelspacing=.5)
    require_matplotlib_panel_alignment(fig,json_out=str(QA/'convergence.alignment.json'),strict=True)
    fig.savefig(FIG/'convergence.pdf')
    fig.savefig(FIG/'convergence.svg')
    fig.savefig(FIG/'convergence.png',dpi=600)
    fig.savefig(FIG/'convergence.tiff',dpi=600)
    plt.close(fig)
    # Slide-size version preserves all observations; only typography/layout changes.
    image=Image.open(FIG/'convergence.png')
    assert image.width>1000
    report(summary,config,gain,hvgain,refhv)
    storyboard(summary,config,gain,hvgain,refhv)
    print(json.dumps({'gain_percent':gain,'p6_vs_p3_percent':hvgain,'report':'one A4 page','storyboard_seconds':230},ensure_ascii=False),flush=True)


def report(summary,config,gain,hvgain,refhv):
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simhei.ttf'))
    c=canvas.Canvas(str(OUT/'课程报告_A4单面.pdf'),pagesize=A4)
    c.setTitle('量子多目标优化实验复现 - A4单面课程报告')
    c.setAuthor('课程实验复现')
    w,h=A4;left=39;right=w-39;cw=right-left
    ink=colors.HexColor('#163042');teal=colors.HexColor('#087e82');muted=colors.HexColor('#536571')
    c.setFillColor(teal);c.rect(left,h-42,32,3,fill=1,stroke=0)
    def text(x,y,s,size=10,col=ink):
        c.setFillColor(col);c.setFont('CN',size);c.drawString(x,y,s)
    def para(s,y,size=10.2,leading=15,width=cw,x=left):
        style=ParagraphStyle('p',fontName='CN',fontSize=size,leading=leading,textColor=ink,wordWrap='CJK')
        p=Paragraph(escape(s),style);_,ph=p.wrap(width,700);p.drawOn(c,x,y-ph)
        return y-ph
    def head(num,title,y):
        text(left,y,num,10,teal);text(left+23,y,title,12.3,ink)
        c.setStrokeColor(colors.HexColor('#d7e0e4'));c.setLineWidth(.4);c.line(left,y-8,right,y-8)
    text(left,h-67,'量子多目标优化实验复现',21)
    text(left,h-87,'Quantum approximate multi-objective optimization',9,muted)
    text(left,h-110,'姓名（打印后手写）：________________    日期：____________',10)

    head('01','复现任务',h-141)
    para('复现论文的三目标加权最大割（MO-MAXCUT）流程：用 QAOA 电路采样候选切分，保留 Pareto 非支配解，再以超体积 HV 评价解集质量。验证在固定样本预算下，增加电路深度能否改善解集。',h-161)
    head('02','数据集与复现方案',h-222)
    para('数据：作者公开的 42 节点、3 个加权目标图（同一 heavy-hex 拓扑，实例 0）；复用在 27 节点上训练的角度与官方前 10 个权重向量，无训练/测试划分。',h-242)
    para('方案：Qiskit Aer 理想电路 MPS 近似模拟，键维数 χ=20；p=1、3、6，各 10 电路×1,000 shots，5 个测量种子；均匀随机对照也每次生成 10,000 个切分，共 20 次运行、200,000 条新样本。',h-279)
    para('评价：三个目标均最大化；HV 使用作者下界参考点（-12.1374，-19.6415，-18.3306），越大越好。固定 QPY 电路与参数，记录种子、版本、输入校验值和逐条位串。',h-326)
    head('03','本机实验结果',h-382)
    c.drawImage(str(FIG/'convergence.png'),left-8,h-579,width=291,height=172,mask='auto')
    text(left+298,h-413,'10,000 样本后的结果',10.2)
    body=[['方法','HV 均值 ± SD','非支配解数*']]
    for name,s in zip(['随机','p=1','p=3','p=6'],summary):
        body.append([name,f"{s['hypervolume_mean']:.0f} ± {s['hypervolume_std']:.0f}",f"{s['non_dominated_vectors_mean']:.1f}"])
    tbl=Table(body,colWidths=[34,112,70],rowHeights=[24,24,24,24,24])
    tbl.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'CN'),('FONTSIZE',(0,0),(-1,0),8),('FONTSIZE',(0,1),(-1,-1),9),
                             ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TEXTCOLOR',(0,0),(-1,-1),ink),
                             ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#ecf3f4')),
                             ('LINEBELOW',(0,0),(-1,0),.6,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce4e7')),
                             ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]))
    tbl.wrapOn(c,216,200);tbl.drawOn(c,left+292,h-549)
    para('*唯一非支配目标向量数的均值。n=5 次测量重复；SD 为样本标准差，曲线阴影为最小至最大值，均非置信区间。',h-588,size=8.6,leading=12)
    head('04','结论与复现边界',h-630)
    para(f"p=6 平均 HV 比随机采样高 {gain:.1f}%，比 p=3 高 {hvgain:.1f}%；说明在本实例和固定权重子集上，较深 QAOA 得到了更好的近似解集。全部 20 次运行已完成，HV 与独立三维几何实现逐次核对通过。",h-650)
    para('本次为缩小预算的模拟复现：每次样本数为论文三目标实验的 1/2,500。未重跑 27 节点参数训练、IBM 真机或完整经典求解器对比；同样本数不等于同耗时，本结果不证明量子加速或普适量子优势。',h-701,size=9.5,leading=14)
    para('论文：Kotil 等，Nature Computational Science 5, 1168–1177 (2025). DOI: 10.1038/s43588-025-00873-y\n代码与数据：github.com/stefan-woerner/qamoo；Zenodo: 10.5281/zenodo.16878920。',h-759,size=8,leading=11)
    c.showPage();c.save()
    d=fitz.open(OUT/'课程报告_A4单面.pdf')
    assert len(d)==1 and abs(d[0].rect.width-595.276)<.1 and abs(d[0].rect.height-841.89)<.1
    txt=d[0].get_text()
    assert '姓名' in txt and '200,000' in txt
    # All text and image blocks must fit the printable margins.
    for block in d[0].get_text('blocks'):
        assert block[0]>=20 and block[1]>=20 and block[2]<=w-20 and block[3]<=h-20,block[:4]
    d[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(QA/'report_preview.png')
    dump(QA/'report_validation.json',{'pages':len(d),'paper':'A4','width_pt':d[0].rect.width,'height_pt':d[0].rect.height,'name_field_blank':True,'all_blocks_within_20pt':True})


def storyboard(summary,config,gain,hvgain,refhv):
    # Shortened narration retains the measured results and scope in 3:50.
    scenes=[
      ('任务：寻找一组好的折中方案',22,[
       '本次复现《量子近似多目标优化》，研究多个冲突目标之间的折中方案。',
       '我使用作者的四十二节点、三目标最大割问题，在本机重新采样，比较不同深度的 Q A O A 与随机采样。']),
      ('问题：一张图，三个评分标准',30,[
       '最大割把节点分成两组。只有两端分属不同组的边才计入得分，得分等于这些边的权重之和。',
       '实验包含四十二个节点，三个目标图共享连接结构，但边权不同。因此，一个四十二位的零一切分向量，会得到三个分数。',
       '动画显示本次真实样本的切分和目标值，彩色连边表示跨组边。']),
      ('评价：Pareto 前沿与超体积',30,[
       '如果一个方案的所有目标都不差于另一个方案，并且至少一个更好，就称它支配另一个方案。',
       '删掉被支配的方案，剩下的非支配解集用来近似 Pareto 前沿。超体积，也就是 H V，衡量这些点相对于固定参考点覆盖的目标空间，越大越好。',
       '这里的二维面积只解释概念，实际实验计算三目标超体积。']),
      ('方案：固定输入，重新采样',38,[
       '我复用作者在二十七节点上训练的角度，和公开权重中的前十个向量，所有组共用这些输入。',
       '采用 Qiskit Aer 的矩阵乘积态近似模拟，键维数二十。比较深度一、三、六，每组十个电路，每电路采样一千次，重复五次。',
       '均匀随机对照每次也生成一万个切分。合计二十次运行、二十万条新样本，保存位串、目标值、种子和软件版本。']),
      ('结果：样本增加时，解集怎样改善',48,[
       '曲线来自本次实测。横轴是累计样本数，纵轴是超体积；曲线表示五次重复的均值，阴影是最小到最大值，不是置信区间。',
       '同样本数下，深度三和六明显优于随机，深度一接近随机。采样一万条后，深度六的平均超体积最高。',
       f"相对随机采样，深度六的平均超体积提高约百分之{gain:.1f}；相对深度三提高约百分之{hvgain:.1f}。这说明较深电路在本实例与固定权重子集上改善了解集。",
       '这里比较的是解集质量。同样本数不等于同耗时，不能据此推断量子计算更快。']),
      ('验证：每个数字都能追溯',28,[
       f"深度六的平均超体积约为{summary[3]['hypervolume_mean']:.0f}，随机对照约为{summary[0]['hypervolume_mean']:.0f}，表中同时给出标准差和非支配解数。",
       '全部二十次运行的超体积与独立三维实现核对通过；全部目标值已重新计算，另外三万条样本精确重放逐位一致。',
       '五次重复只衡量固定问题下的测量波动，不代表跨问题泛化。']),
      ('结论：完成了什么，还能说明多少',34,[
       '本次完成了缩小预算的模拟复现：从公开输入生成新样本，筛选非支配解，计算超体积，并与随机采样比较。深度六在本实例上得到更好的解集。',
       '每次一万样本，是论文三目标实验预算的两千五百分之一。本次复用了训练角度，没有重跑真机，也未完成全部强经典算法对比。',
       '因此结论限定为模拟中的解集质量改善，不宣称量子加速。谢谢。'])
    ]
    assert sum(x[1] for x in scenes)==230
    payload=[];start=0
    for i,(title,duration,paragraphs) in enumerate(scenes):
        payload.append({'index':i,'title':title,'start':start,'duration':duration,'narration':paragraphs})
        start+=duration
    dump(OUT/'storyboard.json',payload)
    lines=['# 3 分 50 秒答辩讲稿','', '动画：7 个章节，总长 03:50；含本地合成中文旁白和字幕，距四分钟上限留出 10 秒。','']
    for s in payload:
        a=s['start'];b=a+s['duration']
        lines += [f"## {a//60:02}:{a%60:02}–{b//60:02}:{b%60:02} {s['title']}",'',*s['narration'],'']
    lines += ['## 可能的答辩问题','',
              '**这是实际量子计算机实验吗？** 是经典计算机上的理想电路 MPS 近似模拟；没有访问 IBM 真机。','',
              '**为什么没有训练测试集？** 这是组合优化复现，目标是搜索一个固定问题的解集，非监督学习预测任务。','',
              '**为什么用五次重复？** 衡量固定实例和权重集合下测量种子的波动；不代表跨实例统计。','',
              '**能否证明量子优势？** 不能。随机采样只是入门基线，既未比较全部强经典算法，也未控制相同墙钟时间。','',
              '**为什么不采用旧的大预算结果？** 旧任务只有部分完成，不能将中间检查点当最终实验。此处采用独立目录中全部完成的二十次实验。','',
              '**复用了什么？** 作者图实例、27 节点训练角度、前 10 个权重、固定 QPY 电路；未把作者历史位串当作本次新测量。','',
              '**HV 的参考点是什么？** 作者三个目标的下界；在全部方法和重复中保持不变。','',
              '**动画中的图就是全部样本吗？** 图切分取自真实位串；二维 Pareto 面积图是明确标注的教学示意；收敛曲线与指标使用全部样本。']
    (OUT/'答辩讲稿与问答.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
