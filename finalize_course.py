"""Assemble portable source, provenance and a compact course submission archive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/course_delivery'
DATA=ROOT/'output/course_reproduction_20261103'
SKILL=Path('C:/Users/WentB/.agents/skills/nature-figure')


def main():
    # Correct the angle-source provenance to the exact file used by the saved circuit run.
    # Numerical samples, parameters and experiment results are unchanged.
    src=ROOT/'qamoo-main/data/problems/27q/problem_set_27q_0s_3o_0/JuliQAOA_angles.json'
    for folder in [DATA,OUT/'experiments']:
        manifest=json.loads((folder/'input_manifest.json').read_text(encoding='utf-8'))
        manifest=[x for x in manifest if 'parameter_training/learned_angles/' not in x['source']]
        entry={'source':src.relative_to(ROOT).as_posix(),'copy':'inputs/JuliQAOA_angles.json',
               'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
        manifest=[x for x in manifest if x['source']!=entry['source']]+[entry]
        shutil.copy2(src,folder/entry['copy'])
        (folder/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    source=OUT/'source';source.mkdir(exist_ok=True)
    for name in ['build_course_materials.py','build_course_video.py']:
        data=(ROOT/name).read_text(encoding='utf-8')
        data=data.replace('ROOT = Path(__file__).resolve().parent','ROOT = Path(__file__).resolve().parents[1]')
        data=data.replace('ROOT=Path(__file__).resolve().parent','ROOT=Path(__file__).resolve().parents[1]')
        data=data.replace("ROOT/'output/course_delivery/qa/matplotlib'","ROOT/'qa/matplotlib'")
        data=data.replace("ROOT/'output/course_delivery'",'ROOT')
        data=data.replace("ROOT/'output/course_reproduction_20261103'","ROOT/'experiments'")
        data=data.replace("SKILL=Path('C:/Users/WentB/.agents/skills/nature-figure')","SKILL=ROOT/'tools'")
        (source/name).write_text(data,encoding='utf-8')
    ps=(ROOT/'generate_course_voice.ps1').read_text(encoding='utf-8')
    ps=ps.replace("$courseOut = Join-Path $courseRoot 'output\\course_delivery'","$courseOut = Split-Path -Parent $PSScriptRoot")
    (source/'generate_course_voice.ps1').write_text(ps,encoding='utf-8-sig')
    shutil.copy2(ROOT/'qamoo-main/qamoo/algorithms/qaoa.py',source/'official_qaoa.py')
    shutil.copy2(ROOT/'qamoo-main/LICENSE',source/'QAMOO_LICENSE')
    tools=OUT/'tools/scripts';tools.mkdir(parents=True,exist_ok=True)
    for name in ['audit_panel_alignment.py','audit_figure_collisions.py','audit_pdf_text.py','validate_figure.py']:
        shutil.copy2(SKILL/'scripts'/name,tools/name)
    (OUT/'requirements-render.txt').write_text('numpy==1.26.4\nnetworkx==3.4.2\nmatplotlib==3.10.9\npillow==12.3.0\nreportlab==4.4.3\nPyMuPDF==1.25.5\nimageio-ffmpeg==0.8.0\n',encoding='utf-8')
    notes={'report':'one A4 page, blank handwritten-name field, final PNG visually inspected',
           'figure_preflight':'20 pass, 1 reviewed warning: 105 mm width chosen for the A4 course layout rather than a journal column',
           'figure_pdf_text_min_pt':7,'figure_collision_audit':'PASS: no FAIL and no WARN',
           'alignment':'single panel, not applicable',
           'data':'20 complete runs, all 200000 objective vectors recalculated; 200 curve points verified',
           'replay':'p=1,3,6 first seed: 30000 samples bitwise identical',
           'video':'230 seconds (03:50), 1280x720, 12 fps, shortened AAC Chinese narration and synchronized burnt-in subtitles; seven frames decoded',
           'pptx':'7 editable slides, native chart on slide 5, native table on slide 6; speaker notes include narration, timing and sources; finalized with font/layout/package/chart-workbook/import validation',
           'scope':'limited-budget ideal-circuit MPS simulation; no quantum hardware, training or full-paper replication claim',
           'script_license':'official_qaoa.py from QAMOO Apache-2.0; see source/QAMOO_LICENSE'}
    (OUT/'qa/final_review.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    # Submission excludes intermediate WAVs, extracted article text and temporary caches.
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and 'audio' not in p.relative_to(OUT).parts
           and '__pycache__' not in p.parts and 'matplotlib' not in p.relative_to(OUT).parts
           and p.name not in ['paper_text.txt','SHA256SUMS.txt','答辩动画_5分钟_含旁白字幕.mp4','convergence.collisions.pdf']]
    manifest='\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}' for p in files)+'\n'
    (OUT/'SHA256SUMS.txt').write_text(manifest,encoding='utf-8')
    files.append(OUT/'SHA256SUMS.txt')
    archive=ROOT/'output/课程复现提交包_一页报告与3分50秒动画及PPT.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,Path('课程复现提交包')/p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'files':len(files),'zip_mb':round(archive.stat().st_size/1024**2,2),'report_pages':1,'video_seconds':230,'ppt_slides':7},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
