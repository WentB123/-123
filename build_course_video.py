"""Render an offline, narrated teaching animation under four minutes.

Animation is a presentation of completed measurements, never a running simulator.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import wave

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import networkx as nx

OUT=ROOT/'output/course_delivery'
DATA=ROOT/'output/course_reproduction_20261103'
AUDIO=OUT/'audio'
QA=OUT/'qa'
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
W,H,FPS=1280,720,12
BG='#f6f8fa';INK='#173345';MUTED='#5f727e';TEAL='#087e82';BLUE='#416e9b';GREY='#86929d';GOLD='#cb8d36'
COLORS=[GREY,'#a6b9d0',BLUE,TEAL]
SCENES=json.loads((OUT/'storyboard.json').read_text(encoding='utf-8'))
DURATION=int(sum(s['duration'] for s in SCENES))
assert DURATION==230 and DURATION<240
TOTAL_STAMP=f'{DURATION//60:02}:{DURATION%60:02}'
NARRATION_FILE=f'narration_{DURATION}s.wav'
SUMMARY=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
CURVES=list(csv.DictReader((DATA/'curves.csv').open(encoding='utf-8-sig')))
SEEDS=json.loads((DATA/'config.json').read_text(encoding='utf-8'))['seeds']
FONTS={size:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size) for size in [16,18,20,22,24,26,28,30,34,38,44,48,52,60,72]}
BOLD={size:ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',size) for size in FONTS}


def dump(p,o):
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')


def wrap(text,max_width,font):
    rows=[];row=''
    for char in text:
        if char=='\n':
            rows.append(row);row='';continue
        if font.getlength(row+char)>max_width:
            if char in '，。；：！？、）】》':
                rows.append(row+char);row='';continue
            rows.append(row);row=char
        else: row+=char
    if row:rows.append(row)
    return rows


def text(draw,pos,s,size=26,color=INK,bold=False):
    draw.text(pos,str(s),font=(BOLD if bold else FONTS)[size],fill=color)


def multiline(draw,pos,s,size=26,color=INK,width=650,spacing=12,bold=False):
    font=(BOLD if bold else FONTS)[size]
    for i,row in enumerate(wrap(s,width,font)):
        text(draw,(pos[0],pos[1]+i*(size+spacing)),row,size,color,bold)


def audio_and_subtitles():
    captions=[];timeline=[]
    total=np.zeros(DURATION*24000,dtype=np.int16)
    for s in SCENES:
        # Allocate each paragraph a proportional slot within its fixed chapter.
        weights=np.array([len(x) for x in s['narration']],float)
        slots=(s['duration']-.8)*weights/weights.sum()
        cursor=s['start']+.4
        for j,(line,slot) in enumerate(zip(s['narration'],slots)):
            src=AUDIO/f"voice_{s['index']}_{j}.wav"
            with wave.open(str(src)) as f:
                actual=f.getnframes()/f.getframerate()
            target=float(slot)-.2
            # Keep normal speech when it fits; accelerate only paragraphs exceeding their slot.
            speed=max(1.0,actual/target)
            assert speed<1.7,(s['index'],j,speed)
            cmd=[FFMPEG,'-hide_banner','-loglevel','error','-i',str(src),'-af',f'atempo={speed:.7f}',
                 '-f','s16le','-ac','1','-ar','24000','pipe:1']
            pcm=np.frombuffer(subprocess.check_output(cmd),dtype=np.int16)
            start=int(cursor*24000);end=start+len(pcm)
            assert end<=(s['start']+s['duration'])*24000
            total[start:end]=pcm
            duration=len(pcm)/24000
            chunks=[];chunk=''
            for part in re.split(r'(?<=[，。；：！？])',line):
                if len(chunk+part)>48 and chunk:
                    chunks.append(chunk);chunk=''
                chunk+=part
            if chunk:chunks.append(chunk)
            t=cursor
            for chunk in chunks:
                dt=duration*len(chunk)/len(line)
                captions.append({'start':t,'end':t+dt,'text':chunk})
                t+=dt
            timeline.append({'scene':s['index'],'paragraph':j,'start':cursor,'slot':float(slot),
                             'speech_seconds':duration,'tempo':speed})
            cursor+=float(slot)
    with wave.open(str(AUDIO/NARRATION_FILE),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(total.tobytes())
    dump(OUT/'captions.json',captions);dump(QA/'voice_timing.json',timeline)
    def stamp(t):
        ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    (OUT/'答辩动画_字幕.srt').write_text('\n\n'.join(f"{i+1}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{c['text']}" for i,c in enumerate(captions)),encoding='utf-8-sig')
    return captions


def load_graph():
    obj=json.loads((DATA/'inputs/problem_graph_0.json').read_text())
    g=nx.node_link_graph(obj,edges='links')
    pos=nx.kamada_kawai_layout(g,weight=None)
    arr=np.array([pos[i] for i in range(42)])
    arr=(arr-arr.min(0))/(arr.max(0)-arr.min(0))
    xy={i:(int(82+arr[i,0]*550),int(195+arr[i,1]*345)) for i in range(42)}
    bits=np.load(DATA/'runs'/f'p6_seed{SEEDS[0]}.npz')['bits']
    values=np.load(DATA/'runs'/f'p6_seed{SEEDS[0]}.npz')['objectives']
    return g,xy,bits,values


GRAPH,XY,BITS,VALUES=load_graph()


def base(scene,t,caption):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,12),fill=TEAL)
    text(d,(60,34),'PAPER REPRODUCTION  /  QAMOO',18,MUTED)
    text(d,(1103,34),f'{int(t)//60:02}:{int(t)%60:02} / {TOTAL_STAMP}',18,MUTED)
    text(d,(60,79),scene['title'],38,bold=True)
    text(d,(62,132),f"{scene['index']+1:02} / 07",18,TEAL)
    d.line((60,166,1220,166),fill='#dce4e9',width=2)
    d.rectangle((0,594,W,H),fill='#e9f0f3')
    if caption:
        lines=wrap(caption,1150,FONTS[26])
        assert len(lines)<=2,caption
        for j,line in enumerate(lines):
            width=FONTS[26].getlength(line)
            text(d,((W-width)/2,614+j*37),line,26)
    d.rectangle((0,H-5,int(W*t/DURATION),H),fill=TEAL)
    return im,d


def graph(draw,phase):
    idx=int(phase*5)%100
    bit=BITS[idx]
    for u,v in GRAPH.edges:
        cut=bit[u]!=bit[v]
        draw.line((*XY[u],*XY[v]),fill=TEAL if cut else '#d5dfe5',width=4 if cut else 2)
    for i,(x,y) in XY.items():
        fill=BLUE if bit[i]==0 else GOLD
        draw.ellipse((x-9,y-9,x+9,y+9),fill=fill,outline='white',width=2)
    return VALUES[idx]


def draw_curve(d,phase):
    x0,y0,x1,y1=130,225,900,520
    def px(x):return x0+(x-1)/9*(x1-x0)
    def py(y):return y1-(y-17)/27*(y1-y0)
    for y in [20,30,40]:
        d.line((x0,py(y),x1,py(y)),fill='#dce4e8',width=1)
        text(d,(75,py(y)-13),str(y),20,MUTED)
    for x in [1,4,7,10]:
        text(d,(px(x)-10,y1+12),str(x),20,MUTED)
    text(d,(85,184),'HV（千）',22,MUTED)
    text(d,(365,552),'累计样本数（千条）',22,MUTED)
    n=1+9*min(1,phase/.75)
    for z,p in enumerate([0,1,3,6]):
        arr=np.array([[float(c['hypervolume'])/1000 for c in CURVES if int(c['p'])==p and int(c['seed'])==seed] for seed in SEEDS])
        means=arr.mean(0)
        xs=np.r_[np.arange(1,int(n)+1),n] if n<10 else np.arange(1,11)
        ys=np.interp(xs,np.arange(1,11),means)
        points=[(px(x),py(y)) for x,y in zip(xs,ys)]
        # Bands are exact per-checkpoint min/max, linearly revealed between checkpoints.
        lower=np.interp(xs,np.arange(1,11),arr.min(0));upper=np.interp(xs,np.arange(1,11),arr.max(0))
        color=COLORS[z]
        rgb=tuple(int(color[i:i+2],16) for i in [1,3,5]);pastel=tuple(int(.15*v+.85*246) for v in rgb)
        poly=[(px(x),py(y)) for x,y in zip(xs,upper)]+[(px(x),py(y)) for x,y in zip(xs[::-1],lower[::-1])]
        if len(poly)>2:d.polygon(poly,fill=pastel)
        if len(points)>1:d.line(points,fill=color,width=4 if p==6 else 3)
        xx,yy=points[-1];d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color)
        label='均匀随机' if p==0 else f'QAOA p={p}'
        text(d,(950,230+z*57),label,24,color,bold=True)
        text(d,(950,259+z*57),f'{means[-1]*1000:,.0f}',22,MUTED)
    text(d,(950,483),'均值与 min–max',18,MUTED)
    text(d,(950,512),'每组 5 次重复',18,MUTED)


def render(scene,t,caption):
    im,d=base(scene,t,caption)
    elapsed=t-scene['start'];u=elapsed/scene['duration'];i=scene['index']
    if i==0:
        v=graph(d,u)
        text(d,(735,209),'42 节点 × 3 个目标',34,bold=True)
        text(d,(735,270),'一个切分，三个得分',28,TEAL)
        multiline(d,(735,334),'复用公开问题与电路\n重新生成测量样本',28,width=450)
        text(d,(735,456),'课程范围：本地 MPS 模拟',24,MUTED)
        text(d,(735,505),'非 IBM 真机重跑',22,MUTED)
    elif i==1:
        v=graph(d,elapsed*.45)
        text(d,(720,198),'当前真实样本的三目标值',28,bold=True)
        for j,value in enumerate(v):
            y=265+j*74
            text(d,(725,y),f'f{j+1}',28,[TEAL,BLUE,GOLD][j],True)
            d.line((790,y+37,1160,y+37),fill='#d9e2e7',width=2)
            text(d,(900,y-8),f'{value:.2f}',38,[TEAL,BLUE,GOLD][j],True)
        text(d,(722,514),'彩色连边：跨组边，计入加权得分',20,MUTED)
        text(d,(722,550),'橙 / 蓝：节点所属两组',20,MUTED)
    elif i==2:
        x0,y0,x1,y1=120,530,690,230
        d.line((x0,y1-15,x0,y0,x1+25,y0),fill=MUTED,width=3)
        text(d,(360,550),'目标一（越大越好）',22,MUTED)
        text(d,(70,190),'目标二（越大越好）',22,MUTED)
        pts=[(1,2),(2,1),(2,3),(3,2),(3,5),(4,3),(4,4),(5,3),(6,2)]
        front=[(3,5),(4,4),(5,3),(6,2)]
        scale=lambda p:(x0+72*p[0],y0-46*p[1])
        if u>.42:
            for p in front:
                x,y=scale(p);d.rectangle((x0,y,x,y0),fill='#d8eeeb',outline='#83bbb7',width=1)
        for k,p in enumerate(pts):
            if k>int(min(1,u/.35)*len(pts)):continue
            if u>.28 and p not in front:color='#c6d0d8';r=6
            else:color=TEAL if p in front else GREY;r=9
            x,y=scale(p);d.ellipse((x-r,y-r,x+r,y+r),fill=color)
        if u>.32:d.line([scale(p) for p in front],fill=TEAL,width=3)
        text(d,(790,227),'先保留非支配解',30,bold=True)
        text(d,(790,295),'再计算覆盖体积',30,TEAL,True)
        multiline(d,(790,375),'二维图仅作概念示意；实测 HV 使用三个目标与固定参考点。',24,width=410)
        text(d,(130,530),'参考点',18,MUTED)
    elif i==3:
        steps=[('公开输入','42 节点 / 3 目标'),('参数迁移','27 节点训练角度'),('QAOA 采样','p=1 / 3 / 6'),('评价解集','Pareto → HV')]
        for j,(title,sub) in enumerate(steps):
            x=65+j*305;y=225
            active=int(elapsed/3)%4==j
            d.rectangle((x,y,x+265,y+123),fill='#e0efee' if active else '#edf1f4')
            text(d,(x+17,y+16),title,28,TEAL if active else INK,True)
            text(d,(x+17,y+73),sub,18,MUTED)
            if j<3:text(d,(x+274,y+39),'→',26,MUTED)
        text(d,(105,403),'10 电路 × 1,000 shots × 5 个种子',34,bold=True)
        text(d,(105,467),'随机对照：同样每次 10,000 样本',28,TEAL)
        text(d,(105,532),'固定权重子集 · χ=20 · 共 20 万条新样本',24,MUTED)
    elif i==4:
        draw_curve(d,u)
    elif i==5:
        headers=['方法','HV 均值 ± SD','非支配解数均值']
        cols=[100,365,800]
        for x,h in zip(cols,headers):text(d,(x,205),h,26,bold=True)
        for j,s in enumerate(SUMMARY):
            y=267+j*59
            name=['均匀随机','QAOA p=1','QAOA p=3','QAOA p=6'][j]
            d.line((90,y+46,1180,y+46),fill='#d9e2e7',width=1)
            text(d,(100,y),name,26,COLORS[j],True)
            text(d,(365,y),f"{s['hypervolume_mean']:,.0f} ± {s['hypervolume_std']:.0f}",28)
            text(d,(800,y),f"{s['non_dominated_vectors_mean']:.1f}",28)
        text(d,(100,526),'20 / 20 次运行：独立 HV 计算核对通过',26,TEAL,True)
        text(d,(100,565),'SD 是测量重复的描述统计；不是置信区间。',18,MUTED)
    else:
        gain=(SUMMARY[3]['hypervolume_mean']/SUMMARY[0]['hypervolume_mean']-1)*100
        text(d,(85,204),'本实例：p=6 解集质量更好',38,TEAL,True)
        text(d,(88,274),f'平均 HV 比随机采样高 {gain:.1f}%',34,bold=True)
        d.line((90,350,1190,350),fill='#cddce1',width=2)
        text(d,(90,385),'本次完成',26,bold=True)
        multiline(d,(90,435),'20 次独立采样运行\n新样本、原始记录、指标核验',24,width=490)
        text(d,(705,385),'结论边界',26,bold=True)
        multiline(d,(705,435),'单实例、10 个固定权重；\n未重训、未跑真机，不证明量子加速。',24,width=485)
        text(d,(90,550),'每次 1 万样本 / 论文每次 2,500 万样本 = 1 / 2,500',22,MUTED)
    return im


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    captions=audio_and_subtitles()
    # Check all subtitle layouts before encoding the video.
    for c in captions:assert len(wrap(c['text'],1150,FONTS[26]))<=2,c
    for s in SCENES:
        t=s['start']+.65*s['duration']
        cap=next((c['text'] for c in captions if c['start']<=t<c['end']),'')
        render(s,t,cap).save(QA/f"scene_{s['index']+1}.png")
    if args.preview:return
    path=OUT/'答辩动画_3分50秒_含旁白字幕.mp4'
    log=(QA/'video_encode.log').open('w',encoding='utf-8')
    cmd=[FFMPEG,'-y','-hide_banner','-loglevel','warning','-f','rawvideo','-vcodec','rawvideo','-s',f'{W}x{H}',
         '-pix_fmt','rgb24','-r',str(FPS),'-i','pipe:0','-i',str(AUDIO/NARRATION_FILE),
         '-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',
         '-t',str(DURATION),'-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
    ci=0;si=0
    try:
        for f in range(DURATION*FPS):
            t=f/FPS
            while si<6 and t>=SCENES[si+1]['start']:si+=1
            while ci<len(captions) and captions[ci]['end']<=t:ci+=1
            cap=captions[ci]['text'] if ci<len(captions) and captions[ci]['start']<=t else ''
            proc.stdin.write(render(SCENES[si],t,cap).tobytes())
            if f%(30*FPS)==0:print(f'Encoded {t:.0f}/{DURATION} seconds',flush=True)
        proc.stdin.close();assert proc.wait()==0
    finally:
        log.close()
    info=subprocess.run([FFMPEG,'-hide_banner','-i',str(path)],capture_output=True).stderr.decode('utf-8',errors='replace')
    (QA/'video_metadata.txt').write_text(info,encoding='utf-8')
    assert f'00:{TOTAL_STAMP}.00' in info and '1280x720' in info and 'Audio: aac' in info
    # Decode representative frames from the actual encoded file.
    for i,t in enumerate([s['start']+.6*s['duration'] for s in SCENES]):
        subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-y','-ss',str(t),'-i',str(path),'-frames:v','1',str(QA/f'encoded_{i+1}.png')],check=True)
    dump(QA/'video_validation.json',{'seconds':DURATION,'width':W,'height':H,'fps':FPS,'chapters':7,'audio':'AAC Chinese synthetic voice',
                                    'subtitle_cues':len(captions),'all_subtitles_max_two_lines':True,
                                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoded_representative_frames':7})
    print('VIDEO COMPLETE',flush=True)


if __name__=='__main__':main()
