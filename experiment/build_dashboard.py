# -*- coding: utf-8 -*-
"""把 results/analysis/*.csv 合并成一个自包含交互式 HTML 仪表盘,写到 EAI Project 根目录。"""
import os, json
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "results", "analysis")
OUT_HTML = os.path.abspath(os.path.join(HERE, "..", "SOCRATES_对比分析_仪表盘.html"))

gs = pd.read_csv(os.path.join(A, "group_summary.csv")).set_index("model")
bs = pd.read_csv(os.path.join(A, "bootstrap_summary.csv")).set_index("model")
md = pd.read_csv(os.path.join(A, "metad_summary.csv")).set_index("model")
byt = pd.read_csv(os.path.join(A, "by_type.csv")).set_index("model")
hl = pd.read_csv(os.path.join(A, "hallucination.csv")).set_index("model")
hm = pd.read_csv(os.path.join(A, "hmetad_final.csv")).set_index("model")

def g(df, m, c, d=None):
    try:
        v = df.loc[m, c]
        return None if (isinstance(v, float) and np.isnan(v)) else v
    except Exception:
        return d

rows = []
for m in gs.index:
    rows.append(dict(
        model=m, tier=str(gs.loc[m, "tier"]), vendor=str(gs.loc[m, "vendor"]),
        thinking=bool(gs.loc[m, "thinking"]), condition=str(gs.loc[m, "condition"]),
        n=int(gs.loc[m, "n"]), acc=round(float(gs.loc[m, "acc"]), 4),
        err=int(gs.loc[m, "n_wrong"]), mean_conf=round(float(gs.loc[m, "mean_conf"]), 3),
        overconf=round(float(gs.loc[m, "overconf"]), 4),
        ece=round(float(bs.loc[m, "ece"]), 4),
        ece_lo=round(float(bs.loc[m, "ece_lo"]), 4), ece_hi=round(float(bs.loc[m, "ece_hi"]), 4),
        auroc2=(round(float(g(bs, m, "auroc2")), 3) if g(bs, m, "auroc2") is not None else None),
        dprime=(round(float(g(md, m, "dprime")), 3) if g(md, m, "dprime") is not None else None),
        meta_d=(round(float(g(md, m, "meta_d")), 3) if g(md, m, "meta_d") is not None else None),
        m_ratio=(round(float(g(bs, m, "m_ratio")), 3) if g(bs, m, "m_ratio") is not None else None),
        m_lo=(round(float(g(bs, m, "m_lo")), 3) if g(bs, m, "m_lo") is not None else None),
        m_hi=(round(float(g(bs, m, "m_hi")), 3) if g(bs, m, "m_hi") is not None else None),
        estimable=bool(g(bs, m, "estimable", False)),
        bayes_mr=(round(float(g(hm, m, "mr_mean")), 3) if g(hm, m, "mr_mean") is not None else None),
        bayes_lo=(round(float(g(hm, m, "hdi_lo")), 3) if g(hm, m, "hdi_lo") is not None else None),
        bayes_hi=(round(float(g(hm, m, "hdi_hi")), 3) if g(hm, m, "hdi_hi") is not None else None),
        evidence=str(g(hm, m, "evidence", "")),
        acc_ordinary=round(float(byt.loc[m, "常规_acc"]), 3),
        acc_discipline=round(float(byt.loc[m, "学科_acc"]), 3),
        acc_trap=round(float(byt.loc[m, "陷阱_acc"]), 3),
        acc_hallucination=round(float(byt.loc[m, "幻觉_acc"]), 3),
        fic_acc=(round(float(g(hl, m, "fic_acc")), 3) if g(hl, m, "fic_acc") is not None else None),
        real_acc=(round(float(g(hl, m, "real_acc")), 3) if g(hl, m, "real_acc") is not None else None),
        fic_conf=(round(float(g(hl, m, "fic_conf")), 3) if g(hl, m, "fic_conf") is not None else None),
        real_conf=(round(float(g(hl, m, "real_conf")), 3) if g(hl, m, "real_conf") is not None else None),
    ))
DATA = json.dumps(rows, ensure_ascii=False)

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOCRATES · 人类 vs LLM 元认知校准对比</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--navy:#16233b;--deep:#0e5a78;--teal:#1c7293;--ink:#1e2a38;--mute:#5e6e7e;
--coral:#d64545;--gold:#e0a458;--bg:#f3f6f9;--card:#fff;--line:#e2e9ef;--human:#d62728;--model:#3b6ea5;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:26px;margin:0 0 2px}.sub{color:var(--mute);margin:0 0 14px;font-size:14px}
.banner{background:#fff6e9;border:1px solid #f0d9b5;border-radius:10px;padding:10px 14px;font-size:13px;color:#7a5a20;margin-bottom:18px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.kpi .v{font-size:26px;font-weight:700;color:var(--deep)}.kpi .l{font-size:12px;color:var(--mute);margin-top:2px}
.kpi .h{color:var(--human)}
.controls{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:16px;
display:flex;flex-wrap:wrap;gap:16px;align-items:center}
.controls label{font-size:12px;color:var(--mute);margin-right:6px}
.chip{display:inline-block;padding:4px 10px;border:1px solid var(--line);border-radius:16px;font-size:12px;cursor:pointer;background:#fff;margin:2px}
.chip.on{background:var(--deep);color:#fff;border-color:var(--deep)}
select{padding:5px 8px;border:1px solid var(--line);border-radius:8px;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:16px}
.panel h3{margin:0 0 10px;font-size:15px}.panel .note{font-size:12px;color:var(--mute);margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{cursor:pointer;color:var(--deep);user-select:none;position:sticky;top:0;background:#fff}
tr.human{background:#fdeaea;font-weight:600}
.na{color:#c0c8d0}.tablewrap{max-height:520px;overflow:auto}
@media(max-width:820px){.grid{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}}
</style></head>
<body><div class="wrap">
<h1>知道自己不知道 · 人类 vs LLM 元认知校准</h1>
<p class="sub">SOCRATES 完整对比分析 · 46 名人类被试(中文) vs 20 个模型配置(英文) · 400 题四类型</p>
<div class="banner"><b>解读前提:</b> 人类答中文卷、模型跑英文卷(无可用中文模型样本)——所有跨物种比较均在此语言口径下;
多数模型准确率 0.97–1.00、错误<30,meta-d′/M-ratio 不可靠估(天花板),仅 5 组可估。</div>

<div class="kpis">
  <div class="kpi"><div class="v h">0.735</div><div class="l">人类准确率(模型多在 0.97–1.00)</div></div>
  <div class="kpi"><div class="v h">1.37</div><div class="l">人类 M-ratio(可测模型仅 0.31–0.69)</div></div>
  <div class="kpi"><div class="v h">0.269</div><div class="l">人类幻觉·虚构题准确率(低于随机)</div></div>
  <div class="kpi"><div class="v">10/20</div><div class="l">错误<10、贝叶斯也只能给出先验(不可信)</div></div>
</div>

<div class="controls">
  <div><label>能力层级</label><span id="tierChips"></span></div>
  <div><label>厂商</label><span id="vendorChips"></span></div>
  <div><label>思考</label><span id="thinkChips"></span></div>
  <div><label><input type="checkbox" id="estOnly"> 仅显示可估 M-ratio 的组</label></div>
</div>

<div class="panel">
  <h3>主图 · 按指标查看各组 <select id="metricSel">
    <option value="acc">准确率 Accuracy</option>
    <option value="ece">校准误差 ECE(含95%CI,越低越好)</option>
    <option value="overconf">过度自信 stated−acc</option>
    <option value="m_ratio">元认知效率 M-ratio · MLE(含95%CI;仅可估组)</option>
    <option value="bayes_mr">元认知效率 M-ratio · HMeta-d 贝叶斯(全组,含95%HDI)</option>
    <option value="auroc2">type-2 AUC(错误过少时不可靠)</option>
    <option value="err">错误 trial 数(天花板)</option>
  </select></h3>
  <canvas id="mainChart" height="120"></canvas>
  <div class="note" id="mainNote"></div>
</div>

<div class="grid">
  <div class="panel"><h3>meta-d′ vs d′(仅可估组)</h3><canvas id="scatterChart" height="200"></canvas>
    <div class="note">对角线 = meta-d′=d′。人类在线上方(高效),模型在下方。</div></div>
  <div class="panel"><h3>幻觉题:虚构 vs 真实 准确率</h3><canvas id="hallChart" height="200"></canvas>
    <div class="note">仅人类在虚构题低于随机(0.5)。</div></div>
</div>

<div class="panel"><h3>全部指标(点表头排序)</h3>
  <div class="tablewrap"><table id="tbl"></table></div>
</div>

<script>
const DATA = __DATA__;
const F = {tier:new Set(), vendor:new Set(), think:new Set(), estOnly:false};
const TIERS=[...new Set(DATA.map(d=>d.tier))];
const VENDORS=[...new Set(DATA.map(d=>d.vendor))];
const THINK=['think','nothink','na'];
function chip(text,set,val,host){const s=document.createElement('span');s.className='chip on';s.textContent=text;
  set.add(val);s.onclick=()=>{if(set.has(val)){set.delete(val);s.classList.remove('on')}else{set.add(val);s.classList.add('on')}render()};
  host.appendChild(s);}
TIERS.forEach(t=>chip(t,F.tier,t,tierChips));
VENDORS.forEach(v=>chip(v,F.vendor,v,vendorChips));
THINK.forEach(t=>chip(t,F.think,t,thinkChips));
estOnly.onchange=()=>{F.estOnly=estOnly.checked;render()};
metricSel.onchange=render;

function filtered(){return DATA.filter(d=>F.tier.has(d.tier)&&F.vendor.has(d.vendor)&&F.think.has(d.condition)
  &&(!F.estOnly||d.estimable));}
const COL=d=>d.model==='human'?'#d62728':'#3b6ea5';
let main,scatter,hall;
const NOTES={acc:'一阶准确率;虚线=随机 0.5。',ece:'ECE 越低越好;误差棒=95% 自助CI;人类最差且CI不与任何模型重叠。',
 overconf:'>0 过度自信,<0 欠自信;人类 +0.086 最高。跨组 acc↔过度自信 ρ=−0.32(p=0.18,不显著)。',
 m_ratio:'M-ratio=meta-d′/d′,已扣一阶能力;虚线=1;MLE 仅错误≥30 的组可靠。人类[1.21,1.54]与所有模型不重叠。',
 bayes_mr:'HMeta-d 贝叶斯,全组都有估计+95%HDI。灰=先验主导(错误<10,后验≈先验~1,不可信);蓝=正则化(10–29);深蓝=数据驱动(≥30)。人类显著高于所有可信模型。',
 auroc2:'正确 trial 信心高于错误的概率;错误过少的近满分模型不可靠(灰显)。',err:'错误 trial 数;<30(红线)无法可靠估 meta-d′。'};
const EVCOL={'data-driven':'#1f77b4','regularized':'#7fb3d5','prior-dominated':'#c9c9c9'};

function render(){
  const rows=filtered();
  const metric=metricSel.value;
  const withCI=(metric==='ece'||metric==='m_ratio'||metric==='bayes_mr');
  const ciLo=(d)=>metric==='ece'?d.ece_lo:metric==='m_ratio'?d.m_lo:metric==='bayes_mr'?d.bayes_lo:null;
  const ciHi=(d)=>metric==='ece'?d.ece_hi:metric==='m_ratio'?d.m_hi:metric==='bayes_mr'?d.bayes_hi:null;
  let rs=rows.slice();
  if(metric==='m_ratio') rs=rs.filter(d=>d.m_ratio!=null);
  rs.sort((a,b)=>(a[metric]??-1)-(b[metric]??-1));
  const labels=rs.map(d=>d.model), vals=rs.map(d=>d[metric]);
  const bg=rs.map(d=>d.model==='human'?'#d62728':
    metric==='bayes_mr'?(EVCOL[d.evidence]||'#3b6ea5'):
    metric==='auroc2'&&d.err<30?'#c9d3dc':COL(d));
  if(main)main.destroy();
  main=new Chart(mainChart,{type:'bar',data:{labels,datasets:[{data:vals,backgroundColor:bg,
    borderColor:'#0002',borderWidth:1}]},
   options:{indexAxis:'y',plugins:{legend:{display:false},tooltip:{callbacks:{afterLabel:(c)=>{
     const d=rs[c.dataIndex];let s='tier '+d.tier+' · err '+d.err;
     if(metric==='bayes_mr')s+=' · '+d.evidence;
     if(withCI){const lo=ciLo(d),hi=ciHi(d);
       if(lo!=null)s+='\\n95%'+(metric==='bayes_mr'?'HDI':'CI')+' ['+lo+', '+hi+']';}return s;}}}},
    scales:{x:{beginAtZero:true}}}});
  // 参考线(用注解式:画一条 dataset)
  mainNote.textContent=NOTES[metric]||'';
  // 散点 meta-d′ vs d'
  const est=DATA.filter(d=>d.estimable&&d.meta_d!=null);
  if(scatter)scatter.destroy();
  const mx=Math.max(...est.map(d=>Math.max(d.dprime,d.meta_d)))*1.1;
  scatter=new Chart(scatterChart,{type:'scatter',data:{datasets:[
    {label:'groups',data:est.map(d=>({x:d.dprime,y:d.meta_d,m:d.model})),
     backgroundColor:est.map(COL),pointRadius:7,pointHoverRadius:9},
    {label:'meta-d′=d′',type:'line',data:[{x:0,y:0},{x:mx,y:mx}],borderColor:'#888',
     borderDash:[6,4],pointRadius:0,fill:false}]},
   options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:(c)=>c.raw.m?(c.raw.m+'  d′='+c.raw.x+' meta-d′='+c.raw.y):''}}},
    scales:{x:{title:{display:true,text:"d′ (first-order)"}},y:{title:{display:true,text:"meta-d′"}}}}});
  // 幻觉题
  const hrows=rows.filter(d=>d.fic_acc!=null);
  if(hall)hall.destroy();
  hall=new Chart(hallChart,{type:'bar',data:{labels:hrows.map(d=>d.model),datasets:[
    {label:'Fictional (should=False)',data:hrows.map(d=>d.fic_acc),backgroundColor:'#cc6677'},
    {label:'Real-obscure (should=True)',data:hrows.map(d=>d.real_acc),backgroundColor:'#44aa99'}]},
   options:{plugins:{legend:{position:'bottom'}},scales:{y:{beginAtZero:true,max:1.05,title:{display:true,text:'accuracy'}},
    x:{ticks:{maxRotation:60,minRotation:30,font:{size:9}}}}}});
  buildTable(rows);
}

let sortKey='acc',sortDir=1;
const COLS=[['model','组'],['tier','层级'],['n','n'],['err','错误'],['acc','准确率'],['overconf','过度自信'],
 ['ece','ECE'],['auroc2','type2AUC'],['dprime','d′'],['meta_d','meta-d′'],['m_ratio','M-ratio(MLE)'],
 ['bayes_mr','M-ratio(贝叶斯)'],['evidence','证据层级'],
 ['acc_hallucination','幻觉acc'],['fic_acc','虚构acc']];
function buildTable(rows){
  rows=rows.slice().sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
    if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*sortDir;});
  let h='<thead><tr>'+COLS.map(c=>'<th data-k="'+c[0]+'">'+c[1]+'</th>').join('')+'</tr></thead><tbody>';
  for(const d of rows){h+='<tr class="'+(d.model==='human'?'human':'')+'">'+COLS.map(c=>{
    let v=d[c[0]];if(v==null)return '<td class="na">—</td>';
    if(typeof v==='number'&&c[0]!=='n'&&c[0]!=='err')v=v.toFixed(c[0].includes('acc')||c[0]==='overconf'||c[0]==='ece'?3:2);
    return '<td>'+v+'</td>';}).join('')+'</tr>';}
  h+='</tbody>';const t=document.getElementById('tbl');t.innerHTML=h;
  t.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    if(k===sortKey)sortDir*=-1;else{sortKey=k;sortDir=1;}buildTable(filtered());});
}
render();
</script>
</div></body></html>"""

html = HTML.replace("__DATA__", DATA)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print("写出:", OUT_HTML, f"({len(html)//1024} KB, {len(rows)} 组)")
