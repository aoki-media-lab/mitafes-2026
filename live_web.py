"""Real-time browser view for live_fused.py: annotated camera stream (MJPEG) + status JSON.

The inference loop only hands over references (frame, detections, track records,
events); resizing, drawing and JPEG encoding run in a separate thread on a reduced
frame, so the counting path is not slowed down. When rendering falls behind, the
oldest pending frames are dropped (display only; counting is unaffected).

Endpoints: /  (page), /stream.mjpg, /snapshot.jpg, /status.json
No authentication: bind to localhost or a private interface (e.g. Tailscale) only. With a token,
every path is served under /<token>/ only (link-only access), anything else returns 404.
"""
import hmac,json,queue,threading,time
from urllib.parse import parse_qs
from pathlib import Path
from collections import Counter,deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import cv2,numpy as np

FPS=25.0
BIN=900            # seconds per time-of-day bin (the page can merge them into hours)
SMOOTH=4           # display-only moving average of head trails: +-4 observations (~0.7 s)
PAGE="""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Counter</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--text:#e8eaed;--muted:#9aa0a6;--ltr:#4fc3f7;--rtl:#ffb74d;--bad:#ef5350;--ok:#66bb6a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Hiragino Sans",sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:12px;padding:12px;max-width:1800px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr;padding:8px 16px}}
.video{background:#000;border-radius:8px;overflow:hidden;position:relative}
.video img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain}
.off{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;color:var(--muted);font-size:20px}
.off[hidden]{display:none}.off small{font-size:13px}
.badge{position:absolute;top:8px;right:8px;padding:2px 8px;border-radius:4px;font-size:12px;background:var(--bad)}
.badge.ok{background:var(--ok);color:#000}
.panel{background:var(--panel);border-radius:8px;padding:12px;margin-bottom:12px}
h2{font-size:13px;color:var(--muted);margin:0 0 8px;font-weight:600}
.counts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;text-align:center}
.counts b{display:block;font-size:34px;font-variant-numeric:tabular-nums}
.counts span{font-size:12px;color:var(--muted)}
.ltr{color:var(--ltr)}.rtl{color:var(--rtl)}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0;font-size:13px}
dt{color:var(--muted)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.sub{margin:0 0 10px;font-size:12px;color:var(--muted)}
.chead{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}.chead h2{margin:0}
.seg{display:flex;border:1px solid #333a46;border-radius:6px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:3px 10px;font-size:12px;background:transparent;color:var(--muted)}
.seg button[aria-pressed=true]{background:#2a2f3a;color:var(--text)}
.crow{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:12px;color:var(--muted)}
select{background:var(--panel);color:var(--text);border:1px solid #333a46;border-radius:6px;padding:3px 6px;font:inherit;font-size:12px}
.chart{position:relative;height:190px}.chart svg{display:block;width:100%;height:100%;overflow:visible}
.chart .grid{stroke:#2a2f3a;stroke-width:1}.chart .axis{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.chart .bar.rtl{fill:var(--rtl)}.chart .bar.ltr{fill:var(--ltr)}.chart .bar.part{fill-opacity:.45}.chart .hit{fill:transparent;cursor:default}
.chart .base{stroke:#5f6672;stroke-width:1}.chart .net{fill:none;stroke:var(--text);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{fill:var(--text);stroke:var(--panel);stroke-width:2}.chart .cross{stroke:#5f6672;stroke-width:1}
.chart.small{height:120px}.leg{display:flex;gap:12px;font-size:12px;color:var(--muted);margin:0 0 4px}.leg[hidden]{display:none}
.note{margin:8px 0 0;font-size:12px;color:var(--muted);line-height:1.5}.note:empty{display:none}
.net b{color:var(--text)}.counts .net{grid-column:1/-1;border-top:1px solid #262a33;padding-top:8px}
.chart .empty{fill:var(--muted);font-size:12px}
.tip{position:fixed;pointer-events:none;background:#000c;border:1px solid #333a46;border-radius:6px;padding:4px 8px;font-size:12px;white-space:nowrap;z-index:5}
details{margin-top:8px;font-size:12px;color:var(--muted)}summary{cursor:pointer}
table{width:100%;border-collapse:collapse;margin-top:6px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:2px 4px;border-bottom:1px solid #262a33}th:first-child,td:first-child{text-align:left}
.legend{font-size:12px;color:var(--muted);line-height:1.7}
.bar{display:flex;gap:10px;align-items:center;margin-top:8px;font-size:13px;color:var(--muted)}
button{background:var(--panel);color:var(--text);border:1px solid #333a46;border-radius:6px;padding:6px 12px;font:inherit;cursor:pointer}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
</style></head><body><main>
<section><div class="video"><img id="v" alt="camera"><div id="off" class="off" hidden>計測時間外<br><small>毎日 6:00〜19:00 に計測します</small></div><div id="st" class="badge">接続中</div></div>
<div class="bar"><button id="mode" type="button"></button><span id="modeinfo"></span></div>
<div class="legend" style="margin-top:6px">
<span class="sw" style="background:#f44336"></span>カウント線
<span class="sw" style="background:#ffa000"></span>判定マージン
<span class="sw" style="background:#43a047"></span>計測範囲(足元)
<span class="sw" style="background:#1e88e5"></span>照合用切り抜き
<span class="sw" style="background:#fff"></span>人物検出
<span class="sw" style="background:#26c6da"></span>傘
・点と線: 追跡中の頭位置と軌跡<br>
「&lt;- #番号」: 今カウントした人（矢印が向き）／「+1 roi」: 画面全体の追跡では見逃し、線のまわりの切り抜きで見つけて追加した人（太線がその人の軌跡、四角がその人。確定まで約7秒遅れて出ます）</div></section>
<aside>
<div class="panel"><h2 id="h">通過数</h2><div class="counts" id="cnt">
<div><b id="nin">-</b><span id="lin">-</span></div>
<div id="outbox"><b id="nout">-</b><span id="lout">-</span></div>
<div id="netbox" class="net"><b id="nnet">-</b><span>構内にいる人数</span></div></div>
<p class="note" id="netnote"></p></div>
<div class="panel"><h2>状態</h2><dl>
<dt>経過</dt><dd id="t">-</dd><dt>受信fps</dt><dd id="fps">-</dd>
<dt>遅延 p50 / p95</dt><dd id="lat">-</dd><dt>破棄フレーム</dt><dd id="drop">-</dd>
<dt>再接続</dt><dd id="rec">-</dd><dt>GPU</dt><dd id="gpu">-</dd>
<dt>表示fps</dt><dd id="vfps">-</dd></dl></div>
<div class="panel"><div class="chead"><h2 id="chtitle">時間帯別の来場者数</h2>
<div class="seg" role="group" aria-label="集計単位"><button type="button" id="b15" aria-pressed="true">15分</button><button type="button" id="b60" aria-pressed="false">1時間</button></div></div>
<div class="crow"><select id="day" aria-label="表示する日"><option value="">今日</option></select><span id="chtotal"></span></div>
<div class="leg" id="leg" hidden></div>
<div id="chart" class="chart" aria-live="off"></div><div id="tip" class="tip" hidden></div>
<div id="netpanel" hidden><h2 style="margin-top:12px">構内にいる人数の推移</h2><div id="netchart" class="chart small" aria-live="off"></div></div>
<details><summary>表で見る</summary><table id="tbl"><thead></thead><tbody></tbody></table></details></div>
</aside></main>
<script>
const $=id=>document.getElementById(id);
const PAGE_VERSION='__PAGE_VERSION__';
const SHOW_NET=false;  // 構内にいる人数（入場−退場）の表示。一旦止めている（2026-11-22）
const dir={ltr:'左→右',rtl:'右→左'};
let labels={};const lab=d=>labels[d]||dir[d];
async function tick(){
  try{
    const s=await (await fetch('status.json',{cache:'no-store'})).json();
    if(s.page&&s.page!==PAGE_VERSION){location.reload();return}  // server was updated: load the new page
    const d=s.day||{},dirs=s.directions||['rtl'],inD=s.in_dir||dirs[0],outD=dirs.find(x=>x!==inD);labels=s.labels||{};
    const fmt=v=>v==null?'-':v.toLocaleString('ja-JP');
    $('nin').textContent=fmt(d[inD]);$('nin').className=inD;$('lin').textContent=lab(inD);
    if(outD){$('nout').textContent=fmt(d[outD]);$('nout').className=outD;$('lout').textContent=lab(outD);$('nnet').textContent=fmt(d.net)}
    $('netnote').textContent=!outD||!SHOW_NET?'':d.net==null?`${lab(outD)}は今日の計測の途中から数えているため、構内にいる人数は出せません。`:
      `${lab(inD)}−${lab(outD)}（${d.since||''}から、この門の分）。計測開始前に入った人や他の門の出入りは含みません。`;
    hist=s.hist||hist;
    if(s.day_code!==curCode){curCode=s.day_code;loadDays()}else if($('day').options[0])$('day').options[0].textContent=curLabel();
    try{draw()}catch(e){console.error(e)}
    const only=!outD;$('outbox').style.display=only?'none':'';$('netbox').style.display=only||!SHOW_NET?'none':'';
    $('cnt').style.gridTemplateColumns=only?'1fr':'1fr 1fr';$('h').textContent=(only&&!labels[inD]?'通過数（'+dir[inD]+'のみ）':'通過数')+(s.day_label?'　'+s.day_label:'');
    const cap=s.capture!==false;if(cap!==capturing){capturing=cap;show()}
    const r=s.report||{};$('t').textContent=s.elapsed!=null?Math.floor(s.elapsed/60)+'分'+Math.floor(s.elapsed%60)+'秒':'-';
    $('fps').textContent=r.received_fps??'-';$('lat').textContent=s.latency!=null?s.latency.toFixed(2)+'s'+(r.latency_p95!=null?' / '+r.latency_p95+'s':''):'-';
    $('drop').textContent=r.dropped_frames??'-';$('rec').textContent=r.reconnects??'-';
    $('gpu').textContent=r.gpu?r.gpu.power_w.toFixed(0)+'W '+r.gpu.temp_c+'°C':'-';$('vfps').textContent=s.render_fps??'-';
    const live=s.age!=null&&s.age<3;$('st').textContent=!cap?'計測時間外':live?'LIVE':'停止/遅延';$('st').className='badge'+(live?' ok':'');
    if(s.stream_kbps!=null&&!lite)$('modeinfo').textContent=`動画（約${(s.stream_kbps/1000).toFixed(1)}Mbps／1人あたり）`;
  }catch(e){$('st').textContent='サーバー応答なし';$('st').className='badge'}
}
// ---- time-of-day chart (single series: one hue, no legend; the in-progress bin is drawn paler)
let hist=null,unit=15;
const pad=n=>String(n).padStart(2,'0'),hm=t=>{const d=new Date(t*1000);return pad(d.getHours())+':'+pad(d.getMinutes())};
// Chart day: '' = the day the server is on (today while counting), else a past day from day.json.
let sel='',past={},curCode=null;
const shown=()=>sel?past[sel]:hist;
async function loadDays(){
  try{const d=await (await fetch('days.json',{cache:'no-store'})).json(),el=$('day'),keep=el.value;
    el.innerHTML=`<option value="">${curLabel()}</option>`+d.days.filter(x=>x.code!==curCode).map(x=>`<option value="${x.code}">${x.label}（${x.total}人）</option>`).join('');
    el.value=[...el.options].some(o=>o.value===keep)?keep:''}catch(e){}
}
const curLabel=()=>hist&&hist.label?hist.label+(hist.live?'（今日）':''):'今日';
$('day').onchange=async()=>{sel=$('day').value;
  if(sel&&!past[sel]){try{past[sel]=await (await fetch('day.json?d='+sel,{cache:'no-store'})).json()}catch(e){sel='';$('day').value=''}}
  drawn='';draw()};
function binsFor(){
  // -> {dirs, rows:[{t, v:[n per dir], net, part}]}; net = running in-out at the end of each bin
  const hist=shown();
  if(!hist||!hist.bins.length)return{dirs:hist?hist.dirs||[]:[],rows:[]};
  const dirs=hist.dirs||['rtl'],size=unit*60,m=new Map();
  for(const[t,...v]of hist.bins){const k=Math.floor(t/size)*size,a=m.get(k)||dirs.map(()=>0);v.forEach((n,i)=>a[i]+=n);m.set(k,a)}
  const now=hist.now,first=Math.min(...m.keys()),last=Math.floor(now/size)*size,out=[];let net=0;
  for(let t=first;t<=last;t+=size){const v=m.get(t)||dirs.map(()=>0);net+=v[0]-(v[1]||0);out.push({t,v,net,part:!!hist.live&&t===last})}
  return{dirs,rows:out.slice(-60)};
}
function niceMax(v){if(v<=4)return 4;const p=10**Math.floor(Math.log10(v)),f=v/p;return (f<=1?1:f<=2?2:f<=5?5:10)*p}
let drawn='';
const fmtn=v=>v.toLocaleString('ja-JP');
function tipText(x,dirs){
  let t=`${hm(x.t)}–${hm(x.t+unit*60)}　`+dirs.map((d,i)=>`${lab(d)} ${fmtn(x.v[i])}人`).join('／');
  if(dirs.length>1&&SHOW_NET)t+=`／構内 ${fmtn(x.net)}人`;return t+(x.part?'（集計中）':'');
}
function hover(el,b,dirs,cross){
  const tip=$('tip');
  el.querySelectorAll('.hit').forEach(r=>{
    r.onmousemove=e=>{const x=b[+r.dataset.i];tip.hidden=false;tip.textContent=tipText(x,dirs);
      tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY-28)+'px';if(cross)cross(+r.dataset.i)};
    r.onmouseleave=()=>{tip.hidden=true;if(cross)cross(-1)};
  });
}
function hourLabels(b,L,bw,H){
  const every=unit===60?Math.max(1,Math.ceil(b.length/6)):Math.max(4,Math.ceil(b.length/6/4)*4);let g='';
  b.forEach((x,i)=>{const d=new Date(x.t*1000);
    if(d.getMinutes()===0&&d.getHours()%Math.max(1,every*unit/60)===0)g+=`<text class="axis" x="${L+i*bw}" y="${H-4}">${d.getHours()}時</text>`});
  return g;
}
const bar=(x0,y0,w,h,up,cls)=>{const r=Math.min(4,w/2,h);  // rounded at the data end, flat on the baseline
  return up?`<path class="${cls}" d="M${x0},${y0}V${y0-h+r}q0,${-r} ${r},${-r}H${x0+w-r}q${r},0 ${r},${r}V${y0}Z"/>`
           :`<path class="${cls}" d="M${x0},${y0}V${y0+h-r}q0,${r} ${r},${r}H${x0+w-r}q${r},0 ${r},${-r}V${y0}Z"/>`};
function draw(){
  const {dirs,rows:b}=binsFor(),key=sel+unit+JSON.stringify(b)+JSON.stringify(labels);if(key===drawn)return;drawn=key;
  const h=shown(),two=dirs.length>1;
  $('chtotal').textContent=h&&b.length?dirs.map((d,i)=>`${two?lab(d)+' ':'合計 '}${fmtn(h.bins.reduce((a,x)=>a+x[i+1],0))}人`).join('　'):'';
  $('leg').hidden=!two;$('leg').innerHTML=two?dirs.map((d,i)=>`<span><span class="sw" style="background:var(--${d})"></span>${lab(d)}（${i?'下':'上'}向き）</span>`).join(''):'';
  $('chtitle').textContent=two?'時間帯別の出入り':'時間帯別の来場者数';
  const el=$('chart'),W=el.clientWidth||300,H=190,L=34,B=18,T=8;
  $('netpanel').hidden=!SHOW_NET||!two||!b.length;
  if(!b.length){el.innerHTML=`<svg viewBox="0 0 ${W} ${H}"><text class="empty" x="${W/2}" y="${H/2}" text-anchor="middle">まだ通過がありません</text></svg>`;$('tbl').tHead.innerHTML='';$('tbl').tBodies[0].innerHTML='';return}
  const bw=(W-L)/b.length,gap=Math.min(2,bw*.25),w=Math.max(1,bw-gap);let g='';
  if(!two){
    const max=niceMax(Math.max(...b.map(x=>x.v[0]))),y=v=>T+(H-T-B)*(1-v/max);
    for(const v of[0,max/2,max])g+=`<line class="grid" x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text class="axis" x="${L-4}" y="${y(v)+3}" text-anchor="end">${v}</text>`;
    b.forEach((x,i)=>{if(x.v[0]>0)g+=bar(L+i*bw+gap/2,y(0),w,y(0)-y(x.v[0]),true,`bar ${dirs[0]}${x.part?' part':''}`)});
  }else{
    // Mirrored bars on one axis: in (first direction) above the baseline, out below.
    const max=niceMax(Math.max(...b.map(x=>Math.max(x.v[0],x.v[1])))),mid=T+(H-T-B)/2,k=(H-T-B)/2/max;
    for(const v of[-max,-max/2,max/2,max])g+=`<line class="grid" x1="${L}" x2="${W}" y1="${mid-v*k}" y2="${mid-v*k}"/><text class="axis" x="${L-4}" y="${mid-v*k+3}" text-anchor="end">${Math.abs(v)}</text>`;
    b.forEach((x,i)=>{const x0=L+i*bw+gap/2,p=x.part?' part':'';
      if(x.v[0]>0)g+=bar(x0,mid-1,w,x.v[0]*k,true,`bar ${dirs[0]}${p}`);
      if(x.v[1]>0)g+=bar(x0,mid+1,w,x.v[1]*k,false,`bar ${dirs[1]}${p}`)});
    g+=`<line class="base" x1="${L}" x2="${W}" y1="${mid}" y2="${mid}"/>`;
  }
  g+=hourLabels(b,L,bw,H);
  b.forEach((x,i)=>{g+=`<rect class="hit" data-i="${i}" x="${L+i*bw}" y="${T}" width="${bw}" height="${H-T-B}"/>`});
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${$('chtitle').textContent}">${g}</svg>`;
  hover(el,b,dirs);
  if(two&&SHOW_NET)drawNet(b,dirs);
  $('tbl').tHead.innerHTML='<tr><th>時間帯</th>'+dirs.map(d=>`<th>${lab(d)}</th>`).join('')+(two&&SHOW_NET?'<th>構内</th>':'')+'</tr>';
  $('tbl').tBodies[0].innerHTML=b.slice().reverse().map(x=>`<tr><td>${hm(x.t)}–${hm(x.t+unit*60)}${x.part?'（集計中）':''}</td>`+
    x.v.map(n=>`<td>${fmtn(n)}</td>`).join('')+(two&&SHOW_NET?`<td>${fmtn(x.net)}</td>`:'')+'</tr>').join('');
}
function drawNet(b,dirs){
  // Running in-out at the end of each bin: a 2px line with a crosshair + dot on hover.
  const el=$('netchart'),W=el.clientWidth||300,H=120,L=34,B=18,T=8,bw=(W-L)/b.length;
  const lo=Math.min(0,...b.map(x=>x.net)),hi=niceMax(Math.max(1,...b.map(x=>x.net))),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const xs=i=>L+(i+1)*bw;let g='';
  for(const v of(lo<-hi*.15?[lo,0,hi]:[0,hi/2,hi]))g+=`<line class="grid" x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text class="axis" x="${L-4}" y="${y(v)+3}" text-anchor="end">${fmtn(Math.round(v))}</text>`;
  g+=`<path class="net" d="M${L},${y(0)}`+b.map((x,i)=>`L${xs(i)},${y(x.net)}`).join('')+'"/>';
  g+=hourLabels(b,L,bw,H)+'<line class="cross" id="nx" y1="'+T+'" y2="'+(H-B)+'" visibility="hidden"/><circle class="dot" id="nd" r="4" visibility="hidden"/>';
  b.forEach((x,i)=>{g+=`<rect class="hit" data-i="${i}" x="${L+i*bw}" y="${T}" width="${bw}" height="${H-T-B}"/>`});
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="構内にいる人数の推移">${g}</svg>`;
  hover(el,b,dirs,i=>{const nx=el.querySelector('#nx'),nd=el.querySelector('#nd'),v=i<0?'hidden':'visible';
    nx.setAttribute('visibility',v);nd.setAttribute('visibility',v);if(i<0)return;
    nx.setAttribute('x1',xs(i));nx.setAttribute('x2',xs(i));nd.setAttribute('cx',xs(i));nd.setAttribute('cy',y(b[i].net))});
}
for(const[id,u]of[['b15',15],['b60',60]])$(id).onclick=()=>{unit=u;$('b15').setAttribute('aria-pressed',u===15);$('b60').setAttribute('aria-pressed',u===60);
  try{localStorage.setItem('unit',u)}catch(e){};draw()};
try{const u=+localStorage.getItem('unit');if(u===60)$('b60').click()}catch(e){}
addEventListener('resize',()=>{drawn='';draw()});
setInterval(loadDays,300000);
setInterval(tick,1000);tick();
// Video (MJPEG, capped server-side) or low-data mode (one still image every 2 s).
// Nothing is downloaded while the tab is hidden.
let lite=new URLSearchParams(location.search).has('lite'),timer=null,capturing=true;
try{const v=localStorage.getItem('lite');if(v!==null&&!location.search)lite=v==='1'}catch(e){}
function still(){$('v').src='snapshot.jpg?'+Date.now()}
function show(){
  clearInterval(timer);timer=null;$('off').hidden=capturing;$('mode').hidden=!capturing;$('modeinfo').hidden=!capturing;
  if(document.hidden||!capturing){$('v').removeAttribute('src');return}
  if(lite){still();timer=setInterval(still,2000)}else{$('v').src='stream.mjpg?'+Date.now()}
  $('mode').textContent=lite?'動画にする':'省データにする';
  $('modeinfo').textContent=lite?'省データ: 2秒ごとの静止画（約0.5Mbps）':'動画';
}
$('mode').onclick=()=>{lite=!lite;try{localStorage.setItem('lite',lite?'1':'0')}catch(e){};show()};
document.addEventListener('visibilitychange',show);
$('v').onerror=()=>{if(!lite)setTimeout(show,2000)};
show();
</script></body></html>"""


def geometry(profile,width,height):
    """Count line, margins and foot ROI as pixel polylines: [(points Nx2, 'line'|'margin'|'roi')]."""
    out=[]
    for off,kind in [(-profile['margin'],'margin'),(profile['margin'],'margin'),(0,'line')]:
        x=(profile['line_x']+off)*width;out.append((np.array([[x,0],[x,height]],np.float64),kind))
    if profile['roi_mode']=='glass_ground':
        xs=np.linspace(0,width,60)
        out+=[(np.stack([xs,np.full_like(xs,.1*height)],1),'roi'),(np.stack([xs,.82*xs-.71*height],1),'roi')]
    elif profile.get('roi_polygon'):
        out.append((np.array([(a*width,b*height) for a,b in profile['roi_polygon']],np.float64),'roi'))
    return out


def run_directions(folder):
    """Directions a run folder counted: setup.json's profile (live runs), else window.json 'directions'
    (recount folders), else the code default (both)."""
    folder=Path(folder)
    for name,get in [('setup.json',lambda d:d.get('profile',{}).get('count_directions')),('window.json',lambda d:d.get('directions'))]:
        try:v=get(json.loads((folder/name).read_text()))
        except (OSError,ValueError):continue
        if v:return set(v)
    return {'ltr','rtl'}


def day_history(runs,directions,date=None,exclude=None,covered=None):
    """(local time, direction) of the counts in run folders <runs>/<YYYYmmdd>_HHMMSS/live_events.jsonl of
    one day (default today), so the browser's daily chart and totals survive restarts and off hours.
    Lines without 'clock' (older runs) are placed at folder time + ~6 s startup + stream seconds.
    A recount folder <runs>/<YYYYmmdd>_recount_*/ (window.json {"start","end"} in epoch seconds, plus
    live_events.jsonl with 'clock') replaces every live count in [start, end) by its own counts.
    covered (a set, optional) is narrowed to the directions that every folder of the day counted
    (runs from before both directions were kept have no left-to-right events)."""
    def read(f,start=None):
        out=[]
        with f.open() as lines:
            for line in lines:
                try:e=json.loads(line)
                except ValueError:continue
                if e.get('direction') in directions and ('clock' in e or start is not None):
                    out.append((e['clock'] if 'clock' in e else start+6+float(e.get('seconds',0)),e['direction']))
        return out
    times=[];windows=[];recounted=[];spans=[]
    for f in sorted(Path(runs).glob((date or time.strftime('%Y%m%d'))+'_*/live_events.jsonl')):
        if exclude is not None and f.parent.resolve()==Path(exclude).resolve():continue
        if (f.parent/'window.json').exists():
            w=json.loads((f.parent/'window.json').read_text());windows.append((w['start'],w['end'],run_directions(f.parent)))
            recounted+=[e for e in read(f) if w['start']<=e[0]<w['end']]
        else:
            try:start=time.mktime(time.strptime(f.parent.name,'%Y%m%d_%H%M%S'))
            except ValueError:continue
            got=read(f,start);times+=got
            spans.append((start,max([start]+[t for t,_ in got]),run_directions(f.parent)))
    if covered is not None:
        for a,b,dirs in windows:covered&=dirs
        # A live run lying entirely inside a recount window counts with the window's directions.
        for a,b,dirs in spans:covered&=dirs.union(*[d for s,e,d in windows if s<=a and b<e])
    return [e for e in times if not any(a<=e[0]<b for a,b,_ in windows)]+recounted


def smooth(pts,k=SMOOTH):
    """Centered moving average (window 2k+1, shortened at the ends); display only."""
    n=len(pts)
    if n<3:return pts
    c=np.vstack([np.zeros((1,2)),np.cumsum(pts,0)]);i=np.arange(n)
    lo=np.maximum(0,i-k);hi=np.minimum(n,i+k+1)
    return (c[hi]-c[lo])/(hi-lo)[:,None]


def place_label(img,text,at,color,placed,scale=.75):
    """Text on a dark box next to point 'at', moved to the first nearby spot that does not overlap an
    earlier label (placed: list of rectangles, updated), so labels of people walking together stay readable."""
    (tw,th),_=cv2.getTextSize(text,0,scale,2);x0,y0=at;H,W=img.shape[:2]
    for dx,dy in [(14,-14),(14,th+18),(14,-th-36),(14,2*th+40),(-tw-24,-14),(-tw-24,th+18),(14,-2*th-58),(14,3*th+62)]:
        x=int(min(max(x0+dx,0),W-tw-8));y=int(min(max(y0+dy,th+6),H-6))
        r=(x-4,y-th-6,x+tw+4,y+6)
        if not any(r[0]<q[2] and q[0]<r[2] and r[1]<q[3] and q[1]<r[3] for q in placed):break
    placed.append(r)
    cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]),(25,25,25),-1);cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]),color,1)
    cv2.putText(img,text,(x,y),0,scale,color,2,cv2.LINE_AA)
    cv2.line(img,(int(x0),int(y0)),(r[0] if r[0]>x0 else r[2],(r[1]+r[3])//2),color,1,cv2.LINE_AA)


def color_for(tid):
    h=(int(tid)*47)%180
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h,200,255]]]),cv2.COLOR_HSV2BGR)[0,0])


import hashlib
PAGE_VERSION=hashlib.sha1(PAGE.encode()).hexdigest()[:10]
PAGE=PAGE.replace('__PAGE_VERSION__',PAGE_VERSION)


class LiveWeb:
    def __init__(self,host,port,width,height,profile,roi_crop,out_width=1280,quality=75,max_fps=0,history=(),max_mbps=0,token=None,soften=0,capture=True,day_label=None,hist_end=None,runs=None,day_code=None,day_end='19:00',
                 record_dir=None,record_segment=3600,covered=None):
        """history: (time, direction) of today's earlier counts; covered: directions every earlier run of
        today counted (None: no earlier runs, i.e. all configured directions are counted all day)."""
        # Counted directions, the "in" direction first; labels come from the setup (e.g. 入場/退場).
        dirs=list(profile.get('count_directions',['ltr','rtl']))
        self.in_dir=profile.get('in_direction',dirs[0]) if profile.get('in_direction') in dirs else dirs[0]
        self.directions=[self.in_dir]+[d for d in dirs if d!=self.in_dir]
        self.labels=dict(profile.get('direction_labels') or {})
        self.W=width;self.H=height;self.s=out_width/width;self.size=(out_width,round(height*self.s))
        self.quality=quality;self.min_gap=1/max_fps if max_fps else 0;self.capture=capture;self.day_label=day_label;self.hist_end=hist_end;
        self.runs=Path(runs) if runs else None;self.day_code=day_code or time.strftime('%Y%m%d');self.day_end=day_end;self.past={}
        self.soften=soften;self.max_bps=max_mbps*1e6;self.last_render=0.;self.to_live=lambda p:np.asarray(p,np.float64).reshape(-1,2)
        style={'line':((54,67,244),3),'margin':((0,160,255),1),'roi':((71,160,67),2)}
        self.lines=[((pts*self.s).astype(np.int32),*style[kind],kind=='roi') for pts,kind in geometry(profile,width,height)]
        x1,y1,x2,y2=roi_crop;self.crop=((int(x1*self.s),int(y1*self.s)),(int(x2*self.s)-1,int(y2*self.s)-1))
        self.frames=queue.Queue(maxsize=40);self.cond=threading.Condition();self.jpeg=None;self.seq=0
        self.trails={};self.flash={};self.counted={};self.marks=[];self.counts={};self.report={}
        # roi-branch tracks (head positions of the last ~12 s) so a count added by the roi branch can be shown
        # on the person it belongs to: its trajectory across the line and its current box, labelled '+1 roi'.
        self.roi_trails={};self.roi_box={};self.roi_marks=[]
        # Small list of the last counts (time, direction, ID) in the bottom-right corner; clock_at(index) gives
        # the local time of an observation (live: now; offline renders set it from the video's start time).
        self.recent=deque(maxlen=10);self.recent_img=None;self.clock_at=lambda index:time.time()
        self.line_x=profile['line_x']*width
        # Counts per 15-minute bin of local time for the daily chart; seeded with today's
        # earlier runs (after a restart) so the chart and day total continue.
        self.bins={};self.history=Counter();self.run_start=time.time()
        for t,d in history:self.bins.setdefault(int(t//BIN)*BIN,Counter())[d]+=1;self.history[d]+=1
        self.covered=[d for d in self.directions if covered is None or d in covered]
        self.sizes=deque(maxlen=30)
        self.latency=None;self.t0=None;self.elapsed=None;self.last_frame_wall=None
        self.rendered=deque(maxlen=50);self.stop=threading.Event()
        self.recorder=Recorder(record_dir,self.size,record_segment) if record_dir else None
        self.renderer=threading.Thread(target=self.render_loop,daemon=True);self.renderer.start()
        web=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def send(self,body,ctype):
                self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def do_GET(self):
                path=self.path.split('?')[0]
                if token:
                    # Link-only access: everything lives under /<token>/; anything else is a plain 404.
                    head,_,rest=path[1:].partition('/')
                    if not hmac.compare_digest(head.encode(),token.encode()):self.send_error(404);return
                    if path=='/'+token:
                        self.send_response(301);self.send_header('Location','/'+token+'/');self.end_headers();return
                    path='/'+rest
                if path in('/','/index.html'):self.send(PAGE.encode(),'text/html; charset=utf-8')
                elif path=='/status.json':self.send(json.dumps(web.status()).encode(),'application/json')
                elif path=='/days.json':self.send(json.dumps(dict(days=web.days())).encode(),'application/json')
                elif path=='/day.json':
                    d=parse_qs(self.path.partition('?')[2]).get('d',[''])[0]
                    h=web.day(d)
                    if h is None:self.send_error(404);return
                    self.send(json.dumps(h).encode(),'application/json')
                elif path=='/snapshot.jpg':
                    if web.jpeg is None:self.send_error(503);return
                    self.send(web.jpeg,'image/jpeg')
                elif path=='/stream.mjpg':
                    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control','no-store');self.end_headers();last=-1
                    try:
                        while not web.stop.is_set():
                            with web.cond:
                                web.cond.wait_for(lambda:web.seq!=last or web.stop.is_set(),timeout=5)
                                jpg,last=web.jpeg,web.seq
                            if jpg is None:continue
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n'%len(jpg)+jpg+b'\r\n')
                    except (BrokenPipeError,ConnectionResetError):pass
                else:self.send_error(404)
        self.server=ThreadingHTTPServer((host,port),Handler);self.server.daemon_threads=True
        threading.Thread(target=self.server.serve_forever,daemon=True).start()

    def submit(self,index,image,people,umbrellas,records,events,counts,latency,elapsed):
        """Called once per processed observation from the inference loop. Never blocks."""
        self.counts=counts;self.latency=latency;self.elapsed=elapsed;self.last_frame_wall=time.time()
        now=time.time()
        for e in events:
            if e['direction'] in self.directions:self.bins.setdefault(int(now//BIN)*BIN,Counter())[e['direction']]+=1
        item=(index,image,people,umbrellas,records,[(e.get('track_id'),e['direction'],e.get('source'),e.get('y'),e.get('frame')) for e in events])
        try:self.frames.put_nowait(item)
        except queue.Full:
            try:self.frames.get_nowait()
            except queue.Empty:pass
            self.frames.put_nowait(item)

    def set_report(self,row):self.report=row

    def status(self):
        now=time.time();r=list(self.rendered)
        fps=round((len(r)-1)/(r[-1]-r[0]),1) if len(r)>1 and r[-1]>r[0] else None
        day=self.day_counts()
        first=min(self.bins) if self.bins else self.run_start
        day['since']=time.strftime('%H:%M',time.localtime(min(first,self.run_start)))
        kbps=round(sum(self.sizes)/len(self.sizes)*8*fps/1000) if self.sizes and fps else None
        return dict(directions=self.directions,in_dir=self.in_dir,labels=self.labels,counts=self.counts,day=day,report=self.report,
                    latency=self.latency,elapsed=self.elapsed,
                    age=now-self.last_frame_wall if self.last_frame_wall else None,render_fps=fps,stream_kbps=kbps,
                    run_start=time.strftime('%H:%M',time.localtime(self.run_start)),
                    hist=self.hist_json(self.bins,self.covered,min(now,self.hist_end) if self.hist_end else now,self.day_label,self.capture),
                    day_code=self.day_code,capture=self.capture,day_label=self.day_label,page=PAGE_VERSION)

    def day_counts(self):
        """Today's counts per direction (earlier runs + this run) and, when both directions were counted
        all day, the difference in - out ('net': people who came in through this gate and are still in)."""
        day={d:self.counts.get(d,0)+self.history[d] for d in('ltr','rtl')}
        day['events']=day['ltr']+day['rtl']
        out=[d for d in self.directions if d!=self.in_dir]
        day['net']=day[self.in_dir]-day[out[0]] if out and out[0] in self.covered else None
        day['covered']=self.covered
        return day

    def hist_json(self,bins,covered,now,label,live):
        """Chart data: bins [[t, n per direction in 'dirs' order], ...]; 'covered' = directions counted all day."""
        dirs=[d for d in self.directions if d in covered] or self.directions[:1]
        return dict(bin=BIN,now=now,dirs=dirs,bins=[[t]+[c.get(d,0) for d in dirs] for t,c in sorted(bins.items())],label=label,live=live)

    def days(self):
        """Counting days found in the run folders, newest first, with totals."""
        if not self.runs:return []
        codes=sorted({d.name[:8] for d in self.runs.glob('20??????_??????') if (d/'live_events.jsonl').exists()},reverse=True)
        return [dict(code=c,label=f'{int(c[4:6])}/{int(c[6:8])}',total=sum(b[1] for b in self.day(c)['bins'])) for c in codes]

    def day(self,code):
        """15-minute counts of one past day (cached; the current day is rebuilt every call)."""
        if not self.runs or len(code)!=8 or not code.isdigit():return None
        if code in self.past and code!=time.strftime('%Y%m%d'):return self.past[code]
        covered=set(self.directions);bins={}
        for t,d in day_history(self.runs,set(self.directions),date=code,covered=covered):bins.setdefault(int(t//BIN)*BIN,Counter())[d]+=1
        end=time.mktime(time.strptime(code+self.day_end,'%Y%m%d%H:%M'))
        h=self.hist_json(bins,covered,min(time.time(),end),f'{int(code[4:6])}/{int(code[6:8])}',False)
        self.past[code]=h;return h

    def render_loop(self):
        base=None
        while not self.stop.is_set():
            try:index,image,people,umbrellas,records,events=self.frames.get(timeout=1)
            except queue.Empty:continue
            # Pace output at the camera's frame spacing; resync if more than 1 s off.
            target=None if base is None else base[1]+(index-base[0])/FPS
            now=time.perf_counter()
            if target is None or abs(target-now)>1:base=(index,now)
            elif target>now:time.sleep(target-now)
            # Track trails and count flashes on every observation; encode only at the
            # display rate cap (bandwidth for slow networks).
            self.update(index,records,events)
            # Bitrate cap: stretch the frame spacing when recent frames are large (busy scenes).
            gap=max(self.min_gap,sum(self.sizes)/len(self.sizes)*8/self.max_bps if self.max_bps and self.sizes else 0)
            # Keep a schedule (due += gap) instead of "gap since the last frame": observations come
            # every 80 ms, so the latter rounds any cap between 6.25 and 12.5 fps down to 6.25.
            now=time.perf_counter()
            web_due=not gap or now>=self.last_render
            if not web_due and self.recorder is None:continue
            img=self.render(index,image,people,umbrellas,records)
            if web_due:
                self.last_render=max(self.last_render,now-gap)+gap if gap else now
                jpg=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,self.quality,cv2.IMWRITE_JPEG_OPTIMIZE,1])[1].tobytes()
                with self.cond:self.jpeg=jpg;self.seq+=1;self.cond.notify_all()
                self.sizes.append(len(jpg))
                self.rendered.append(time.time())
            if self.recorder is not None:
                # Recording gets every observation (12.5 fps) plus the clock and the day's count.
                W,H=self.size;y=10
                for t,c in self.total_lines():
                    (tw,th),_=cv2.getTextSize(t,0,.9,2)
                    cv2.rectangle(img,(W-tw-28,y),(W-10,y+th+20),(20,20,20),-1);cv2.putText(img,t,(W-tw-19,y+th+10),0,.9,c,2,cv2.LINE_AA);y+=th+24
                cv2.rectangle(img,(10,H-48),(170,H-10),(20,20,20),-1)
                cv2.putText(img,time.strftime('%H:%M:%S'),(20,H-20),0,.9,(255,255,255),2,cv2.LINE_AA)
                self.recorder.write(index,img)

    def total_lines(self):
        """Corner text of the recording (ASCII only: OpenCV has no Japanese glyphs)."""
        day=self.day_counts();color={'ltr':(247,195,79),'rtl':(77,183,255)}
        if len(self.directions)==1:return [(f"{'left->right' if self.in_dir=='ltr' else 'right->left'}  {day[self.in_dir]}",color[self.in_dir])]
        out=self.directions[1]
        return [(f'IN  {day[self.in_dir]}',color[self.in_dir]),(f'OUT  {day[out]}',color[out])]

    def update(self,index,records,events):
        full,roi=records;alive={}
        for person in full['people']:
            if 'anchor' not in person:continue
            tid=person['id'];alive[tid]=person['anchor'];self.trails.setdefault(tid,deque(maxlen=50)).append(person['anchor'])
        for person in roi['people']:
            if 'anchor' not in person:continue
            self.roi_trails.setdefault(person['id'],deque(maxlen=150)).append((index,person['anchor']))
            self.roi_box[person['id']]=(index,person['xyxy'])
        for tid,direction,source,y,*rest in events:
            # 'Recent counts' list: local time of the crossing (a roi addition is decided ~7 s after it).
            frame=int(rest[0]) if rest and rest[0] is not None else index
            self.recent.appendleft((self.clock_at(index)-(index-frame)/FPS,direction,tid,source=='roi_added'));self.recent_img=None
            if source!='roi_added':
                self.counted[tid]=direction;self.flash[tid]=(direction,index+38);continue
            # A passage the full view missed, added from the roi branch (live: decided ~7 s after the crossing).
            # Shown on that roi track; a circle on the line only if its trajectory is no longer known.
            if self.roi_trails.get(tid):self.roi_marks.append((tid,direction,index+38,frame))
            else:self.marks.append(((self.line_x,float(y or self.H/2)),direction,index+38))
        self.flash={k:v for k,v in self.flash.items() if v[1]>=index}
        self.marks=[m for m in self.marks if m[2]>=index]
        self.roi_marks=[m for m in self.roi_marks if m[2]>=index]
        if index%100==0:
            keep=set(alive)|set(self.flash)
            self.trails={k:v for k,v in self.trails.items() if k in keep}
            self.counted={k:v for k,v in self.counted.items() if k in keep}
            shown={m[0] for m in self.roi_marks}
            self.roi_trails={k:v for k,v in self.roi_trails.items() if v[-1][0]>=index-150 or k in shown}
            self.roi_box={k:v for k,v in self.roi_box.items() if k in self.roi_trails}

    def draw(self,index,image,people,umbrellas,records):
        img=self.render(index,image,people,umbrellas,records)
        return cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,self.quality,cv2.IMWRITE_JPEG_OPTIMIZE,1])[1].tobytes()

    def render(self,index,image,people,umbrellas,records):
        """Annotated picture at the output size (also used to render review videos)."""
        s=self.s;img=cv2.resize(image,self.size,interpolation=cv2.INTER_AREA)
        # Smooth the camera picture slightly before drawing: the floor's fine grain is what makes
        # JPEG frames large (~36% smaller at 0.7); the overlays drawn afterwards stay sharp.
        if self.soften:img=cv2.GaussianBlur(img,(0,0),self.soften)
        for pts,c,t,closed in self.lines:cv2.polylines(img,[pts],closed,c,t,cv2.LINE_AA)
        cv2.rectangle(img,*self.crop,(229,136,30),1)
        for b in people:cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),(255,255,255),1)
        for b in umbrellas:cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),(218,198,38),2)
        full,roi=records;alive={p['id'] for p in full['people'] if 'anchor' in p}
        # Same line width everywhere; strength is shown by opacity: older parts of a trail
        # fade out, ordinary tracks stay pale, counted tracks keep the count colour, and a
        # just-counted track (also one that vanished at the line) is drawn fully opaque.
        layers={}                                   # alpha -> list of (kind, args)
        labels=[]
        for tid,trail in self.trails.items():
            hit=self.flash.get(tid);done=self.counted.get(tid)
            if tid not in alive and not hit:continue
            pts=smooth(self.to_live(list(trail)))*s
            c=((247,195,79) if (hit or (done,))[0]=='ltr' else (77,183,255)) if hit or done else color_for(tid)
            top=1. if hit else (.8 if done else .55);n=len(pts)
            for j in range(1,n):
                a=round(top*(.15+.85*j/(n-1))*5)/5 or .2
                layers.setdefault(a,[]).append(('line',(tuple(map(int,pts[j-1])),tuple(map(int,pts[j]))),c))
            x,y=map(int,pts[-1]);layers.setdefault(top,[]).append(('dot',(x,y),c))
            labels.append(((x+8,y-6),f"{'->' if hit[0]=='ltr' else '<-'} #{tid}" if hit else (f"{tid} ok" if done else str(tid)),c,hit))
        for a in sorted(layers):
            ink=np.zeros_like(img);mask=np.zeros(img.shape[:2],np.uint8)
            for kind,g,c in layers[a]:
                if kind=='line':cv2.line(ink,*g,c,3,cv2.LINE_AA);cv2.line(mask,*g,255,3,cv2.LINE_AA)
                else:cv2.circle(ink,g,5,c,-1,cv2.LINE_AA);cv2.circle(mask,g,5,255,-1,cv2.LINE_AA)
            # Blend only inside the drawn area (outside it the result equals the picture anyway):
            # full-frame float blending per opacity level was most of the drawing time.
            x,y,w,h=cv2.boundingRect(mask)
            if not w:continue
            m=mask[y:y+h,x:x+w].astype(np.float32)*(a/255)
            img[y:y+h,x:x+w]=cv2.blendLinear(img[y:y+h,x:x+w],ink[y:y+h,x:x+w],1-m,m)
        placed=[]
        for tid,direction,_,frame in self.roi_marks:
            c=(247,195,79) if direction=='ltr' else (77,183,255)
            # Only the passage itself (+-2 s around the crossing), not everything the track did later.
            near=[q for f,q in self.roi_trails.get(tid,()) if abs(f-frame)<=50]
            pts=(smooth(self.to_live(near))*s).astype(np.int32) if near else np.zeros((0,2),np.int32)
            if len(pts)>1:
                cv2.polylines(img,[pts],False,(0,0,0),8,cv2.LINE_AA);cv2.polylines(img,[pts],False,c,4,cv2.LINE_AA)
            # Where the trajectory meets the count line: '+1 roi #id' with the direction arrow.
            k=int(np.argmin(np.abs(pts[:,0]-self.line_x*s))) if len(pts) else 0
            x,y=map(int,pts[k]) if len(pts) else (int(self.line_x*s),0)
            cv2.circle(img,(x,y),12,(0,0,0),6,cv2.LINE_AA);cv2.circle(img,(x,y),12,c,3,cv2.LINE_AA)
            place_label(img,f"+1 roi {'->' if direction=='ltr' else '<-'} #{tid}",(x,y),c,placed)
            seen,b=self.roi_box.get(tid,(None,None))
            if seen is not None and index-seen<=4:   # the person, while the roi branch still tracks them
                cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),(0,0,0),6)
                cv2.rectangle(img,(int(b[0]*s),int(b[1]*s)),(int(b[2]*s),int(b[3]*s)),c,3)
        if self.recent:self.draw_recent(img)
        for (x,y),direction,_ in self.marks:
            x,y=int(x*s),int(y*s);c=(247,195,79) if direction=='ltr' else (77,183,255)
            cv2.circle(img,(x,y),14,c,3,cv2.LINE_AA)
            cv2.putText(img,'+1',(x+18,y+6),0,.8,(0,0,0),3,cv2.LINE_AA);cv2.putText(img,'+1',(x+18,y+6),0,.8,c,2,cv2.LINE_AA)
        for org,text,c,hit in labels:
            if hit:place_label(img,text,(org[0]-8,org[1]+6),c,placed);continue   # just counted: boxed, never overlapping
            cv2.putText(img,text,org,0,.5,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img,text,org,0,.5,(200,200,200),1,cv2.LINE_AA)
        for person in roi['people']:
            if 'anchor' in person:
                x,y=self.to_live([person['anchor']])[0]*s;cv2.circle(img,(int(x),int(y)),3,(255,0,255),-1)
        return img

    def draw_recent(self,img):
        """'直近のカウント' list, newest first, outlined text (cached until the list changes), bottom right."""
        if self.recent_img is None:
            from PIL import Image,ImageDraw,ImageFont
            size=max(14,round(self.size[0]/1280*19))
            try:font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',size,index=0)
            except OSError:font=ImageFont.load_default()
            rows=[('直近のカウント',(230,230,230))]+[(time.strftime('%H:%M:%S',time.localtime(t))+(' 右→左' if d=='rtl' else ' 左→右')+f' #{tid}'+(' roi' if roi else ''),
                                                    (255,183,77) if d=='rtl' else (79,195,247)) for t,d,tid,roi in sorted(self.recent,key=lambda r:-r[0])[:6]]
            w=max(font.getbbox(t)[2] for t,_ in rows)+12;h=len(rows)*(size+6)+10
            canvas=Image.new('RGBA',(w,h),(0,0,0,0));draw=ImageDraw.Draw(canvas)
            for i,(t,c) in enumerate(rows):draw.text((6,4+i*(size+6)),t,font=font,fill=c+(255,),stroke_width=3,stroke_fill=(0,0,0,255))
            a=np.asarray(canvas);self.recent_img=(a[...,2::-1].astype(np.float32),a[...,3:].astype(np.float32)/255)
        rgb,alpha=self.recent_img;h,w=alpha.shape[:2];H,W=img.shape[:2]
        y0,x0=H-h-12,W-w-12
        if y0<0 or x0<0:return
        patch=img[y0:y0+h,x0:x0+w].astype(np.float32)
        img[y0:y0+h,x0:x0+w]=(patch*(1-alpha)+rgb*alpha).astype(np.uint8)

    def close(self):
        self.stop.set()
        with self.cond:self.cond.notify_all()
        self.server.shutdown();self.server.server_close()
        self.renderer.join(timeout=5)
        if self.recorder is not None:self.recorder.close()


class Recorder:
    """Writes the annotated live view to H.264 MP4 files (12.5 fps, one file per clock hour by default).

    Frames go to ffmpeg (NVENC) from a separate thread, so a slow encoder never blocks drawing; when its
    queue is full a frame is dropped and counted. Observations the renderer skipped are filled by repeating
    the previous picture (up to 10 s), so the video keeps real time. Files are fragmented MP4: a file cut off
    by a crash still plays up to the last fragment. Any failure only stops the recording, never the counter."""
    FPS=12.5

    def __init__(self,folder,size,segment=3600,encoder='h264_nvenc'):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.size=size
        self.q=queue.Queue(maxsize=60);self.last=None;self.prev=None;self.stats=dict(written=0,repeated=0,dropped=0,error=None)
        w,h=size
        codec=['-c:v',encoder,'-preset','p4','-cq','27'] if encoder=='h264_nvenc' else ['-c:v','libx264','-preset','veryfast','-crf','26']
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(self.FPS),'-i','pipe:0',
             *codec,'-g','125','-pix_fmt','yuv420p','-f','segment','-segment_time',str(segment),'-segment_atclocktime','1',
             '-reset_timestamps','1','-strftime','1','-segment_format','mp4',
             '-segment_format_options','movflags=+frag_keyframe+empty_moov+default_base_moof',
             str(self.folder/'rec_%Y%m%d_%H%M%S.mp4')]
        import subprocess
        try:self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=(self.folder/'ffmpeg.log').open('ab'))
        except OSError as exc:self.proc=None;self.stats['error']=str(exc);return
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def write(self,index,img):
        if self.proc is None:return
        frames=[]
        if self.last is not None and self.prev is not None:
            missing=min((index-self.last)//2-1,125)        # observations come every 2 camera frames
            frames+=[self.prev]*max(0,missing);self.stats['repeated']+=max(0,missing)
        frames.append(img);self.last=index;self.prev=img
        for f in frames:
            try:self.q.put_nowait(f)
            except queue.Full:self.stats['dropped']+=1

    def run(self):
        try:
            while True:
                f=self.q.get()
                if f is None:break
                self.proc.stdin.write(f.tobytes());self.stats['written']+=1
        except (BrokenPipeError,OSError,ValueError) as exc:self.stats['error']=str(exc);self.proc=None

    def close(self):
        if self.proc is None and not hasattr(self,'thread'):return
        try:self.q.put(None,timeout=5)
        except queue.Full:pass
        self.thread.join(timeout=30)
        proc=getattr(self,'proc',None)
        if proc is not None:
            try:proc.stdin.close();proc.wait(timeout=60)
            except Exception as exc:self.stats['error']=str(exc)
        (self.folder/'recorder.json').write_text(json.dumps(self.stats)+'\n')
