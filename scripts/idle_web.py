"""Off-hours browser view: the same page and URL as live_fused.py --web-port, without the camera.

Shows the most recent counting day (today if it has runs, else the latest day in --runs) with its
total and time-of-day chart; the video area says the counter is off. No camera frames are read or sent.
Exits at --until (epoch seconds) so scripts/run_schedule.sh can start counting.
"""
import argparse,json,secrets,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_web import LiveWeb,day_history

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--setup',type=Path,required=True);p.add_argument('--runs',type=Path,default=ROOT/'runs/live')
p.add_argument('--until',type=float,required=True);p.add_argument('--end-time',default='19:00',help='End of the counting window (chart axis)')
p.add_argument('--web-host',default='127.0.0.1');p.add_argument('--web-port',type=int,default=8080)
p.add_argument('--web-token-file',type=Path)
a=p.parse_args()
setup=json.loads(a.setup.read_text());profile=setup['profile'];W,H=2880,1620
crop=[v*m for v,m in zip(setup.get('roi_crop') or [0,0,1,1],[W,H,W,H])]
token=None
if a.web_token_file:
    f=a.web_token_file.expanduser()
    if not f.exists():f.touch(mode=0o600);f.write_text(secrets.token_urlsafe(18)+'\n')
    token=f.read_text().strip()
days=sorted({d.name[:8] for d in a.runs.glob('20??????_??????') if (d/'live_events.jsonl').exists()})
today=time.strftime('%Y%m%d');day=today if today in days or not days else days[-1]
directions=set(profile.get('count_directions',['ltr','rtl']))
covered=set(directions)
history=day_history(a.runs,directions,date=day,covered=covered)
end=time.mktime(time.strptime(day+a.end_time,'%Y%m%d%H:%M'))
label=f'{int(day[4:6])}/{int(day[6:8])}'
web=LiveWeb(a.web_host,a.web_port,W,H,profile,crop,960,60,history=history,token=token,
            capture=False,day_label=label,hist_end=end,runs=a.runs,day_code=day,day_end=a.end_time,covered=covered)
print(json.dumps(dict(event='idle',day=day,counts=len(history),until=time.strftime('%F %T',time.localtime(a.until)))),flush=True)
stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
while not stop[0] and time.time()<a.until:time.sleep(min(5,max(.1,a.until-time.time())))
web.close()
