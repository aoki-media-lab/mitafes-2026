"""Rebuild per-day totals (runs/live_daily/YYYY-MM-DD.json) from live_fused.py run folders.

Each event's clock time is its 'time' field when present (runs after 2026-11-21). For older
runs it is interpolated from the event's frame number using the 10-second status rows
(observations processed vs. elapsed real time) from the run's start (run_info.json, else the
creation time of live_status.jsonl, which is opened as reading starts). A day that already has
a file is left alone unless --overwrite is given. Rebuilt days are marked "rebuilt": true.
"""
import argparse,datetime,json,os,subprocess
from collections import defaultdict
from pathlib import Path

def run_start(folder):
    info=folder/'run_info.json'
    if info.exists():return json.loads(info.read_text())['started']
    out=subprocess.run(['stat','-c','%W',str(folder/'live_status.jsonl')],capture_output=True,text=True).stdout.strip()
    if not out or out=='0':raise ValueError(f'No start time for {folder}')
    return float(out)

def frame_clock(folder,t0,stride=2):
    """Frame number -> clock time, from the status rows (wall time t, observations so far).

    Old runs counted frames assuming 25 fps; the camera drops to 20 fps after dark, so the
    frame number alone runs slow. Status rows every 10 s tie it to real time."""
    import bisect
    pts=[(0,0.)]
    for line in (folder/'live_status.jsonl').open():
        r=json.loads(line);pts.append((r['processed_obs']*stride,r['t']))
    xs=[f for f,_ in pts]
    def clock(frame):
        i=min(max(bisect.bisect_left(xs,frame),1),len(pts)-1)
        (f0,w0),(f1,w1)=pts[i-1],pts[i]
        if f1==f0:return t0+w1
        return t0+w0+(frame-f0)*(w1-w0)/(f1-f0)
    return clock

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+');p.add_argument('--daily-dir',type=Path,default=Path('runs/live_daily'))
    p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();days=defaultdict(lambda:dict(rtl=0,ltr=0,bins=defaultdict(lambda:dict(rtl=0,ltr=0)),first=None,last=None,runs=set()))
    for folder in a.runs:
        t0=run_start(folder);clock=frame_clock(folder,t0)
        for line in (folder/'live_events.jsonl').open():
            e=json.loads(line);t=e['time'] if 'time' in e else clock(int(e['frame']))
            day=datetime.datetime.fromtimestamp(t).date().isoformat();d=days[day]
            d[e['direction']]+=1;d['bins'][int(t//60)*60][e['direction']]+=1;d['runs'].add(folder.name)
            d['first']=t if d['first'] is None else min(d['first'],t);d['last']=t if d['last'] is None else max(d['last'],t)
    a.daily_dir.mkdir(parents=True,exist_ok=True)
    for day,d in sorted(days.items()):
        path=a.daily_dir/f'{day}.json'
        out=dict(date=day,day_started=d['first'],rtl=d['rtl'],ltr=d['ltr'],bins={str(k):v for k,v in sorted(d['bins'].items())},
                 recent=[],rebuilt=True,sources=sorted(d['runs']),saved=d['last'])
        note='exists, kept' if path.exists() and not a.overwrite else ('dry run' if a.dry_run else 'written')
        print(f"{day}: rtl {d['rtl']} ltr {d['ltr']} ({datetime.datetime.fromtimestamp(d['first']):%H:%M}-{datetime.datetime.fromtimestamp(d['last']):%H:%M}) {note}")
        if note=='written':
            tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(out));os.replace(tmp,path)

if __name__=='__main__':main()
