"""Always-on service: the web UI runs continuously, counting runs only during the daily hours.

Inside the counting hours (default 6:00-19:00 local time) it reads the camera and counts with
live_fused.run_session; outside them the camera is not read at all (no stream, no GPU work)
and the page shows the saved counts and the next start time. Restarting inside the hours
continues the day's totals from runs/live_daily/. A failed session (e.g. camera unreachable
at start) is retried after 30 s while the hours last.

  python live_service.py --view plaza                 # 6:00-19:00, web UI on :8080
  python live_service.py --view plaza --hours 6-19 --web-port 8080
"""
import datetime,json,signal,sys,threading,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from live_fused import build_parser,run_session
from live_web import LiveState,serve
from views import load_view


def parse_hours(text):
    start,end=(int(x) for x in text.split('-'))
    if not 0<=start<end<=24:raise ValueError('--hours must look like 6-20')
    return start,end


def window_at(now,hours):
    """(start, end) of today's counting window, as datetimes."""
    day=now.replace(hour=0,minute=0,second=0,microsecond=0)
    return day+datetime.timedelta(hours=hours[0]),day+datetime.timedelta(hours=hours[1])


def next_start(now,hours):
    start,end=window_at(now,hours)
    if now<start:return start
    if now<end:return now
    return start+datetime.timedelta(days=1)


def main():
    p=build_parser();p.description=__doc__
    p.add_argument('--hours',default='6-19',help='Daily counting hours, local time (default 6-19)')
    p.add_argument('--output-root',type=Path,default=ROOT/'runs',help='Session folders are created here')
    p.add_argument('--no-record-video',action='store_true',help='Do not save the visualization video')
    p.set_defaults(view='plaza',record_video=Path.home()/'Videos'/'mitafes-live',record_directions=['rtl','ltr'])
    a=p.parse_args();hours=parse_hours(a.hours);view=load_view(a.view)
    if a.no_record_video:a.record_video=None
    web=LiveState(view['name'],daily_dir=a.daily_dir,hours=hours,entry_direction=view.get('entry_direction'));serve(web,a.web_host,a.web_port)
    print(json.dumps(dict(event='service_started',web_port=a.web_port,hours=a.hours,view=view['name'])),flush=True)
    stop=threading.Event()
    def request_stop(signum,frame):
        print(json.dumps(dict(event='stop_requested',signal=signum)),flush=True);stop.set()
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    while not stop.is_set():
        now=datetime.datetime.now();start,end=window_at(now,hours)
        if start<=now<end:
            web.set_state('starting');web.set_status(next_start=None)
            output=a.output_root/f"live_{now:%Y%m%d_%H%M%S}"
            began=time.time()
            try:
                run_session(a,view,output,web=web,until=end.timestamp(),external_stop=stop)
            except Exception:
                traceback.print_exc();web.set_state('error')
                print(json.dumps(dict(event='session_failed',retry_seconds=30)),flush=True)
                stop.wait(30);continue
            web.save(force=True)
            if time.time()-began<60 and not stop.is_set():stop.wait(30)   # never spin on an input that ends at once
            continue
        nxt=next_start(now,hours)
        web.set_state('off_hours');web.clear_live(next_start=nxt.timestamp())
        print(json.dumps(dict(event='off_hours',next_start=nxt.isoformat(timespec='minutes'))),flush=True)
        # Wake at the start time (re-checked every minute, so clock changes are harmless).
        while not stop.is_set() and datetime.datetime.now()<nxt:stop.wait(min(60,max(1,(nxt-datetime.datetime.now()).total_seconds())))
    web.set_state('stopped');web.save(force=True)


if __name__=='__main__':main()
