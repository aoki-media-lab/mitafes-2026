"""Small built-in web UI for live_fused.py: live counts, recent passages, status and a preview.

Standard library only (no internet access needed by the browser either). Endpoints:
  /              the page
  /api/state     current state as JSON
  /api/stream    the same state pushed once per second (Server-Sent Events)
  /snapshot.jpg  latest preview frame with the counting line and detections (if enabled)
The RTSP URL is never part of the state.
"""
import datetime,json,os,threading,time
from collections import deque
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path

PAGE=(Path(__file__).resolve().parent/'web'/'index.html')


class LiveState:
    """Shared state of the web UI, with one saved file per day.

    With ``daily_dir`` the day's totals, per-minute bins and recent passages are written to
    ``daily_dir/YYYY-MM-DD.json`` every few seconds. A restart on the same day continues from
    that file. At the change of the local date the old day's file is written a last time and
    left in place, and counting continues from zero in the new day's file.
    """
    def __init__(self,view_name,bin_seconds=60,history_bins=24*60,daily_dir=None,hours=None,entry_direction=None):
        self.lock=threading.Lock();self.view=view_name;self.started=time.time()
        # Which counted direction enters the campus (the other one leaves); None = not known.
        if entry_direction not in (None,'rtl','ltr'):raise ValueError('entry_direction must be rtl or ltr')
        self.entry=entry_direction
        self.counts=dict(rtl=0,ltr=0,events=0);self.status={};self.recent=deque(maxlen=40)
        self.bin_seconds=bin_seconds;self.bins={};self.history_bins=history_bins
        self.snapshot=None;self.snapshot_time=None;self.version=0;self.state='starting'
        self.daily_dir=Path(daily_dir) if daily_dir else None
        self.base=dict(rtl=0,ltr=0);self.day=self.today();self.day_started=self.started;self.carried=False
        self.last_save=0.;self.frame_cond=threading.Condition();self.frame_seq=0;self.hours=hours
        if self.daily_dir:
            self.daily_dir.mkdir(parents=True,exist_ok=True)
            path=self.day_file(self.day)
            legacy=self.daily_dir.parent/'live_counts.json'   # single file used before 2026-11-21
            if legacy.exists():
                try:
                    old=json.loads(legacy.read_text())
                    if not self.day_file(old['date']).exists():self._write(old)
                    legacy.rename(legacy.with_suffix('.json.migrated'))
                except (OSError,ValueError,KeyError):pass
            if path.exists():
                saved=json.loads(path.read_text())
                self.base=dict(rtl=saved['rtl'],ltr=saved['ltr']);self.day_started=saved.get('day_started',self.started)
                self.bins={int(t):v for t,v in saved.get('bins',{}).items()}
                self.recent.extend(saved.get('recent',[]));self.carried=True

    @staticmethod
    def today():return datetime.date.today().isoformat()

    def day_file(self,day):return self.daily_dir/f'{day}.json'

    def _snapshot_day(self):
        return dict(date=self.day,day_started=self.day_started,rtl=self.base['rtl']+self.counts['rtl'],
                    ltr=self.base['ltr']+self.counts['ltr'],bins={str(k):v for k,v in self.bins.items()},
                    recent=list(self.recent),view=self.view,saved=time.time())

    def _write(self,data):
        path=self.day_file(data['date']);tmp=path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data));os.replace(tmp,path)

    def _rollover(self):
        """Called with the lock held. Closes the old day and starts the new one from zero."""
        today=self.today()
        if today==self.day:return
        if self.daily_dir:self._write(self._snapshot_day())
        self.day=today;self.base=dict(rtl=-self.counts['rtl'],ltr=-self.counts['ltr'])
        self.bins={};self.recent.clear();self.day_started=time.time();self.carried=False

    def begin_session(self):
        """A new counting session starts its own totals at zero: fold the previous ones in."""
        with self.lock:
            self._rollover()
            self.base['rtl']+=self.counts['rtl'];self.base['ltr']+=self.counts['ltr']
            self.counts=dict(rtl=0,ltr=0,events=0);self.version+=1

    def clear_live(self,**status):
        """Outside the counting hours: no camera image, no live figures."""
        with self.frame_cond:self.snapshot=None;self.snapshot_time=None;self.frame_seq+=1;self.frame_cond.notify_all()
        with self.lock:self.status=dict(status);self.counts['pending']=0;self.version+=1

    def day_data(self,day):
        """Totals and per-minute bins of one date (today from memory, other days from file)."""
        with self.lock:
            if day==self.day:
                return dict(date=day,rtl=self.base['rtl']+self.counts['rtl'],ltr=self.base['ltr']+self.counts['ltr'],
                            bins=[dict(t=t,**v) for t,v in sorted(self.bins.items())],live=True)
        if not self.daily_dir or not self.day_file(day).exists():return None
        d=json.loads(self.day_file(day).read_text())
        return dict(date=day,rtl=d['rtl'],ltr=d['ltr'],rebuilt=d.get('rebuilt',False),live=False,
                    bins=[dict(t=int(t),**v) for t,v in sorted(d.get('bins',{}).items(),key=lambda kv:int(kv[0]))])

    def save(self,force=False):
        if not self.daily_dir or (not force and time.time()-self.last_save<5):return
        with self.lock:
            self._rollover();data=self._snapshot_day()
        self._write(data);self.last_save=time.time()

    def days(self,limit=14):
        """Totals of the saved days, newest first (today included)."""
        if not self.daily_dir:return []
        out=[]
        for path in sorted(self.daily_dir.glob('*.json'),reverse=True)[:limit]:
            try:d=json.loads(path.read_text())
            except (OSError,ValueError):continue
            out.append(dict(date=d['date'],rtl=d['rtl'],ltr=d['ltr'],rebuilt=d.get('rebuilt',False)))
        return out

    def add_events(self,events,source):
        now=time.time()
        with self.lock:
            self._rollover()
            for e in events:
                d=e['direction'];self.recent.appendleft(dict(direction=d,time=now,source=source,
                                                             stream_seconds=float(e['seconds'])))
                b=int(now//self.bin_seconds)*self.bin_seconds
                self.bins.setdefault(b,dict(rtl=0,ltr=0))[d]+=1
            for old in sorted(self.bins)[:-self.history_bins]:self.bins.pop(old)
            self.version+=1

    def set_counts(self,totals):
        with self.lock:
            self._rollover()   # first, so passages after midnight belong to the new day
            self.counts=dict(rtl=totals['rtl'],ltr=totals['ltr'],events=totals['events'],
                             pending=totals.get('pending',0));self.version+=1

    def set_status(self,**status):
        with self.lock:self.status.update(status);self.version+=1

    def set_state(self,state):
        with self.lock:self.state=state;self.version+=1

    def set_snapshot(self,jpeg):
        """Publish one preview frame: kept for /snapshot.jpg and pushed to /video.mjpg viewers."""
        with self.frame_cond:
            self.snapshot=jpeg;self.snapshot_time=time.time();self.frame_seq+=1;self.frame_cond.notify_all()

    def as_dict(self):
        with self.lock:
            now=time.time();c=dict(self.counts)
            c['rtl']+=self.base['rtl'];c['ltr']+=self.base['ltr'];c['events']=c['rtl']+c['ltr']
            return dict(view=self.view,state=self.state,now=now,started=self.day_started,run_started=self.started,
                        carried_over=self.carried,day=self.day,counts=c,hours=self.hours,entry=self.entry,
                        status=dict(self.status),recent=list(self.recent),bin_seconds=self.bin_seconds,
                        bins=[dict(t=t,**v) for t,v in sorted(self.bins.items())],
                        snapshot_time=self.snapshot_time)


def serve(state,host,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,code,body,ctype,extra=()):
            self.send_response(code);self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            for k,v in extra:self.send_header(k,v)
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=self.path.split('?',1)[0]
            if path in ('/','/index.html'):return self.send(200,PAGE.read_bytes(),'text/html; charset=utf-8')
            if path=='/api/state':return self.send(200,json.dumps(state.as_dict()).encode(),'application/json')
            if path=='/api/days':return self.send(200,json.dumps(state.days()).encode(),'application/json')
            if path=='/api/day':
                query=dict(q.split('=',1) for q in self.path.split('?',1)[1].split('&') if '=' in q) if '?' in self.path else {}
                day=query.get('date',state.day)
                if not (len(day)==10 and day[4]==day[7]=='-' and day.replace('-','').isdigit()):return self.send(400,b'bad date','text/plain')
                data=state.day_data(day)
                return self.send(200,json.dumps(data).encode(),'application/json') if data else self.send(404,b'no data','text/plain')
            if path=='/snapshot.jpg':
                jpeg=state.snapshot
                return self.send(200,jpeg,'image/jpeg') if jpeg else self.send(404,b'no snapshot','text/plain')
            if path=='/video.mjpg':
                # Motion JPEG: every browser shows it in a plain <img>. ?fps= limits the rate.
                try:fps=min(25.,max(.2,float(dict(q.split('=',1) for q in self.path.split('?',1)[1].split('&') if '=' in q).get('fps',12.5))))
                except (IndexError,ValueError):fps=12.5
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control','no-store');self.end_headers()
                seen=-1;due=0.
                try:
                    while True:
                        with state.frame_cond:
                            state.frame_cond.wait_for(lambda:state.frame_seq!=seen,timeout=5)
                            seen=state.frame_seq;jpeg=state.snapshot
                        now=time.monotonic()
                        if jpeg is None or now<due-.02:continue
                        due=max(due+1/fps,now-.5)   # keep the average rate, never build up a backlog
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
                        self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):return
            if path=='/api/stream':
                self.send_response(200);self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-store');self.end_headers()
                try:
                    while True:
                        self.wfile.write(b'data: '+json.dumps(state.as_dict()).encode()+b'\n\n');self.wfile.flush()
                        time.sleep(1)
                except (BrokenPipeError,ConnectionResetError):return
            self.send(404,b'not found','text/plain')
    server=ThreadingHTTPServer((host,port),Handler);server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server
