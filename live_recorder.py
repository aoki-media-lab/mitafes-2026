"""Saves the live counting as a visualization video (counted_overlay), never slowing the count.

The counting loop hands over each processed frame with its people and every count event.
Frames are reduced to the output size at once and kept for ``delay`` seconds before they are
drawn: the live reconciliation confirms secondary (roi) counts ~7 s after the passage, so
the delay lets the video show exactly the counts the totals contain, including the box of a
person found only by the zoomed branch.

* Hand-over never blocks: a full queue drops the frames (counted in ``dropped_frames``).
* Any failure (encoder, disk) stops the recording only; counting goes on.
* One file per clock hour, fragmented MP4, so a file cut by a crash or power loss still plays.
* Recording stops when free disk space falls below ``min_free_gb``.
"""
import json,queue,shutil,subprocess,threading,time,traceback
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2

from counted_overlay import CountedOverlay

ROI_ID_OFFSET=10_000_000   # roi-branch track ids are drawn as id + offset (as merge_indexed offsets them)


class LiveRecorder(threading.Thread):
    def __init__(self,view,directory,base=None,directions=('rtl',),size=(1920,1080),stride=2,delay=8.,
                 cq=23,min_free_gb=20.,queue_batches=3):
        super().__init__(daemon=True,name='live-recorder')
        self.view=view;self.directory=Path(directory);self.size=size;self.cq=cq;self.min_free_gb=min_free_gb
        self.fps=float(view['fps']);self.stride=stride;self.delay_frames=round(delay*self.fps)
        self.overlay=CountedOverlay(view,size,directions,stride,base,roi_offset=ROI_ID_OFFSET)
        self.frames=queue.Queue(maxsize=queue_batches);self.events=queue.SimpleQueue()
        self.buffer=deque();self.added_roi=set();self.latest_index=-1
        self.encoder=None;self.segment_hour=None;self.segment_path=None;self.segments=[]
        self.stats=dict(recorded_frames=0,dropped_frames=0,segments=0,failed=None,stopped_for_disk=False)
        self.last_disk_check=0.;self.closed=False

    # ---- called from the counting loop (must never block) ----
    def put(self,items):
        """items: [(index, wall_time, full_image, full_people, roi_people)]"""
        if self.closed or self.stats['failed'] or self.stats['stopped_for_disk']:return
        try:self.frames.put_nowait(items)
        except queue.Full:self.stats['dropped_frames']+=len(items)

    def add_events(self,events,source):
        for e in events:
            e=dict(e,frame=int(e['frame']),track_id=int(e['track_id']))
            if source=='roi':e['track_id']+=ROI_ID_OFFSET
            self.events.put(e)

    def close(self,timeout=60):
        """Finish: draw everything still buffered and close the file."""
        if self.closed:return
        self.closed=True
        while True:
            try:self.frames.put(None,timeout=1);break
            except queue.Full:
                if not self.is_alive():break
        self.join(timeout)

    # ---- recorder thread ----
    def run(self):
        try:
            while True:
                items=self.frames.get()
                self._drain_events()
                if items is None:break
                if self.stats['failed'] or self.stats['stopped_for_disk']:continue
                for index,wall,image,full_people,roi_people in items:
                    small=cv2.resize(image,self.size,interpolation=cv2.INTER_AREA)
                    self.buffer.append((index,wall,small,full_people,roi_people));self.latest_index=max(self.latest_index,index)
                while self.buffer and self.buffer[0][0]<=self.latest_index-self.delay_frames:self._render(self.buffer.popleft())
            while self.buffer and not (self.stats['failed'] or self.stats['stopped_for_disk']):self._render(self.buffer.popleft())
        except Exception as exc:
            self.stats['failed']=f'{type(exc).__name__}: {exc}'
            print(json.dumps(dict(event='recorder_failed',error=self.stats['failed'])),flush=True);traceback.print_exc()
        finally:
            self.buffer.clear();self._close_segment()
            while True:                      # free any frames still queued
                try:self.frames.get_nowait()
                except queue.Empty:break

    def _drain_events(self):
        while True:
            try:e=self.events.get_nowait()
            except queue.Empty:return
            if e['track_id']>=ROI_ID_OFFSET:self.added_roi.add(e['track_id'])
            self.overlay.add_event(e)

    def _render(self,item):
        index,wall,small,full_people,roi_people=item
        self._drain_events()
        when=datetime.fromtimestamp(wall)
        if not self._ensure_segment(when):return
        people=list(full_people)+[dict(q,id=q['id']+ROI_ID_OFFSET) for q in roi_people if q['id']+ROI_ID_OFFSET in self.added_roi]
        self.overlay.draw(small,index,people,when)
        self.encoder.stdin.write(small.tobytes());self.stats['recorded_frames']+=1

    def _ensure_segment(self,when):
        now=time.monotonic()
        if now-self.last_disk_check>=30:
            self.last_disk_check=now
            self.directory.mkdir(parents=True,exist_ok=True)
            free=shutil.disk_usage(self.directory).free/2**30
            if free<self.min_free_gb:
                self.stats['stopped_for_disk']=True;self._close_segment()
                print(json.dumps(dict(event='recorder_stopped_low_disk',free_gb=round(free,1))),flush=True)
                return False
        hour=when.strftime('%Y-%m-%d %H')
        if self.encoder is None or hour!=self.segment_hour:
            self._close_segment();self.segment_hour=hour
            folder=self.directory/when.strftime('%Y-%m-%d');folder.mkdir(parents=True,exist_ok=True)
            path=folder/f"{when:%H%M%S}.mp4";k=1
            while path.exists():path=folder/f"{when:%H%M%S}_{k}.mp4";k+=1
            command=['ffmpeg','-hide_banner','-nostdin','-v','error','-n','-f','rawvideo','-pixel_format','bgr24',
                     '-video_size',f'{self.size[0]}x{self.size[1]}','-framerate',str(self.fps/self.stride),'-i','pipe:0',
                     '-c:v','h264_nvenc','-preset','p4','-rc','vbr','-cq',str(self.cq),'-b:v','0','-g',str(round(10*self.fps/self.stride)),
                     '-pix_fmt','yuv420p','-movflags','+frag_keyframe+empty_moov+default_base_moof','-an',str(path)]
            self.encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=(path.with_suffix('.log')).open('w'))
            self.segment_path=path;self.segments.append(str(path));self.stats['segments']+=1
            print(json.dumps(dict(event='recorder_segment',path=str(path))),flush=True)
        return True

    def _close_segment(self):
        if self.encoder is None:return
        encoder,path=self.encoder,self.segment_path;self.encoder=None
        try:encoder.stdin.close()
        except OSError:pass
        try:code=encoder.wait(timeout=30)
        except subprocess.TimeoutExpired:encoder.kill();code=-9
        log=path.with_suffix('.log')
        if code==0 and log.exists() and log.stat().st_size==0:log.unlink()
        if code!=0:print(json.dumps(dict(event='recorder_encoder_exit',code=code,path=str(path))),flush=True)
