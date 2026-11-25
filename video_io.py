"""Bounded FFmpeg decode and optional online NVENC annotated output."""
import json,math,queue,subprocess,threading,time
from pathlib import Path
import cv2,numpy as np
from runtime_control import put_buffer


def produce_ffmpeg(a,buffer,stop,meta,wall_start,start_index):
    offset=-round(a.start_seconds*meta['fps']);pending=[]
    try:
        for source in a.source:
            cap=cv2.VideoCapture(str(source));total=int(cap.get(7));w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(5);cap.release()
            if (w,h)!=(meta['width'],meta['height']) or abs(fps-meta['fps'])>.001:raise ValueError('Segment geometry/FPS differs')
            first=max(0,start_index-offset)
            if first>=total:offset+=total;continue
            end=total if a.max_seconds is None else min(total,math.ceil(a.max_seconds*fps-offset))
            if end<=first:break
            selected_first=first+(-(offset+first))%a.stride
            indices=range(offset+selected_first,offset+end,a.stride)
            width=a.decode_width or w;height=round(h*width/w/2)*2
            hardware=['-hwaccel','cuda','-hwaccel_device',str(a.device),'-hwaccel_output_format','cuda'] if a.decoder=='nvdec' else []
            filters=[f'select=not(mod(n+{offset+first}\\,{a.stride}))']
            if a.decoder=='nvdec':filters += [f'scale_cuda={width}:{height}:interp_algo=bilinear','hwdownload','format=nv12','format=bgr24']
            else:filters += [f'scale={width}:{height}:flags=bilinear','format=bgr24']
            command=['ffmpeg','-hide_banner','-v','error',*hardware,'-ss',str(first/fps),'-i',str(source),
                     '-an','-vf',','.join(filters),'-vsync','0','-frames:v',str(len(indices)),
                     '-threads','2','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
            log_path=a.output/f'decode_{offset+first}.log'
            with log_path.open('w') as log:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,bufsize=0)
                try:
                    for index in indices:
                        if stop.is_set():break
                        image=np.empty((height,width,3),np.uint8);view=memoryview(image).cast('B');filled=0
                        while filled<len(view):
                            count=process.stdout.readinto(view[filled:])
                            if not count:raise RuntimeError(f'Incomplete decoded frame {index}; see {log_path}')
                            filled+=count
                        scheduled=wall_start+(index-start_index)/fps
                        if a.pace:stop.wait(max(0,scheduled-time.perf_counter()))
                        pending.append(dict(index=index,image=image,arrival=time.perf_counter(),scheduled=scheduled))
                        if len(pending)==a.batch:
                            while not stop.is_set():
                                try:buffer.put(pending,timeout=.1);break
                                except queue.Full:pass
                            pending=[]
                finally:
                    process.stdout.close()
                    if stop.is_set() and process.poll() is None:process.terminate()
                    code=process.wait(timeout=15)
                if code and not stop.is_set():raise RuntimeError(f'FFmpeg exited {code}; see {log_path}')
            offset+=total
            if stop.is_set() or (a.max_seconds is not None and offset/fps>=a.max_seconds):break
        if pending and not put_buffer(buffer,pending,stop):return
        put_buffer(buffer,None,stop)
    except BaseException as exc:put_buffer(buffer,exc,stop)


class PreviewSink:
    """Finite queue and background H.264 encoder; final flush is timed by runner."""
    def __init__(self,path,width,height,fps,profile,encoder,device):
        self.width=width;self.height=height;self.profile=profile;self.frames=0;self.error=None
        self.queue=queue.Queue(2);self.path=Path(path)
        self.size=(1280,round(height*1280/width/2)*2)
        if encoder=='nvenc':codec=['-c:v','h264_nvenc','-gpu',str(device),'-preset','p4','-tune','ll','-rc','vbr','-cq','21','-b:v','0','-bf','0']
        else:codec=['-c:v','libx264','-preset','veryfast','-threads','4','-crf','21']
        command=['ffmpeg','-hide_banner','-y','-v','error','-filter_threads','1','-filter_complex_threads','1',
                 '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{self.size[0]}x{self.size[1]}',
                 '-framerate',str(fps),'-i','pipe:0',*codec,'-pix_fmt','bgr0' if encoder=='nvenc' else 'yuv420p',
                 '-profile:v','high','-movflags','+faststart','-an',str(path)]
        self.log=self.path.with_suffix('.encode.log').open('w')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=self.log)
        self.thread=threading.Thread(target=self._write,daemon=True);self.thread.start()

    def _write(self):
        try:
            while True:
                image=self.queue.get()
                if image is None:break
                self.process.stdin.write(image.tobytes());self.frames+=1
        except BaseException as exc:self.error=exc
        finally:self.process.stdin.close()

    def put(self,image,record,summary,fps):
        if image.shape[1::-1]!=self.size:image=cv2.resize(image,self.size)
        else:image=image.copy()
        scale=self.size[0]/self.width
        for p in record['people']:
            x1,y1,x2,y2=[round(v*scale) for v in p['xyxy']]
            cv2.rectangle(image,(x1,y1),(x2,y2),(60,220,60),2)
            if 'anchor' in p:cv2.circle(image,tuple(round(v*scale) for v in p['anchor']),4,(60,220,60),2)
        for p in record['umbrellas']:
            x1,y1,x2,y2=[round(v*scale) for v in p['xyxy']];cv2.rectangle(image,(x1,y1),(x2,y2),(255,120,220),2)
        if self.profile.get('line'):
            (x1,y1),(x2,y2)=self.profile['line'];W,H=self.size
            cv2.line(image,(round(x1*W),round(y1*H)),(round(x2*W),round(y2*H)),(0,230,255),2)
        else:
            line=round(self.profile['line_x']*self.size[0]);cv2.line(image,(line,0),(line,self.size[1]-1),(0,230,255),2)
        text=f'GPU | {record["frame"]/fps:.2f}s | LTR {summary["ltr"]} RTL {summary["rtl"]}'
        cv2.putText(image,text,(14,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4)
        cv2.putText(image,text,(14,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
        while True:
            if self.error:raise self.error
            try:self.queue.put(image,timeout=.1);return
            except queue.Full:pass

    def close(self):
        while self.thread.is_alive():
            if self.error:raise self.error
            try:self.queue.put(None,timeout=.1);break
            except queue.Full:pass
        self.thread.join();code=self.process.wait(timeout=30);self.log.close()
        if self.error:raise self.error
        if code:raise RuntimeError(f'Encoder exited {code}')
        return dict(path=str(self.path),frames=self.frames,width=self.size[0],height=self.size[1],generated_online=True)
