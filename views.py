"""Camera-view presets for run_fused.py / live_fused.py.

A view file (views/*.json) fixes everything that depends on where the camera points:
frame size, the base counting profile, profile overrides (counting line, ROI polygon,
margins) and the two detection branches (full frame and a zoomed crop around the line).
``glass`` reproduces the original calibrated preset exactly.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VIEWS=ROOT/'views'


def load_view(name_or_path):
    path=Path(name_or_path)
    if not path.suffix:path=VIEWS/f'{name_or_path}.json'
    view=json.loads(path.read_text())
    base=json.loads((ROOT/view['base_profile']).read_text())
    base.update(view.get('overrides',{}))
    view['profile']=base;view['path']=str(path)
    for b in view['branches']:b['profile']=dict(base,**b.get('profile',{}))
    return view


def counting_line(view):
    from line_counter import CountingLine
    return CountingLine(view['profile'],view['width'],view['height'])


DETECTOR_DEFAULTS=dict(pose_model='yolo11m-pose.pt',conf=.1,half=False,enhance=None,tta_flip=False,iou=.7)


def detector_settings(view,override=None):
    """Person-detector settings of a view: ``detector`` in the view file, then ``override``.

    pose_model  weight file in models/ (or a path) for both person branches; a list of files
                runs them all and unites their detections (gpu_backend.EnsembleModel)
    conf        detection confidence passed to NMS (tracks still need pose_new_track_thresh to start)
    half        FP16 person networks
    enhance     None, "clahe" or "gamma:<g>": applied to each letterboxed network input
    tta_flip    also run the person model on the mirrored input and unite the detections
    iou         NMS IoU for people: boxes overlapping more than this are merged (higher keeps
                more of two people who overlap)
    Without a ``detector`` entry the original settings (yolo11m-pose, 0.1, FP32, none) apply.
    """
    out=dict(DETECTOR_DEFAULTS);out.update(view.get('detector',{}));out.update(override or {})
    unknown=set(out)-set(DETECTOR_DEFAULTS)
    if unknown:raise ValueError(f'Unknown detector settings: {sorted(unknown)}')
    return out


SCHEDULE_KEYS={'conf','iou'}   # applied at NMS time, so they can change while running


def _minutes(text):
    h,m=(int(v) for v in text.split(':'));return h*60+m


def detector_at(view,when,override=None):
    """Detector settings at local time ``when`` (datetime): the view's ``schedule`` entries
    ({"from": "HH:MM", "until": "HH:MM", "detector": {...}}; "24:00" is the end of the day, a window
    may wrap past midnight) override detector_settings(view) inside their window; ``override``
    (command line) wins last. Only SCHEDULE_KEYS may be scheduled."""
    out=detector_settings(view)
    t=when.hour*60+when.minute
    for entry in view.get('schedule',[]):
        bad=set(entry['detector'])-SCHEDULE_KEYS
        if bad:raise ValueError(f'Only {sorted(SCHEDULE_KEYS)} can be scheduled, not {sorted(bad)}')
        a,b=_minutes(entry['from']),_minutes(entry['until'])
        if (a<=t<b) if a<b else (t>=a or t<b):out.update(entry['detector'])
    out.update(override or {})
    return out
