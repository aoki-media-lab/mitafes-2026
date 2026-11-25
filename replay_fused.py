"""Re-run counting + reconciliation of run_fused.py from dumped detections (--dump-detections).

Inference is not repeated, so line position, ROI, counting directions and tracker
parameters can be compared quickly. The roi branch's detections come from the crop
used when they were dumped; a different roi_crop in the setup is rejected.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_fused import Branch,load_setup,branch_specs,line_band
from merge_indexed import merge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='run_fused.py output made with --dump-detections')
    p.add_argument('--setup',type=Path);p.add_argument('--override',default='{}',help='JSON merged into both branch profiles')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-conf',type=float,help='Drop person detections below this confidence (emulates a higher detector floor than the dump\'s)')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an unused output directory')
    setup=load_setup(a.setup);setup['profile'].update(json.loads(a.override))
    dumped=json.loads((a.run/'roi'/'summary.json').read_text())
    meta=dict(width=2880,height=1620,fps=25.0)
    branches=[Branch(n,s,c,o,meta,a.output,None,None,None,None,2) for n,s,c,o in branch_specs(setup)]
    old=json.loads((a.run/'summary.json').read_text()) if (a.run/'summary.json').exists() else {}
    for b,name in zip(branches,['full','roi']):
        with (a.run/name/'detections.jsonl').open() as f:
            for line in f:
                item=json.loads(line)
                if a.min_conf is not None:
                    keep=[i for i,d in enumerate(item['detections']) if int(d[5])!=0 or d[4]>=a.min_conf]
                    item['detections']=[item['detections'][i] for i in keep];item['keypoints']=[item['keypoints'][i] for i in keep]
                b.update(item)
        b.close(dict(mode='replay',source_run=str(a.run)))
    result=merge(a.output/'full',a.output/'roi',a.output/'result',line_band(branches[0].profile,2880))
    result.update(setup=setup,min_conf=a.min_conf);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(ltr=result['ltr'],rtl=result['rtl'],events=result['events'],added=len(result['added']))))


if __name__=='__main__':main()
