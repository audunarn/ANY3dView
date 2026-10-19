"""Report same-fixture timings and exact pixels/pick parity without rerendering."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(sys.argv[1]); results={}
for backend in ['gpu','software']:
    before=root/('baseline-controlled-'+backend)
    after=root/('candidate-'+backend+('-final' if backend=='software' else ''))
    b=json.loads((before/'result.json').read_text()); c=json.loads((after/'result.json').read_text())
    assert (b['triangles'],b['viewport'],b['dpr'])==(c['triangles'],c['viewport'],c['dpr'])
    report={}
    for name in ['orbit','field_index','animation']:
        old=b[name]['median_ms']; new=c[name]['median_ms']
        report[name]={'baseline_ms':old,'candidate_ms':new,'speedup':old/new,'candidate_p95_ms':c[name]['p95_ms']}
    report['images']={}
    for name in ['initial','orbit','animation']:
        a=np.asarray(Image.open(before/(name+'.png'))).astype(int)
        z=np.asarray(Image.open(after/(name+'.png'))).astype(int)
        report['images'][name]={'identical':bool(np.array_equal(a,z)),
            'changed_pixels':int(np.any(a!=z,axis=-1).sum()),'max_channel_difference':int(np.abs(a-z).max())}
    report['picking_identical']=b['picks']==c['picks']
    report['animation_picking_identical']=b['animation_picks']==c['animation_picks']
    report['hover_calls']=[b['hover']['calls'],c['hover']['calls']]
    report['idle_ticks']=[b['idle']['timer_ticks'],c['idle']['timer_ticks']]
    report['animation_uploads']=[b['animation']['geometry_uploads'],c['animation']['geometry_uploads']]
    oracle_before=root/('baseline-oracle-'+backend+('2' if backend=='software' else ''))
    oracle_after=root/('candidate-oracle-'+backend+('-final' if backend=='software' else ''))
    if (oracle_after/'oracle.json').exists():
        old=json.loads((oracle_before/'oracle.json').read_text())
        new=json.loads((oracle_after/'oracle.json').read_text())
        report['mutation_picking_identical']=old==new
        report['mutation_images']={}
        for name in old:
            left=np.asarray(Image.open(oracle_before/(name+'.png')))
            right=np.asarray(Image.open(oracle_after/(name+'.png')))
            report['mutation_images'][name]=bool(np.array_equal(left,right))
    results[backend]=report
(root/'comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
assert all(r['picking_identical'] and r['animation_picking_identical'] and all(i['identical'] for i in r['images'].values()) for r in results.values()), 'Visual or pick mismatch'
assert all(r.get('mutation_picking_identical') and all(r.get('mutation_images',{}).values()) for r in results.values()), 'Mutation oracle missing or mismatch'
