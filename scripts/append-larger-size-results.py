#!/usr/bin/env python3
import json
from pathlib import Path
from dcl.degree.inventory import ROOT
from dcl.degree.report import table,write_csv


if __name__=='__main__':
    rows=json.loads((ROOT/'larger-size-pilot/index.json').read_text())
    assert len(rows)==240
    flat=[dict(checkpoint=r['checkpoint_id'],teacher=r['teacher'],architecture=r['architecture'],seed=r['seed'],
        condition=r['condition']['id'],n=r['condition']['n'],delta=r['condition']['delta'],
        successes=r['successes'],executions=r['executions']) for r in rows]
    write_csv(ROOT/'larger-size-pilot/checkpoint-summary.csv',flat)
    text='\n## Ampliación exploratoria de tamaño hasta 4096\n\n'
    text+='Después de la confirmación, otra batería congelada en un namespace separado evalúa los sesenta checkpoints principales de Luby, Johansson, ColorTrials y JP. '
    text+='Son los cuatro profesores paralelos; GM y Greedy conservan su batería ya completada hasta1024 y BE permanece fuera del contraste. '
    text+=f"Se obtuvieron {sum(r['successes'] for r in rows)}/{sum(r['executions'] for r in rows)} ejecuciones correctas. "
    text+='Cada entrada siguiente es un vector de cinco resultados, uno por checkpoint; cada resultado tiene denominador uno. No es certificación ni parte de las 810000 ejecuciones confirmatorias.\n\n'
    trows=[]
    conditions=sorted({r['condition'] for r in flat})
    for teacher,arch in sorted({(r['teacher'],r['architecture']) for r in flat}):
        cells=[]
        for c in conditions:
            rr=sorted([r for r in flat if r['teacher']==teacher and r['architecture']==arch and r['condition']==c],key=lambda r:r['seed'])
            cells.append(','.join(str(r['successes']) for r in rr)+' (cada uno /1)')
        trows.append([teacher,arch,*cells])
    text+=table(['Profesor','Arquitectura',*conditions],trows)+'\n'
    p=Path('docs/DEGREE_GENERALIZATION_RESULTS.md')
    p.write_text(p.read_text().split('\n## Ampliación exploratoria de tamaño hasta 4096')[0]+text)
    p=Path('docs/MULTITEACHER_PAPER_CLOSURE_REVIEW.md')
    s=p.read_text().replace('El piloto regular mantiene n256 al variar grado y Δ3/10 al variar n64–1024.',
        'El piloto regular mantiene n256 al variar grado y Δ3/10 al variar n64–1024; otra extensión exploratoria separada llega a n2048/4096 para los cuatro profesores paralelos, con una ejecución por checkpoint/punto.')
    p.write_text(s)
