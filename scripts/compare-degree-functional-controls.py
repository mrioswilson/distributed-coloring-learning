#!/usr/bin/env python3
"""Paired final-success differences, kept distinct from head-query fidelity."""
import json
from pathlib import Path
from dcl.degree.inventory import ROOT
from dcl.degree.report import write_csv,table
from dcl.study.storage import atomic_json


if __name__=='__main__':
    inventory=json.loads((ROOT/'checkpoint-inventory.json').read_text())+json.loads((ROOT/'supplemental-checkpoint-inventory.json').read_text())
    plan=json.loads((ROOT/'confirmatory/design.json').read_text())
    out=[]
    for cf in inventory:
        if not cf['primary'] or cf['teacher'] not in ('COLOR_TRIALS','SPECULATIVE_GM'):continue
        for method in ('matched','passive'):
            ctrls=[r for r in inventory if r['teacher']==cf['teacher'] and r['architecture']==cf['architecture'] and
                   r['seed']==cf['seed'] and r['method']==method and r['update']==cf['update']]
            assert len(ctrls)==1
            control=ctrls[0]
            for condition in plan['conditions']:
                if condition.get('primary_only'):continue
                a=json.loads((ROOT/'confirmatory'/cf['id']/condition['id']/'rows.json').read_text())
                b=json.loads((ROOT/'confirmatory'/control['id']/condition['id']/'rows.json').read_text())
                assert [r['case_id'] for r in a]==[r['case_id'] for r in b]
                ac,bc=sum(r['success'] for r in a),sum(r['success'] for r in b)
                out.append(dict(teacher=cf['teacher'],architecture=cf['architecture'],seed=cf['seed'],update=cf['update'],
                    condition=condition['id'],control=method,executions=len(a),cf_successes=ac,control_successes=bc,
                    cf_only_success=sum(x['success'] and not y['success'] for x,y in zip(a,b)),
                    control_only_success=sum(not x['success'] and y['success'] for x,y in zip(a,b)),
                    difference=(ac-bc)/len(a)))
    write_csv(ROOT/'paired-functional-equal-update.csv',out)
    atomic_json(ROOT/'paired-functional-equal-update.json',out)
    md='\n## Funcionalidad apareada a igual hito\n\n'
    md+='Vectores en orden s94101–s94105: diferencia CF menos matched de éxitos por 1000, con los mismos grafos y cintas. '
    md+='No se mezclan las cinco políticas. CT usa 4000 updates en ambos brazos; GM 500. '
    md+='Las etiquetas totales siguen siendo diferentes. Los controles certificados de hitos anteriores se conservan además en las filas crudas.\n\n'
    table_rows=[]
    for t in ('COLOR_TRIALS','SPECULATIVE_GM'):
        for a in ('normalized','deepsets','transformer'):
            vals=[]
            for c in ('global-n64-d3-mixture1000','global-n64-d4-mixture1000','global-n64-d10-mixture1000','local-n64-d4-mixture1000'):
                rr=sorted([r for r in out if r['teacher']==t and r['architecture']==a and r['control']=='matched' and r['condition']==c],key=lambda r:r['seed'])
                vals.append(','.join(str(r['cf_successes']-r['control_successes']) for r in rr))
            table_rows.append([t,a,*vals])
    md+=table(['Profesor','Arquitectura','Δ3 global','Δ4 global','Δ10 global','Δ4 local'],table_rows)+'\n\n'
    md+='La ventaja funcional CT/E1 del hito4000 se concentra en s94104 y ya existe a grado tres: '
    md+='no debe presentarse toda la diferencia de grado diez como una mejora nueva causada por extrapolar grado. '
    md+='En GM/Transformer, los 181 fallos locales de s94102 son compartidos por CF, matched y pasivo, aunque sus COMMIT difieren en el replay. '
    md+='Mejor fidelidad condicional no implica una mejora funcional en todas las combinaciones.\n'
    path=Path('docs/DEGREE_GENERALIZATION_FAILURE_ANALYSIS.md')
    text=path.read_text().split('\n## Funcionalidad apareada a igual hito')[0]
    path.write_text(text+md)
    print(table_rows)
