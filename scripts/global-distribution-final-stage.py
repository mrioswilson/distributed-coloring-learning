"""Wait for this study only, finish prespecified outputs, preserve logs."""
import json,time,subprocess,os,sys
from pathlib import Path
base=Path(__file__).resolve().parents[1];os.chdir(base)
root=base/'data/ucgnn-global-distribution';out=base/'results/ucgnn-global-distribution'
while True:
    q=json.loads((root/'queue.json').read_text())
    if all(t['status']=='complete' for t in q['tasks']):break
    if q['gpu_seconds']>=43140:raise SystemExit('Budget exhausted: keep pending cells explicit; no final-completion claim')
    time.sleep(10)
time.sleep(2)
py=sys.executable
commands=[
 [py,'-m','dcl.globaldist.supplement'],
 [py,'-m','dcl.globaldist.analyze'],
 [py,'-c','from dcl.globaldist.analyze import independent_power; independent_power()'],
 [py,'-c','from dcl.globaldist.analyze import localized_control; localized_control()'],
 [py,'-m','dcl.globaldist.audit'],
 [py,'-m','dcl.globaldist.report'],
 [py,'-m','pytest','-q','tests/test_globaldist.py','tests/test_globaldist_integration.py','tests/test_localpalettes.py','tests/test_graphs.py','tests/test_oracle.py','--junitxml=results/ucgnn-global-distribution/regression-tests.xml'],
 ['/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic','-X','compile','--keep-logs','paper/ucgnn/main.tex'],
]
for command in commands:
    print('Executing',command,flush=True)
    subprocess.run(command,check=True)
print('Final-stage complete; inspection and delivery packaging remain.',flush=True)
