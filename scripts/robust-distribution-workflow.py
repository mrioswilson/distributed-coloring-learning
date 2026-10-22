"""Gate -> immutable commit -> fresh sampling/analysis -> audit/report/PDF.

This orchestration never changes a test based on new p-values. It stops if
validation fails. Visual review and final delivery remain an explicit last
step after all processes finish.
"""
from pathlib import Path
import json,time,sys,subprocess,os,xml.etree.ElementTree as ET

base=Path(__file__).resolve().parents[1];os.chdir(base)
sys.path.insert(0,str(base/'src'))
from dcl.robustdist.common import OUT,DATA,read,atomic_json,file_hash
from dcl.study.storage import now

def run(module,*args):
    cmd=[sys.executable,'-m',module,*args];print('Executing',cmd,flush=True);subprocess.run(cmd,check=True)

def ready(*names):return all((OUT/name).exists() for name in names)

if not (OUT/'freeze.json').exists():
    while not ready('benchmark-validation.json','actual-null-validation.json'):time.sleep(10)
    run('dcl.robustdist.accept','--round','1')
    while not ready('benchmark-validation2.json','actual-null-validation2.json','benchmark-sensitivity2.json','tail-validation.json'):time.sleep(10)
    run('dcl.robustdist.accept','--round','2')
    tree=ET.parse(OUT/'regression-tests.xml').getroot()
    assert not list(tree.iter('failure')) and not list(tree.iter('error'))
    assert len(list(tree.iter('testcase')))>=65
    # Preserve the exact first-round source independently of subsequent work.
    first=read(OUT/'validation-freeze.json')
    for name,digest in first['source_hashes'].items():
        raw=subprocess.check_output(['git','show',f"{first['commit']}:{name}"])
        import hashlib
        assert hashlib.sha256(raw).hexdigest()==digest,name
        target=DATA/'validation-v1-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    run('dcl.robustdist.freeze')
    subprocess.run(['git','add','src/dcl/robustdist','tests/test_robustdist.py','tests/test_robustdist_holdout.py','scripts/robust-distribution-workflow.py','docs/OLD_TEST_STATISTICAL_AUDIT.md','docs/ROBUST_TEST_DEVELOPMENT_PLAN.md','docs/ROBUST_INSTRUMENT_VALIDATION.md','docs/ROBUST_INSTRUMENT_VALIDATION_V2.md','docs/ROBUST_COLORING_DISTRIBUTION_THEORY.md','docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md',str(OUT)],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    subprocess.run(['git','commit','-q','-m','Freeze validated robust coloring protocol before independent confirmation'],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    atomic_json(OUT/'protocol-commit.json',dict(commit=commit,created=now(),protocol_sha256=file_hash('docs/ROBUST_GLOBAL_DISTRIBUTION_PROTOCOL.md'),before_confirmation=True))
    print('CONFIRMATION FROZEN AT',commit,flush=True)

if not (DATA/'queue-confirmatory.json').exists() or not all(t['status']=='complete' for t in read(DATA/'queue-confirmatory.json')['tasks']):
    with (DATA/'confirmatory-sampling.log').open('a') as sampling_log,(DATA/'confirmatory-analysis.log').open('a') as analysis_log:
        sampler=subprocess.Popen([sys.executable,'-m','dcl.robustdist.sampling','confirmatory'],stdout=sampling_log,stderr=subprocess.STDOUT)
        analyzer=subprocess.Popen([sys.executable,'-m','dcl.robustdist.confirmation','--watch'],stdout=analysis_log,stderr=subprocess.STDOUT)
        atomic_json(DATA/'workflow-processes.json',dict(parent=os.getpid(),sampler=sampler.pid,analyzer=analyzer.pid,started=now()))
        while sampler.poll() is None:
            if analyzer.poll() is not None and analyzer.returncode:
                # Sampling can finish independently; preserve all banks and stop.
                print('Analyzer failed; sampling remains separately resumable',flush=True)
                raise RuntimeError('Inspect confirmation analysis log')
            time.sleep(10)
        assert sampler.returncode==0,'Sampling process failed'
        state=read(DATA/'queue-confirmatory.json')
        if not all(t['status']=='complete' for t in state['tasks']):
            analyzer.terminate();analyzer.wait();raise RuntimeError('Budget stopped sampling; retain and report incomplete cells')
        assert analyzer.wait()==0,'Confirmation analysis failed'
elif not (OUT/'primary.json').exists():run('dcl.robustdist.confirmation','--watch')

run('dcl.robustdist.audit')
run('dcl.robustdist.report')
subprocess.run(['/workspaces/AlgoLearning/ucgnn-paper-study/.tools/tectonic','-X','compile','--keep-logs','paper/ucgnn/main.tex'],check=True)
print('All experiments and machine checks finished. Final visual PDF review and delivery packaging remain.',flush=True)
