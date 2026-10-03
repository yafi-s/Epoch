"""Bounded, offline failure/load qualification. Run: python -m durable.experiment."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from durable.adapter import evaluate, execute_one
from durable.store import Store, Rejected

CREDS = {'eval': 'public-offline-fixture'}
TOKEN = CREDS['eval']
KINDS = ('normal', 'lease_expiry', 'timeout', 'duplicate', 'coordinator_restart', 'cancellation', 'tool_failure')


def plan(i):
    return {'calls': [{'tool': 'add', 'a': i, 'b': 7}], 'expected': [i + 7]}


def percentile(values, q):
    # Empirical nearest-rank quantile; with eight recoveries p99 is the maximum.
    return sorted(values)[max(0, min(len(values)-1, math.ceil(len(values)*q)-1))] if values else None


def fault_matrix(path, repetitions=32):
    clock = [100.]
    s = Store(path, CREDS, clock=lambda: clock[0])
    rows = []
    for kind in KINDS:
        for i in range(repetitions):
            payload = {'calls': [{'tool': 'unsupported'}]} if kind == 'tool_failure' else plan(i)
            start = time.perf_counter()
            job = s.submit('eval', TOKEN, f'{kind}-{i}', payload, ttl=1 if kind == 'timeout' else 60)
            first = s.claim('eval', TOKEN, 'first', lease_seconds=1)
            stale = None
            if kind == 'coordinator_restart':
                s.close()
                s = Store(path, CREDS, clock=lambda: clock[0])
            if kind in ('lease_expiry', 'coordinator_restart'):
                clock[0] += 1.001
                second = s.claim('eval', TOKEN, 'replacement')
                stale = s.commit(first, TOKEN, evaluate(first.payload), effect=i + 7)
                assert stale == 'stale'
                assert s.commit(second, TOKEN, evaluate(second.payload), effect=i + 7) == 'committed'
            elif kind == 'timeout':
                clock[0] += 1.001
                assert s.commit(first, TOKEN, evaluate(first.payload)) == 'stale'
            elif kind == 'cancellation':
                s.cancel('eval', TOKEN, job)
                assert s.commit(first, TOKEN, evaluate(first.payload)) == 'stale'
            elif kind == 'tool_failure':
                try:
                    evaluate(first.payload)
                except ValueError as exc:
                    s.fail(first, TOKEN, str(exc), retryable=False)
            else:
                output = evaluate(first.payload)
                assert s.commit(first, TOKEN, output, effect=i + 7) == 'committed'
                if kind == 'duplicate':
                    assert s.submit('eval', TOKEN, f'{kind}-{i}', payload) == job
                    assert s.commit(first, TOKEN, output, effect=i + 7) == 'duplicate'
            row = s.get('eval', TOKEN, job)
            expected = 'cancelled' if kind == 'cancellation' else 'failed' if kind in ('timeout','tool_failure') else 'succeeded'
            assert row['state'] == expected
            rows.append({'episode': f'{kind}-{i}', 'kind': kind, 'state': row['state'],
                         'attempts': row['attempt'], 'stale_result': stale,
                         'task_correct': json.loads(row['result'])['task_correct'] if row['result'] else None,
                         'evaluation_ms': (time.perf_counter()-start)*1000,
                         'fault_clock': 'logical; lease expiry models worker loss'})
    committed = s.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0]
    assert committed == repetitions * 4
    assert s.db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','leased')").fetchone()[0] == 0
    assert s.db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    s.close()
    return rows, committed


def process_loss(path, count=8):
    rows = []
    s = Store(path, CREDS)
    script = '''
import os,sys
from durable.store import Store
s=Store(sys.argv[1], {'eval':'public-offline-fixture'})
l=s.claim('eval','public-offline-fixture','process',lease_seconds=.25)
assert l is not None
os._exit(73)
'''
    for i in range(count):
        job = s.submit('eval', TOKEN, str(i), plan(i))
        start = time.perf_counter()
        child = subprocess.run([sys.executable, '-c', script, str(path)], timeout=10)
        assert child.returncode == 73
        while (lease := s.claim('eval', TOKEN, 'replacement')) is None:
            if time.perf_counter()-start > 3:
                raise RuntimeError('recovery exceeded 3 seconds')
            time.sleep(.005)
        s.commit(lease, TOKEN, evaluate(lease.payload), effect=i+7)
        assert s.get('eval', TOKEN, job)['state'] == 'succeeded'
        rows.append({'episode': i, 'worker_exit_code': child.returncode, 'attempts': lease.attempt,
                     'submit_to_recovery_ms': (time.perf_counter()-start)*1000,
                     'lease_seconds': .25, 'state': 'succeeded'})
    s.close()
    return rows


def load(path, workers, requests=256):
    s = Store(path, CREDS, max_active=requests, tenant_active=requests)
    start = time.perf_counter()
    admitted = [s.submit('eval', TOKEN, str(i), plan(i)) for i in range(requests)]
    admission_done = time.perf_counter()
    def worker(i):
        store = Store(path, CREDS, max_active=requests, tenant_active=requests)
        try:
            while execute_one(store, 'eval', TOKEN, str(i)):
                pass
        finally:
            store.close()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    end = time.perf_counter()
    details = [dict(row) for row in s.db.execute('''
        SELECT j.id, j.state, j.attempt, j.created, j.finished,
               MIN(e.at)-j.created AS queue_wait_s
        FROM jobs j JOIN events e ON e.job_id=j.id AND e.kind='leased' GROUP BY j.id
    ''')]
    assert len(details) == requests and all(row['state']=='succeeded' for row in details)
    assert s.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0] == requests
    latencies = [(row['finished']-row['created'])*1000 for row in details]
    waits = [row['queue_wait_s']*1000 for row in details]
    s.close()
    return {'workers': workers, 'offered': requests, 'admitted': len(admitted), 'completed': requests,
            'offered_admission_rate_per_s': requests/(admission_done-start),
            'observed_completion_rate_per_s': requests/(end-start),
            'elapsed_s': end-start, 'latency_p50_ms': percentile(latencies,.5),
            'latency_p99_ms': percentile(latencies,.99), 'queue_wait_p99_ms': percentile(waits,.99),
            'scope': 'burst of deterministic CPU tool fixtures; includes all submissions and SQLite commits'}, details


def admission_probe(path):
    s = Store(path, CREDS, max_active=16, tenant_active=16)
    accepted, rejected = 0, 0
    ids=[]
    for i in range(64):
        try:
            ids.append(s.submit('eval', TOKEN, str(i), plan(i)))
            accepted += 1
        except Rejected as exc:
            assert str(exc) == 'admission_limit'
            rejected += 1
    for job in ids:
        assert s.cancel('eval',TOKEN,job)
    assert s.db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','leased')").fetchone()[0]==0
    s.close()
    assert (accepted,rejected)==(16,48)
    return {'offered':64,'admitted':accepted,'rejected':rejected,'active_bound':16,'terminal_cancellations':accepted}


def retry_baseline(repetitions=32):
    """Executed in-memory retry worker, same logical-clock episode matrix.

    Deliberately simple baseline: no durable queue, commit fencing or effect key.
    This is a structural correctness comparison, not a performance claim.
    """
    rows=[]
    for kind in KINDS:
        for i in range(repetitions):
            jobs={'job':{'state':'leased','result':None}}
            effects=[]
            def commit():
                jobs['job']={'state':'succeeded','result':evaluate(plan(i))}
                effects.append(i+7)
            if kind=='coordinator_restart':
                jobs.clear()
            elif kind=='tool_failure':
                try:
                    evaluate({'calls':[{'tool':'unsupported'}]})
                except ValueError:
                    jobs['job']['state']='failed'
            else:
                if kind=='cancellation': jobs['job']['state']='cancelled'
                if kind=='timeout': jobs['job']['state']='failed'
                # Replayed/late completions are accepted without a lease fence.
                commit()
                if kind in ('duplicate','lease_expiry'): commit()
            expected='cancelled' if kind=='cancellation' else 'failed' if kind in ('timeout','tool_failure') else 'succeeded'
            state=jobs['job']['state'] if jobs else 'lost'
            rows.append({'episode':f'{kind}-{i}','kind':kind,'state':state,
                         'expected_state':expected,'wrong_terminal':state!=expected,
                         'effects':len(effects),'duplicate_effects':max(0,len(effects)-1)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/evidence/durable')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        faults, effects = fault_matrix(work/'faults.db')
        process = process_loss(work/'process.db')
        loads, load_details = [], []
        for workers in (1,4):
            for repeat in range(3):
                result, details = load(work/f'load-{workers}-{repeat}.db', workers)
                loads.append({**result,'repeat':repeat})
                load_details.extend({'workers':workers,'repeat':repeat,**row} for row in details)
        admission = admission_probe(work/'admission.db')
    baseline=retry_baseline()
    for name, rows in [('faults', faults), ('process-loss',process), ('load',loads), ('load-jobs',load_details),('retry-baseline',baseline)]:
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf8')
    summary = {'episodes':len(faults),'terminal_counts':dict(Counter(row['state'] for row in faults)),
               'atomic_effects':effects,'duplicate_atomic_effects':0,'silent_loss':0,
               'real_process_exits':len(process),'recovery_p99_ms':percentile([r['submit_to_recovery_ms'] for r in process],.99),
               'admission':admission,'loads':loads,
               'baseline':{'name':'in-memory retry queue without fencing','scope':'executed logical fault comparison, not performance measurement',
                           'episodes':len(baseline),'lost_on_restart':sum(r['state']=='lost' for r in baseline),
                           'duplicate_effects':sum(r['duplicate_effects'] for r in baseline),
                           'wrong_terminal_outcomes':sum(r['wrong_terminal'] for r in baseline)},
               'limits':['No remote workers, LLM evaluation, external side effects, or production use measured.',
                         '224 fault episodes use a logical clock; eight separate process exits use wall time.',
                         'Exactly-once guarantee applies only to the same SQLite transaction.',
                         'Process exit is not power-loss or filesystem corruption testing.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    sources = sorted(Path('durable').rglob('*.py'))
    manifest = {'python':sys.version,'platform':platform.platform(),'schema_version':1,
                'command':'python -m durable.experiment', 'seed':'fixed episode IDs 0..31',
                'source_hash_format':'sha256-lf-normalized-text',
                'source_sha256':{str(p):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources},
                'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json*') if p.name != 'manifest.json'}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('loads','baseline','limits')}))


if __name__ == '__main__':
    main()
