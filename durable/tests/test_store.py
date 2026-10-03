import dataclasses
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest

from durable.adapter import evaluate, execute_one
from durable.store import Store, Rejected


CREDS = {'alpha': 'offline-alpha', 'beta': 'offline-beta'}
PLAN = {'calls': [{'tool': 'add', 'a': 2, 'b': 3}], 'expected': [5]}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / 'jobs.db')
        self.clock = 100.
        self.s = Store(self.path, CREDS, clock=lambda: self.clock, tenant_active=4, tenant_running=1)

    def tearDown(self):
        self.s.close()
        self.temp.cleanup()

    def submit(self, key='one', payload=PLAN, **kwargs):
        return self.s.submit('alpha', CREDS['alpha'], key, payload, **kwargs)

    def claim(self, **kwargs):
        return self.s.claim('alpha', CREDS['alpha'], 'worker', **kwargs)

    def test_idempotency_payload_conflict_and_atomic_effect(self):
        job = self.submit()
        self.assertEqual(job, self.submit())
        with self.assertRaisesRegex(Rejected, 'conflict'):
            self.submit(payload={'different': True})
        lease = self.claim()
        output = evaluate(lease.payload)
        self.assertEqual(self.s.commit(lease, CREDS['alpha'], output, effect=5), 'committed')
        self.assertEqual(self.s.commit(lease, CREDS['alpha'], output, effect=5), 'duplicate')
        self.assertEqual(self.s.commit(lease, CREDS['alpha'], output, effect=6), 'stale')
        self.assertEqual(self.s.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0], 1)

    def test_expiry_fences_worker_and_restart_preserves_work(self):
        job = self.submit()
        first = self.claim(lease_seconds=2)
        self.clock += 2
        self.s.close()
        self.s = Store(self.path, CREDS, clock=lambda: self.clock)
        second = self.claim()
        self.assertEqual(second.attempt, 2)
        self.assertGreater(second.fence, first.fence)
        self.assertEqual(self.s.commit(first, CREDS['alpha'], 5), 'stale')
        self.assertEqual(self.s.commit(second, CREDS['alpha'], 5), 'committed')
        self.assertEqual(self.s.get('alpha', CREDS['alpha'], job)['state'], 'succeeded')

    def test_authentication_and_cross_tenant_isolation(self):
        job = self.submit()
        for tenant, token in [('alpha', 'wrong'), ('gamma', 'offline-alpha')]:
            with self.assertRaisesRegex(Rejected, 'unauthorized'):
                self.s.submit(tenant, token, 'two', PLAN)
        for operation in (self.s.get, self.s.cancel):
            with self.assertRaisesRegex(Rejected, 'unknown_job'):
                operation('beta', CREDS['beta'], job)
        lease = self.claim()
        forged = dataclasses.replace(lease, tenant='beta')
        self.assertEqual(self.s.commit(forged, CREDS['beta'], 5), 'stale')
        self.assertEqual(self.s.commit(dataclasses.replace(lease, worker='other'), CREDS['alpha'], 5), 'stale')

    def test_quotas_release_on_terminal_and_other_tenant_can_progress(self):
        jobs = [self.submit(str(i)) for i in range(4)]
        with self.assertRaisesRegex(Rejected, 'admission_limit'):
            self.submit('extra')
        lease = self.claim()
        self.assertIsNone(self.claim())
        self.s.submit('beta', CREDS['beta'], 'one', PLAN)
        self.assertIsNotNone(self.s.claim('beta', CREDS['beta'], 'beta-worker'))
        self.s.cancel('alpha', CREDS['alpha'], jobs[0])
        self.submit('extra')

    def test_cancellation_and_deadline_reject_late_results(self):
        job = self.submit(ttl=1)
        lease = self.claim()
        self.clock += 1
        self.assertFalse(self.s.renew(lease, CREDS['alpha']))
        self.assertEqual(self.s.commit(lease, CREDS['alpha'], 5), 'stale')
        self.assertEqual(self.s.get('alpha', CREDS['alpha'], job)['error'], 'deadline')
        job = self.submit('cancel')
        lease = self.claim()
        self.assertTrue(self.s.cancel('alpha', CREDS['alpha'], job))
        self.assertFalse(self.s.cancel('alpha', CREDS['alpha'], job))
        self.assertEqual(self.s.commit(lease, CREDS['alpha'], 5), 'stale')

    def test_bounded_retries_and_renewal(self):
        job = self.submit()
        first = self.claim(lease_seconds=1)
        self.clock += .5
        self.assertTrue(self.s.renew(first, CREDS['alpha'], lease_seconds=2))
        self.clock += 1
        self.assertEqual(self.s.get('alpha', CREDS['alpha'], job)['state'], 'leased')
        self.s.fail(first, CREDS['alpha'], 'retry')
        for _ in range(2):
            self.s.fail(self.claim(), CREDS['alpha'], 'retry')
        self.assertEqual(self.s.get('alpha', CREDS['alpha'], job)['state'], 'failed')
        self.assertIsNone(self.claim())

    def test_limits_invalid_payload_and_task_correctness_separation(self):
        for payload in ({'huge': 'x' * 32769}, {'bad': float('nan')}):
            with self.assertRaises(ValueError):
                self.submit(payload=payload)
        for ttl in (0, -1, float('inf')):
            with self.assertRaises(Rejected):
                self.submit(ttl=ttl)
        self.assertFalse(evaluate({**PLAN, 'expected': [6]})['task_correct'])
        job = self.submit(payload={'calls': [{'tool': 'shell'}]})
        execute_one(self.s, 'alpha', CREDS['alpha'], 'worker')
        self.assertEqual(self.s.get('alpha', CREDS['alpha'], job)['state'], 'failed')

    def test_concurrent_claims_and_duplicate_submissions(self):
        self.s.close()
        self.path = str(Path(self.temp.name) / 'concurrent.db')
        self.s = Store(self.path, CREDS, tenant_active=32, tenant_running=4)
        for i in range(24):
            self.submit(str(i))
        def worker(i):
            s = Store(self.path, CREDS, tenant_active=32, tenant_running=4)
            try:
                done = 0
                while execute_one(s, 'alpha', CREDS['alpha'], f'w{i}'):
                    done += 1
                return done
            finally:
                s.close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(worker, range(4))), 24)
        self.assertEqual(self.s.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0], 24)
        self.assertEqual(self.s.db.execute("SELECT COUNT(*) FROM jobs WHERE state='succeeded'").fetchone()[0], 24)

    def test_abrupt_process_exit_rolls_back_partial_commit_and_survives_commit(self):
        job = self.submit()
        root = str(Path(__file__).resolve().parents[2])
        script = '''
import os, sys
from durable.store import Store
s=Store(sys.argv[1], {'alpha':'offline-alpha'},clock=lambda:100.)
l=s.claim('alpha','offline-alpha','child')
if sys.argv[2]=='before':
    s.db.execute('BEGIN IMMEDIATE')
    s.db.execute('INSERT INTO effects(job_id,value) VALUES(?,?)',(l.job_id,'5'))
    os._exit(71)
s.commit(l,'offline-alpha',5,effect=5)
os._exit(72)
'''
        result = subprocess.run([sys.executable, '-c', script, self.path, 'before'], cwd=root, timeout=10)
        self.assertEqual(result.returncode, 71)
        self.assertEqual(self.s.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0], 0)
        self.clock += 6
        lease = self.claim()
        self.assertEqual(lease.attempt, 2)
        self.s.commit(lease, CREDS['alpha'], 5, effect=5)
        self.submit('after')
        result = subprocess.run([sys.executable, '-c', script, self.path, 'after'], cwd=root, timeout=10)
        self.assertEqual(result.returncode, 72)
        self.assertEqual(self.s.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0], 2)
        self.assertEqual(self.s.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_expiry_is_checked_after_acquiring_database_lock(self):
        path=str(Path(self.temp.name)/'contention.db')
        owner=Store(path,CREDS)
        try:
            job=owner.submit('alpha',CREDS['alpha'],'one',PLAN)
            lease=owner.claim('alpha',CREDS['alpha'],'worker',lease_seconds=.1)
            blocker=sqlite3.connect(path,isolation_level=None)
            ready=threading.Event()
            proceed=threading.Event()
            def late_commit():
                connection=Store(path,CREDS)
                try:
                    ready.set()
                    if not proceed.wait(5):
                        raise RuntimeError('test synchronization timed out')
                    return connection.commit(lease,CREDS['alpha'],5,effect=5)
                finally:
                    connection.close()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(late_commit)
                self.assertTrue(ready.wait(5))
                blocker.execute('BEGIN IMMEDIATE')
                proceed.set()
                time.sleep(.2)
                blocker.execute('COMMIT')
                self.assertEqual(pending.result(timeout=5),'stale')
            blocker.close()
            self.assertEqual(owner.db.execute('SELECT COUNT(*) FROM effects').fetchone()[0],0)
            self.assertEqual(owner.get('alpha',CREDS['alpha'],job)['state'],'pending')
        finally:
            owner.close()

    def test_database_policy_survives_reopen_and_rejects_conflicts(self):
        path=str(Path(self.temp.name)/'policy.db')
        s=Store(path,CREDS,max_active=1,max_attempts=1)
        job=s.submit('alpha',CREDS['alpha'],'one',PLAN)
        s.close()
        inherited=Store(path,CREDS)
        try:
            with self.assertRaisesRegex(Rejected,'admission_limit'):
                inherited.submit('beta',CREDS['beta'],'two',PLAN)
            lease=inherited.claim('alpha',CREDS['alpha'],'worker')
            self.assertTrue(inherited.fail(dataclasses.replace(lease,attempt=0),CREDS['alpha'],'failed'))
            self.assertEqual(inherited.get('alpha',CREDS['alpha'],job)['state'],'failed')
            with self.assertRaisesRegex(Rejected,'policy_mismatch'):
                Store(path,CREDS,max_active=2)
        finally:
            inherited.close()

    def test_malformed_admitted_json_terminates_without_crashing_worker(self):
        for i,payload in enumerate(({'calls':[{'tool':'lookup','key':[]}]},
                                    {'calls':[{'tool':'add','a':10**400,'b':1}]})):
            job=self.submit(str(i),payload=payload)
            self.assertTrue(execute_one(self.s,'alpha',CREDS['alpha'],'worker'))
            self.assertEqual(self.s.get('alpha',CREDS['alpha'],job)['state'],'failed')


if __name__ == '__main__':
    unittest.main()
