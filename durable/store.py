"""SQLite coordinator with bounded admission, lease fencing, and atomic results.

One connection per thread/process. Lease clocks use Unix seconds; tests inject a
logical clock. Side effects guaranteed here are SQLite records in this database,
not arbitrary external tool actions. No secrets are persisted in payloads/logs.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import sqlite3
import time


class Rejected(ValueError):
    pass


def canonical(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode()) > 32768:
        raise Rejected("payload_limit")
    return text


@dataclass(frozen=True)
class Lease:
    job_id: str
    tenant: str
    worker: str
    fence: int
    attempt: int
    payload: dict
    expires: float


class Store:
    def __init__(self, path, credentials, *, max_active=None, tenant_active=None,
                 tenant_running=None, max_attempts=None, clock=time.time):
        supplied = dict(max_active=max_active, tenant_active=tenant_active,
                        tenant_running=tenant_running, max_attempts=max_attempts)
        if any(value is not None and (type(value) is not int or value <= 0) for value in supplied.values()):
            raise ValueError("limits must be positive")
        self.credentials = dict(credentials)
        self.clock = clock
        self.db = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, tenant TEXT NOT NULL, request_key TEXT NOT NULL,
                payload TEXT NOT NULL, state TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0, fence INTEGER NOT NULL DEFAULT 0,
                worker TEXT, lease_until REAL, deadline REAL NOT NULL,
                created REAL NOT NULL, finished REAL, result TEXT, error TEXT,
                UNIQUE(tenant, request_key),
                CHECK(state IN ('pending','leased','succeeded','failed','cancelled')));
            CREATE INDEX IF NOT EXISTS jobs_admission ON jobs(state,tenant,created);
            CREATE TABLE IF NOT EXISTS effects(
                job_id TEXT PRIMARY KEY REFERENCES jobs(id), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL,
                job_id TEXT NOT NULL, kind TEXT NOT NULL, fence INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS runtime_policy(
                id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            PRAGMA user_version=2;
        ''')
        try:
            with self.transaction():
                row = self.db.execute("SELECT value FROM runtime_policy WHERE id=1").fetchone()
                if row:
                    policy = json.loads(row['value'])
                    if any(value is not None and value != policy[key] for key, value in supplied.items()):
                        raise Rejected("policy_mismatch")
                else:
                    if self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]:
                        raise Rejected("unversioned_policy; explicit migration required")
                    policy = dict(max_active=64, tenant_active=16, tenant_running=4, max_attempts=3)
                    policy.update({key: value for key, value in supplied.items() if value is not None})
                    self.db.execute("INSERT INTO runtime_policy(id,value) VALUES(1,?)", (canonical(policy),))
            self.max_active, self.tenant_active = policy['max_active'], policy['tenant_active']
            self.tenant_running, self.max_attempts = policy['tenant_running'], policy['max_attempts']
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def authorize(self, tenant, token):
        expected = self.credentials.get(tenant)
        if expected is None or not isinstance(token, str) or not hmac.compare_digest(expected, token):
            raise Rejected("unauthorized")

    def now(self):
        now = self.clock()
        if not math.isfinite(now):
            raise ValueError("clock must be finite")
        return now

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def event(self, job, kind, now, fence):
        self.db.execute("INSERT INTO events(at,job_id,kind,fence) VALUES(?,?,?,?)", (now, job, kind, fence))

    def expire(self, now):
        for row in self.db.execute("SELECT * FROM jobs WHERE state IN ('pending','leased')").fetchall():
            deadline = row['deadline'] <= now
            expired = row['state'] == 'leased' and row['lease_until'] <= now
            if not deadline and not expired:
                continue
            terminal = deadline or row['attempt'] >= self.max_attempts
            state = 'failed' if terminal else 'pending'
            error = 'deadline' if deadline else 'attempts_exhausted' if terminal else 'lease_expired'
            self.db.execute("UPDATE jobs SET state=?, worker=NULL,lease_until=NULL,fence=fence+1,error=?,finished=? WHERE id=?",
                            (state, error, now if terminal else None, row['id']))
            self.event(row['id'], error, now, row['fence'] + 1)

    def submit(self, tenant, token, request_key, payload, *, ttl=60):
        self.authorize(tenant, token)
        if not isinstance(request_key, str) or not request_key or len(request_key) > 256:
            raise Rejected("invalid_request_key")
        if not isinstance(payload, dict) or not math.isfinite(ttl) or ttl <= 0:
            raise Rejected("invalid_submission")
        text = canonical(payload)
        job_id = hashlib.sha256(canonical([tenant, request_key]).encode()).hexdigest()
        with self.transaction():
            now = self.now()
            self.expire(now)
            old = self.db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old:
                if old['payload'] != text:
                    raise Rejected("idempotency_conflict")
                return job_id
            active = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','leased')").fetchone()[0]
            own = self.db.execute("SELECT COUNT(*) FROM jobs WHERE tenant=? AND state IN ('pending','leased')", (tenant,)).fetchone()[0]
            if active >= self.max_active or own >= self.tenant_active:
                raise Rejected("admission_limit")
            self.db.execute("INSERT INTO jobs(id,tenant,request_key,payload,state,deadline,created) VALUES(?,?,?,?,'pending',?,?)",
                            (job_id, tenant, request_key, text, now + ttl, now))
            self.event(job_id, 'accepted', now, 0)
        return job_id

    def claim(self, tenant, token, worker, *, lease_seconds=5, job_ids=None):
        self.authorize(tenant, token)
        if not worker or len(worker) > 256 or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise Rejected("invalid_lease")
        if job_ids is not None and (not isinstance(job_ids,list) or not job_ids or len(job_ids)>self.max_active):
            raise Rejected("invalid_job_filter")
        with self.transaction():
            now = self.now()
            self.expire(now)
            running = self.db.execute("SELECT COUNT(*) FROM jobs WHERE tenant=? AND state='leased'", (tenant,)).fetchone()[0]
            if running >= self.tenant_running:
                return None
            query="SELECT * FROM jobs WHERE tenant=? AND state='pending'"
            parameters=[tenant]
            if job_ids is not None:
                query += " AND id IN ("+','.join('?' for _ in job_ids)+")"
                parameters.extend(job_ids)
            row = self.db.execute(query+" ORDER BY created,id LIMIT 1", parameters).fetchone()
            if row is None:
                return None
            expires = min(now + lease_seconds, row['deadline'])
            fence, attempt = row['fence'] + 1, row['attempt'] + 1
            self.db.execute("UPDATE jobs SET state='leased',worker=?,lease_until=?,fence=?,attempt=?,error=NULL WHERE id=?",
                            (worker, expires, fence, attempt, row['id']))
            self.event(row['id'], 'leased', now, fence)
            return Lease(row['id'], tenant, worker, fence, attempt, json.loads(row['payload']), expires)

    def valid(self, lease, now):
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (lease.job_id,)).fetchone()
        return row, bool(row and row['tenant'] == lease.tenant and row['worker'] == lease.worker
                         and row['fence'] == lease.fence and row['state'] == 'leased'
                         and row['lease_until'] > now and row['deadline'] > now)

    def renew(self, lease, token, *, lease_seconds=5):
        self.authorize(lease.tenant, token)
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise Rejected("invalid_lease")
        with self.transaction():
            now = self.now()
            self.expire(now)
            row, valid = self.valid(lease, now)
            if not valid:
                return False
            self.db.execute("UPDATE jobs SET lease_until=? WHERE id=?", (min(now + lease_seconds, row['deadline']), lease.job_id))
            return True

    def commit(self, lease, token, result, *, effect=None):
        self.authorize(lease.tenant, token)
        result_text = canonical(result)
        effect_text = None if effect is None else canonical(effect)
        with self.transaction():
            now = self.now()
            self.expire(now)
            row, valid = self.valid(lease, now)
            if (row and row['state'] == 'succeeded' and row['tenant'] == lease.tenant
                    and row['fence'] == lease.fence and row['worker'] == lease.worker
                    and row['result'] == result_text):
                prior = self.db.execute("SELECT value FROM effects WHERE job_id=?", (lease.job_id,)).fetchone()
                if (None if prior is None else prior['value']) == effect_text:
                    return 'duplicate'
            if not valid:
                return 'stale'
            if effect_text is not None:
                self.db.execute("INSERT INTO effects(job_id,value) VALUES(?,?)", (lease.job_id, effect_text))
            self.db.execute("UPDATE jobs SET state='succeeded',result=?,finished=? WHERE id=?", (result_text, now, lease.job_id))
            self.event(lease.job_id, 'committed', now, lease.fence)
            return 'committed'

    def fail(self, lease, token, error, *, retryable=True):
        self.authorize(lease.tenant, token)
        if not isinstance(error, str) or len(error) > 512:
            raise Rejected("invalid_error")
        with self.transaction():
            now = self.now()
            self.expire(now)
            row, valid = self.valid(lease, now)
            if not valid:
                return False
            state = 'pending' if retryable and row['attempt'] < self.max_attempts else 'failed'
            self.db.execute("UPDATE jobs SET state=?,worker=NULL,lease_until=NULL,fence=fence+1,error=?,finished=? WHERE id=?",
                            (state, error, now if state == 'failed' else None, lease.job_id))
            self.event(lease.job_id, 'retry' if state == 'pending' else 'failed', now, lease.fence + 1)
            return True

    def cancel(self, tenant, token, job_id):
        self.authorize(tenant, token)
        with self.transaction():
            now = self.now()
            self.expire(now)
            row = self.db.execute("SELECT * FROM jobs WHERE id=? AND tenant=?", (job_id, tenant)).fetchone()
            if row is None:
                raise Rejected("unknown_job")
            if row['state'] not in ('pending', 'leased'):
                return False
            self.db.execute("UPDATE jobs SET state='cancelled',fence=fence+1,finished=? WHERE id=?", (now, job_id))
            self.event(job_id, 'cancelled', now, row['fence'] + 1)
            return True

    def get(self, tenant, token, job_id):
        self.authorize(tenant, token)
        with self.transaction():
            self.expire(self.now())
            row = self.db.execute("SELECT * FROM jobs WHERE id=? AND tenant=?", (job_id, tenant)).fetchone()
            if row is None:
                raise Rejected("unknown_job")
            return dict(row)
