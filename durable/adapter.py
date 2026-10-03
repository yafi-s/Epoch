"""Deterministic tool-evaluation adapter. Infrastructure and task scores differ."""
import math


def evaluate(payload):
    """Execute a bounded allowlist. Never eval(), shell commands, or network calls."""
    calls = payload.get('calls')
    if not isinstance(calls, list) or not 1 <= len(calls) <= 16:
        raise ValueError('invalid_plan')
    answers = []
    for call in calls:
        if not isinstance(call, dict) or call.get('tool') not in ('add', 'multiply', 'lookup'):
            raise ValueError('unknown_tool')
        if call['tool'] == 'lookup':
            # Fixed public fixture; no arbitrary path or tenant data access.
            key = call.get('key')
            if not isinstance(key, str):
                raise ValueError('invalid_lookup_key')
            answer = {'lease': 'temporary ownership', 'fence': 'stale-writer rejection'}.get(key)
            if answer is None:
                raise ValueError('missing_fixture')
        else:
            a, b = call.get('a'), call.get('b')
            if any(type(x) not in (int, float) or abs(x) > 1e9 or not math.isfinite(x) for x in (a, b)):
                raise ValueError('invalid_operand')
            answer = a + b if call['tool'] == 'add' else a * b
        answers.append(answer)
    return {'answers': answers, 'task_correct': answers == payload.get('expected')}


def execute_one(store, tenant, token, worker):
    lease = store.claim(tenant, token, worker)
    if lease is None:
        return False
    try:
        result = evaluate(lease.payload)
    except ValueError as exc:
        store.fail(lease, token, str(exc), retryable=False)
    else:
        store.commit(lease, token, result, effect={'answers': result['answers']})
    return True
