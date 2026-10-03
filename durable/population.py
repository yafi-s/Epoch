"""Bridge existing Epoch GA individuals to the durable evaluator.

Job identity includes run/generation/index, with payload-conflict detection. Supply
a pure local evaluator (or an adapter whose external effects are idempotent).
"""
import json
import math


def evaluate_population(store, population, evaluator, *, run_id, tenant, token,
                        lease_seconds=60, job_ttl=3600):
    # Incremental admission works even when the population exceeds tenant quota.
    for i, individual in enumerate(population.individuals):
        job = store.submit(tenant, token, f'{run_id}/{population.generation}/{i}',
                           {'genome': individual.genome}, ttl=job_ttl)
        state = store.get(tenant, token, job)
        if state['state'] not in ('succeeded','failed','cancelled'):
            lease = store.claim(tenant, token, 'local-ga', job_ids=[job], lease_seconds=lease_seconds)
            if lease is None:
                raise RuntimeError('Outstanding leased job; resume after lease expiration')
            try:
                score = float(evaluator(lease.payload['genome']))
                if not math.isfinite(score):
                    raise ValueError('non-finite fitness')
            except Exception as exc:
                if not store.fail(lease, token, type(exc).__name__, retryable=False):
                    raise RuntimeError('Evaluation lease expired before failure; increase lease_seconds and resume') from exc
            else:
                outcome = store.commit(lease, token, {'fitness':score})
                if outcome not in ('committed', 'duplicate'):
                    raise RuntimeError('Evaluation lease expired; increase lease_seconds and resume')
            state = store.get(tenant, token, job)
        individual.fitness = json.loads(state['result'])['fitness'] if state['state']=='succeeded' else 0.
    return population
