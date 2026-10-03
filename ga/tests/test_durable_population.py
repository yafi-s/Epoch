import pytest
from durable.population import evaluate_population
from durable.store import Store
from ga.engine import GAConfig,GAEngine
from ga.search_space import SearchSpace


def test_existing_ga_population_runs_and_resume_uses_committed_results(tmp_path):
    engine=GAEngine(GAConfig(population_size=4,num_generations=2),SearchSpace.stress_test_cnn_space())
    population=engine.initialize()
    store=Store(tmp_path/'jobs.db',{'ga':'offline'})
    calls=[]
    def evaluator(genome):
        calls.append(genome)
        return float(genome['learning_rate'])
    try:
        unrelated=store.submit('ga','offline','unrelated',{'calls':[]})
        evaluate_population(store,population,evaluator,run_id='fixture',tenant='ga',token='offline')
        assert len(calls)==4
        assert [x.fitness for x in population.individuals]==[x.genome['learning_rate']for x in population.individuals]
        assert store.get('ga','offline',unrelated)['state']=='pending'
        evaluate_population(store,population,lambda _:pytest.fail('recomputed committed job'),run_id='fixture',tenant='ga',token='offline')
        assert store.db.execute("SELECT COUNT(*) FROM jobs WHERE state='succeeded'").fetchone()[0]==4
    finally:
        store.close()


def test_default_population_exceeding_quota_is_evaluated_incrementally(tmp_path):
    population=GAEngine(GAConfig(),SearchSpace.stress_test_cnn_space()).initialize()
    store=Store(tmp_path/'jobs.db',{'ga':'offline'})
    try:
        evaluate_population(store,population,lambda _: .75,run_id='default',tenant='ga',token='offline')
        assert len(population.individuals)==20
        assert all(x.fitness==.75 for x in population.individuals)
        assert store.db.execute("SELECT COUNT(*) FROM jobs WHERE state='succeeded'").fetchone()[0]==20
    finally:
        store.close()


def test_long_evaluator_configurable_lease_and_expiry_is_reported(tmp_path):
    clock=[100.]
    store=Store(tmp_path/'jobs.db',{'ga':'offline'},clock=lambda:clock[0])
    population=GAEngine(GAConfig(population_size=4),SearchSpace.stress_test_cnn_space()).initialize()
    def six_second_evaluation(_):
        clock[0]+=6
        return .75
    try:
        evaluate_population(store,population,six_second_evaluation,run_id='long',tenant='ga',token='offline')
        assert all(x.fitness==.75 for x in population.individuals)
        with pytest.raises(RuntimeError,match='lease expired'):
            evaluate_population(store,population,six_second_evaluation,run_id='expired',tenant='ga',token='offline',lease_seconds=5)
        # Expired execution is visible and resumable, not silently scored as zero.
        evaluate_population(store,population,six_second_evaluation,run_id='expired',tenant='ga',token='offline',lease_seconds=60)
        assert all(x.fitness==.75 for x in population.individuals)
        def expired_failure(_):
            clock[0]+=61
            raise ValueError('evaluation error')
        with pytest.raises(RuntimeError,match='lease expired before failure'):
            evaluate_population(store,population,expired_failure,run_id='failed-expired',tenant='ga',token='offline')
    finally:
        store.close()
