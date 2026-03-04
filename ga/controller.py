"""GA Controller: bridges the GAEngine with the C++ scheduler via gRPC."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

import grpc

sys.path.insert(0, "generated/python")
import epoch_pb2
import epoch_pb2_grpc

from ga.engine import GAConfig, GAEngine
from ga.population import Population
from ga.search_space import SearchSpace

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Configuration for the GA controller.

    Attributes:
        scheduler_address: gRPC address of the scheduler.
        ga_config: Configuration for the genetic algorithm.
        poll_interval: Seconds between polling for generation results.
    """

    scheduler_address: str = "localhost:50051"
    ga_config: GAConfig = field(default_factory=GAConfig)
    poll_interval: float = 0.1
    rpc_timeout_s: float = 10.0
    generation_timeout_s: float = 300.0
    progress_log_interval_s: float = 10.0
    max_wall_clock_s: float = 0.0
    expected_workers: int = 0


@dataclass
class GenerationKPI:
    """Per-generation throughput and latency proxy metrics."""

    generation: int
    jobs_count: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    active_workers: int = 0
    wall_clock_ms: int = 0
    train_total_ms: int = 0
    train_p50_ms: int = 0
    train_p90_ms: int = 0
    ideal_wall_ms: float = 0.0
    queue_overhead_ms: float = 0.0
    queue_overhead_pct: float = 0.0
    gen_jobs_per_sec: float = 0.0
    dispatch_latency_p50_ms: float = 0.0
    dispatch_latency_p90_ms: float = 0.0
    dispatch_latency_max_ms: float = 0.0
    worker_idle_gap_p50_ms: float = 0.0
    worker_idle_gap_p90_ms: float = 0.0
    worker_idle_gap_max_ms: float = 0.0
    queue_wait_p50_ms: float = 0.0
    queue_wait_p90_ms: float = 0.0
    queue_wait_max_ms: float = 0.0
    dispatch_samples: int = 0
    idle_gap_samples: int = 0
    queue_wait_samples: int = 0


@dataclass
class RunStats:
    """Aggregate run-level job status counts and controller stop metadata."""

    total_results: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    unknown: int = 0
    jobs_planned: int = 0
    jobs_submitted: int = 0
    generations_planned: int = 0
    generations_completed: int = 0
    stopped_early: bool = False
    stop_reason: str = "completed"
    generation_kpis: list[GenerationKPI] = field(default_factory=list)


class GAController:
    """Orchestrates the full GA optimization loop via the scheduler.

    Initializes a population, submits each generation to the scheduler,
    polls for results, assigns fitness, evolves, and repeats.

    Attributes:
        config: Controller configuration.
        metrics_store: Optional MetricsStore for recording per-generation stats.
    """

    def __init__(
        self,
        config: ControllerConfig,
        metrics_store: object | None = None,
        search_space: SearchSpace | None = None,
    ) -> None:
        self.config = config
        self.metrics_store = metrics_store
        self._engine = GAEngine(config.ga_config, space=search_space)
        self._channel: grpc.Channel | None = None
        self._stub: epoch_pb2_grpc.SchedulerControlStub | None = None
        self._run_stats = RunStats()
        self._run_start_monotonic = 0.0

    @property
    def run_stats(self) -> RunStats:
        """Return aggregate result-status counts from the most recent run."""
        return RunStats(
            total_results=self._run_stats.total_results,
            completed=self._run_stats.completed,
            failed=self._run_stats.failed,
            timed_out=self._run_stats.timed_out,
            unknown=self._run_stats.unknown,
            jobs_planned=self._run_stats.jobs_planned,
            jobs_submitted=self._run_stats.jobs_submitted,
            generations_planned=self._run_stats.generations_planned,
            generations_completed=self._run_stats.generations_completed,
            stopped_early=self._run_stats.stopped_early,
            stop_reason=self._run_stats.stop_reason,
            generation_kpis=[
                GenerationKPI(
                    generation=kpi.generation,
                    jobs_count=kpi.jobs_count,
                    completed=kpi.completed,
                    failed=kpi.failed,
                    timed_out=kpi.timed_out,
                    active_workers=kpi.active_workers,
                    wall_clock_ms=kpi.wall_clock_ms,
                    train_total_ms=kpi.train_total_ms,
                    train_p50_ms=kpi.train_p50_ms,
                    train_p90_ms=kpi.train_p90_ms,
                    ideal_wall_ms=kpi.ideal_wall_ms,
                    queue_overhead_ms=kpi.queue_overhead_ms,
                    queue_overhead_pct=kpi.queue_overhead_pct,
                    gen_jobs_per_sec=kpi.gen_jobs_per_sec,
                    dispatch_latency_p50_ms=kpi.dispatch_latency_p50_ms,
                    dispatch_latency_p90_ms=kpi.dispatch_latency_p90_ms,
                    dispatch_latency_max_ms=kpi.dispatch_latency_max_ms,
                    worker_idle_gap_p50_ms=kpi.worker_idle_gap_p50_ms,
                    worker_idle_gap_p90_ms=kpi.worker_idle_gap_p90_ms,
                    worker_idle_gap_max_ms=kpi.worker_idle_gap_max_ms,
                    queue_wait_p50_ms=kpi.queue_wait_p50_ms,
                    queue_wait_p90_ms=kpi.queue_wait_p90_ms,
                    queue_wait_max_ms=kpi.queue_wait_max_ms,
                    dispatch_samples=kpi.dispatch_samples,
                    idle_gap_samples=kpi.idle_gap_samples,
                    queue_wait_samples=kpi.queue_wait_samples,
                )
                for kpi in self._run_stats.generation_kpis
            ],
        )

    def run(self) -> Population:
        """Execute the full GA loop.

        Returns:
            The final population after all generations.
        """
        self._channel = grpc.insecure_channel(self.config.scheduler_address)
        self._stub = epoch_pb2_grpc.SchedulerControlStub(self._channel)
        jobs_planned = self.config.ga_config.population_size * self.config.ga_config.num_generations
        self._run_stats = RunStats(
            jobs_planned=jobs_planned,
            generations_planned=self.config.ga_config.num_generations,
        )
        self._run_start_monotonic = time.monotonic()

        try:
            population = self._engine.initialize()
            logger.info(
                "GA started: pop_size=%d generations=%d",
                self.config.ga_config.population_size,
                self.config.ga_config.num_generations,
            )

            for gen in range(self.config.ga_config.num_generations):
                if self._should_stop_before_generation(gen):
                    break

                gen_start = time.monotonic()
                logger.info("Generation %d started", gen)

                # Submit generation to scheduler
                submitted_jobs = self._submit_generation(gen, population)
                self._run_stats.jobs_submitted += submitted_jobs

                # Poll for results
                results, scheduler_wall_clock_ms, runtime_metrics = self._wait_for_results(gen)
                self._record_result_statuses(results)
                gen_kpi = self._compute_generation_kpi(
                    gen, results, scheduler_wall_clock_ms, runtime_metrics
                )
                self._run_stats.generation_kpis.append(gen_kpi)
                self._run_stats.generations_completed += 1

                # Assign fitness
                self._assign_fitness(population, results)

                gen_elapsed_ms = int((time.monotonic() - gen_start) * 1000)
                logger.info(
                    "Gen %d complete: best=%.4f avg=%.4f worst=%.4f time=%dms",
                    gen,
                    population.best_fitness,
                    population.avg_fitness,
                    population.worst_fitness,
                    gen_elapsed_ms,
                )

                # Record metrics
                if self.metrics_store is not None:
                    self.metrics_store.record_generation(
                        generation=gen,
                        best_fitness=population.best_fitness,
                        avg_fitness=population.avg_fitness,
                        worst_fitness=population.worst_fitness,
                        wall_clock_ms=gen_elapsed_ms,
                    )

                if (
                    self.config.expected_workers > 0
                    and gen_kpi.active_workers < self.config.expected_workers
                ):
                    logger.error(
                        "Stopping early after generation %d: observed active_workers=%d "
                        "below expected_workers=%d",
                        gen,
                        gen_kpi.active_workers,
                        self.config.expected_workers,
                    )
                    self._mark_stopped("insufficient_workers")
                    break

                # Evolve (except after the last generation)
                if gen < self.config.ga_config.num_generations - 1:
                    population = self._engine.evolve(population)

            if not self._run_stats.stopped_early:
                self._run_stats.stop_reason = "completed"

            logger.info(
                "GA complete: final best=%.4f stop_reason=%s generations_completed=%d/%d",
                population.best_fitness,
                self._run_stats.stop_reason,
                self._run_stats.generations_completed,
                self._run_stats.generations_planned,
            )
            return population

        finally:
            if self._channel:
                self._channel.close()

    def _submit_generation(self, generation_id: int, population: Population) -> int:
        """Submit all individuals in the population as jobs."""
        configs = [ind.to_protobuf() for ind in population.individuals]
        request = epoch_pb2.SubmitGenerationRequest(
            generation_id=generation_id,
            configs=configs,
        )
        try:
            response = self._stub.SubmitGeneration(request, timeout=self.config.rpc_timeout_s)
        except grpc.RpcError as e:
            logger.error(
                "SubmitGeneration failed for generation %d: code=%s details=%s",
                generation_id,
                e.code(),
                e.details(),
            )
            raise
        logger.info("Submitted %d jobs for generation %d", response.num_jobs, generation_id)
        return int(response.num_jobs)

    def _wait_for_results(
        self, generation_id: int
    ) -> tuple[list[epoch_pb2.TrainingResult], int, epoch_pb2.GenerationRuntimeMetrics]:
        """Poll until all results for the generation are available."""
        start = time.monotonic()
        next_progress_log = start + self.config.progress_log_interval_s
        while True:
            request = epoch_pb2.GetResultsRequest(generation_id=generation_id)
            try:
                response = self._stub.GetGenerationResults(
                    request, timeout=self.config.rpc_timeout_s
                )
            except grpc.RpcError as e:
                logger.error(
                    "GetGenerationResults RPC failed for generation %d: code=%s details=%s",
                    generation_id,
                    e.code(),
                    e.details(),
                )
                raise

            if response.complete:
                logger.info(
                    "Generation %d results received (wall_clock=%dms)",
                    generation_id,
                    response.wall_clock_ms,
                )
                return (
                    list(response.results),
                    int(response.wall_clock_ms),
                    response.runtime_metrics,
                )

            now = time.monotonic()
            elapsed_s = now - start
            if self.config.generation_timeout_s > 0 and elapsed_s > self.config.generation_timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for generation {generation_id} results after "
                    f"{elapsed_s:.1f}s (scheduler={self.config.scheduler_address})"
                )

            if now >= next_progress_log:
                logger.warning(
                    "Still waiting for generation %d results (elapsed=%.1fs, scheduler=%s)",
                    generation_id,
                    elapsed_s,
                    self.config.scheduler_address,
                )
                next_progress_log = now + self.config.progress_log_interval_s

            time.sleep(self.config.poll_interval)

    def _compute_generation_kpi(
        self,
        generation_id: int,
        results: list[epoch_pb2.TrainingResult],
        scheduler_wall_clock_ms: int,
        runtime_metrics: epoch_pb2.GenerationRuntimeMetrics | None,
    ) -> GenerationKPI:
        jobs_count = len(results)
        completed = 0
        failed = 0
        timed_out = 0
        workers: set[str] = set()
        train_times: list[int] = []

        for result in results:
            if result.worker_id:
                workers.add(result.worker_id)
            train_times.append(max(0, int(result.training_time_ms)))
            if result.status == epoch_pb2.JOB_STATUS_COMPLETED:
                completed += 1
            elif result.status == epoch_pb2.JOB_STATUS_TIMEOUT:
                timed_out += 1
            elif result.status == epoch_pb2.JOB_STATUS_FAILED:
                failed += 1

        wall_clock_ms = max(0, int(scheduler_wall_clock_ms))
        train_total_ms = sum(train_times)
        active_workers = len(workers)
        train_p50_ms = self._percentile_int(train_times, 0.50)
        train_p90_ms = self._percentile_int(train_times, 0.90)
        ideal_wall_ms = (
            (train_total_ms / max(active_workers, 1)) if jobs_count > 0 else 0.0
        )
        queue_overhead_ms = max(float(wall_clock_ms) - ideal_wall_ms, 0.0)
        queue_overhead_pct = (
            (queue_overhead_ms / wall_clock_ms) if wall_clock_ms > 0 else 0.0
        )
        gen_jobs_per_sec = (
            (jobs_count / (wall_clock_ms / 1000.0)) if wall_clock_ms > 0 else 0.0
        )

        dispatch_latency_p50_ms = 0.0
        dispatch_latency_p90_ms = 0.0
        dispatch_latency_max_ms = 0.0
        worker_idle_gap_p50_ms = 0.0
        worker_idle_gap_p90_ms = 0.0
        worker_idle_gap_max_ms = 0.0
        queue_wait_p50_ms = 0.0
        queue_wait_p90_ms = 0.0
        queue_wait_max_ms = 0.0
        dispatch_samples = 0
        idle_gap_samples = 0
        queue_wait_samples = 0
        if runtime_metrics is not None:
            dispatch_latency_p50_ms = float(runtime_metrics.dispatch_latency_p50_ms)
            dispatch_latency_p90_ms = float(runtime_metrics.dispatch_latency_p90_ms)
            dispatch_latency_max_ms = float(runtime_metrics.dispatch_latency_max_ms)
            worker_idle_gap_p50_ms = float(runtime_metrics.worker_idle_gap_p50_ms)
            worker_idle_gap_p90_ms = float(runtime_metrics.worker_idle_gap_p90_ms)
            worker_idle_gap_max_ms = float(runtime_metrics.worker_idle_gap_max_ms)
            queue_wait_p50_ms = float(runtime_metrics.queue_wait_p50_ms)
            queue_wait_p90_ms = float(runtime_metrics.queue_wait_p90_ms)
            queue_wait_max_ms = float(runtime_metrics.queue_wait_max_ms)
            dispatch_samples = int(runtime_metrics.dispatch_samples)
            idle_gap_samples = int(runtime_metrics.idle_gap_samples)
            queue_wait_samples = int(runtime_metrics.queue_wait_samples)

        return GenerationKPI(
            generation=generation_id,
            jobs_count=jobs_count,
            completed=completed,
            failed=failed,
            timed_out=timed_out,
            active_workers=active_workers,
            wall_clock_ms=wall_clock_ms,
            train_total_ms=train_total_ms,
            train_p50_ms=train_p50_ms,
            train_p90_ms=train_p90_ms,
            ideal_wall_ms=ideal_wall_ms,
            queue_overhead_ms=queue_overhead_ms,
            queue_overhead_pct=queue_overhead_pct,
            gen_jobs_per_sec=gen_jobs_per_sec,
            dispatch_latency_p50_ms=dispatch_latency_p50_ms,
            dispatch_latency_p90_ms=dispatch_latency_p90_ms,
            dispatch_latency_max_ms=dispatch_latency_max_ms,
            worker_idle_gap_p50_ms=worker_idle_gap_p50_ms,
            worker_idle_gap_p90_ms=worker_idle_gap_p90_ms,
            worker_idle_gap_max_ms=worker_idle_gap_max_ms,
            queue_wait_p50_ms=queue_wait_p50_ms,
            queue_wait_p90_ms=queue_wait_p90_ms,
            queue_wait_max_ms=queue_wait_max_ms,
            dispatch_samples=dispatch_samples,
            idle_gap_samples=idle_gap_samples,
            queue_wait_samples=queue_wait_samples,
        )

    def _assign_fitness(
        self, population: Population, results: list[epoch_pb2.TrainingResult]
    ) -> None:
        """Match training results to individuals and assign fitness values."""
        # Results come back in the same order as configs were submitted
        for i, result in enumerate(results):
            if i < len(population.individuals):
                if result.status == epoch_pb2.JOB_STATUS_COMPLETED:
                    population.individuals[i].fitness = result.validation_accuracy
                else:
                    population.individuals[i].fitness = 0.0

    def _record_result_statuses(self, results: list[epoch_pb2.TrainingResult]) -> None:
        """Aggregate status counts for reporting/throughput analysis."""
        for result in results:
            self._run_stats.total_results += 1
            if result.status == epoch_pb2.JOB_STATUS_COMPLETED:
                self._run_stats.completed += 1
            elif result.status == epoch_pb2.JOB_STATUS_TIMEOUT:
                self._run_stats.timed_out += 1
            elif result.status == epoch_pb2.JOB_STATUS_FAILED:
                self._run_stats.failed += 1
            else:
                self._run_stats.unknown += 1

    def _should_stop_before_generation(self, generation_id: int) -> bool:
        if self.config.max_wall_clock_s <= 0:
            return False
        if generation_id <= 0:
            return False

        estimated_next_gen_s = self._estimate_next_generation_seconds()
        if estimated_next_gen_s <= 0:
            return False

        elapsed_s = time.monotonic() - self._run_start_monotonic
        projected_s = elapsed_s + estimated_next_gen_s
        if projected_s <= self.config.max_wall_clock_s:
            return False

        logger.warning(
            "Stopping before generation %d due to wall-clock guard: elapsed=%.1fs "
            "estimated_next_gen=%.1fs projected=%.1fs limit=%.1fs",
            generation_id,
            elapsed_s,
            estimated_next_gen_s,
            projected_s,
            self.config.max_wall_clock_s,
        )
        self._mark_stopped("max_wall_clock_reached")
        return True

    def _estimate_next_generation_seconds(self) -> float:
        wall_clock_s = [
            kpi.wall_clock_ms / 1000.0
            for kpi in self._run_stats.generation_kpis
            if kpi.wall_clock_ms > 0
        ]
        if not wall_clock_s:
            return 0.0
        wall_clock_s.sort()
        return float(wall_clock_s[len(wall_clock_s) // 2])

    def _mark_stopped(self, reason: str) -> None:
        self._run_stats.stopped_early = True
        if self._run_stats.stop_reason == "completed":
            self._run_stats.stop_reason = reason

    @staticmethod
    def _percentile_int(values: list[int], percentile: float) -> int:
        if not values:
            return 0
        sorted_values = sorted(values)
        index = min(len(sorted_values) - 1, int(len(sorted_values) * percentile))
        return int(sorted_values[index])
