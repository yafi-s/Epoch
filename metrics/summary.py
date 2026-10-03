"""Comparable observed completion rates across historical summary schemas."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObservedThroughput:
    completed: int
    total_elapsed_s: float

    @property
    def jobs_per_sec(self):
        return self.completed / self.total_elapsed_s


def total_elapsed_seconds(summary):
    try:
        elapsed=float(summary['wall_clock_total_s'])
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
    return elapsed if not isinstance(summary['wall_clock_total_s'],bool) and math.isfinite(elapsed) and elapsed>0 else None


def observed_throughput(summary):
    """Use successful counts and the total run clock; never infer from submissions.

    Missing/invalid counts or clocks are unknown, not zero or legacy modeled rates.
    The optional metric clock may start at first dispatch and is not comparable
    with the total runtime used by cross-run reports and visualizations.
    """
    try:
        completed=float(summary['completed'])
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
    elapsed=total_elapsed_seconds(summary)
    if (isinstance(summary['completed'],bool) or not math.isfinite(completed)
            or completed<0 or not completed.is_integer()
            or elapsed is None):
        return None
    return ObservedThroughput(int(completed),elapsed)


def legacy_modeled_throughput(summary):
    key='legacy_modeled_jobs_per_sec' if summary.get('metric_schema_version')==2 else 'jobs_per_sec'
    try:
        value=float(summary[key])
        return value if math.isfinite(value) and value>=0 else None
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
