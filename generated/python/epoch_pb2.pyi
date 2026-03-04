from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Optimizer(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPTIMIZER_UNSPECIFIED: _ClassVar[Optimizer]
    OPTIMIZER_SGD: _ClassVar[Optimizer]
    OPTIMIZER_ADAM: _ClassVar[Optimizer]
    OPTIMIZER_RMSPROP: _ClassVar[Optimizer]
    OPTIMIZER_ADAMW: _ClassVar[Optimizer]

class Activation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTIVATION_UNSPECIFIED: _ClassVar[Activation]
    ACTIVATION_RELU: _ClassVar[Activation]
    ACTIVATION_ELU: _ClassVar[Activation]
    ACTIVATION_SELU: _ClassVar[Activation]
    ACTIVATION_TANH: _ClassVar[Activation]
    ACTIVATION_SWISH: _ClassVar[Activation]

class JobStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    JOB_STATUS_UNSPECIFIED: _ClassVar[JobStatus]
    JOB_STATUS_COMPLETED: _ClassVar[JobStatus]
    JOB_STATUS_FAILED: _ClassVar[JobStatus]
    JOB_STATUS_TIMEOUT: _ClassVar[JobStatus]
OPTIMIZER_UNSPECIFIED: Optimizer
OPTIMIZER_SGD: Optimizer
OPTIMIZER_ADAM: Optimizer
OPTIMIZER_RMSPROP: Optimizer
OPTIMIZER_ADAMW: Optimizer
ACTIVATION_UNSPECIFIED: Activation
ACTIVATION_RELU: Activation
ACTIVATION_ELU: Activation
ACTIVATION_SELU: Activation
ACTIVATION_TANH: Activation
ACTIVATION_SWISH: Activation
JOB_STATUS_UNSPECIFIED: JobStatus
JOB_STATUS_COMPLETED: JobStatus
JOB_STATUS_FAILED: JobStatus
JOB_STATUS_TIMEOUT: JobStatus

class HyperparamConfig(_message.Message):
    __slots__ = ("learning_rate", "batch_size", "optimizer", "conv_filters", "kernel_size", "dense_units", "dropout_rate", "activation", "epochs", "dataset")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_FIELD_NUMBER: _ClassVar[int]
    CONV_FILTERS_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    DENSE_UNITS_FIELD_NUMBER: _ClassVar[int]
    DROPOUT_RATE_FIELD_NUMBER: _ClassVar[int]
    ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    batch_size: int
    optimizer: Optimizer
    conv_filters: _containers.RepeatedScalarFieldContainer[int]
    kernel_size: int
    dense_units: _containers.RepeatedScalarFieldContainer[int]
    dropout_rate: float
    activation: Activation
    epochs: int
    dataset: str
    def __init__(self, learning_rate: _Optional[float] = ..., batch_size: _Optional[int] = ..., optimizer: _Optional[_Union[Optimizer, str]] = ..., conv_filters: _Optional[_Iterable[int]] = ..., kernel_size: _Optional[int] = ..., dense_units: _Optional[_Iterable[int]] = ..., dropout_rate: _Optional[float] = ..., activation: _Optional[_Union[Activation, str]] = ..., epochs: _Optional[int] = ..., dataset: _Optional[str] = ...) -> None: ...

class JobAssignment(_message.Message):
    __slots__ = ("job_id", "generation_id", "config")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    generation_id: int
    config: HyperparamConfig
    def __init__(self, job_id: _Optional[str] = ..., generation_id: _Optional[int] = ..., config: _Optional[_Union[HyperparamConfig, _Mapping]] = ...) -> None: ...

class TrainingResult(_message.Message):
    __slots__ = ("job_id", "generation_id", "worker_id", "validation_accuracy", "training_loss", "training_time_ms", "status", "error_message")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    TRAINING_LOSS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    generation_id: int
    worker_id: str
    validation_accuracy: float
    training_loss: float
    training_time_ms: int
    status: JobStatus
    error_message: str
    def __init__(self, job_id: _Optional[str] = ..., generation_id: _Optional[int] = ..., worker_id: _Optional[str] = ..., validation_accuracy: _Optional[float] = ..., training_loss: _Optional[float] = ..., training_time_ms: _Optional[int] = ..., status: _Optional[_Union[JobStatus, str]] = ..., error_message: _Optional[str] = ...) -> None: ...

class WorkerRegistration(_message.Message):
    __slots__ = ("worker_id", "num_gpus", "memory_mb", "auth_key")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_GPUS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    AUTH_KEY_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    num_gpus: int
    memory_mb: int
    auth_key: str
    def __init__(self, worker_id: _Optional[str] = ..., num_gpus: _Optional[int] = ..., memory_mb: _Optional[int] = ..., auth_key: _Optional[str] = ...) -> None: ...

class Heartbeat(_message.Message):
    __slots__ = ("worker_id", "timestamp_ms")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    timestamp_ms: int
    def __init__(self, worker_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class WorkerMessage(_message.Message):
    __slots__ = ("registration", "heartbeat", "result")
    REGISTRATION_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    registration: WorkerRegistration
    heartbeat: Heartbeat
    result: TrainingResult
    def __init__(self, registration: _Optional[_Union[WorkerRegistration, _Mapping]] = ..., heartbeat: _Optional[_Union[Heartbeat, _Mapping]] = ..., result: _Optional[_Union[TrainingResult, _Mapping]] = ...) -> None: ...

class SchedulerMessage(_message.Message):
    __slots__ = ("job_assignment", "shutdown")
    JOB_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    job_assignment: JobAssignment
    shutdown: bool
    def __init__(self, job_assignment: _Optional[_Union[JobAssignment, _Mapping]] = ..., shutdown: bool = ...) -> None: ...

class SubmitGenerationRequest(_message.Message):
    __slots__ = ("generation_id", "configs")
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIGS_FIELD_NUMBER: _ClassVar[int]
    generation_id: int
    configs: _containers.RepeatedCompositeFieldContainer[HyperparamConfig]
    def __init__(self, generation_id: _Optional[int] = ..., configs: _Optional[_Iterable[_Union[HyperparamConfig, _Mapping]]] = ...) -> None: ...

class SubmitGenerationResponse(_message.Message):
    __slots__ = ("accepted", "num_jobs")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    NUM_JOBS_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    num_jobs: int
    def __init__(self, accepted: bool = ..., num_jobs: _Optional[int] = ...) -> None: ...

class GetResultsRequest(_message.Message):
    __slots__ = ("generation_id",)
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    generation_id: int
    def __init__(self, generation_id: _Optional[int] = ...) -> None: ...

class GenerationRuntimeMetrics(_message.Message):
    __slots__ = ("dispatch_latency_p50_ms", "dispatch_latency_p90_ms", "dispatch_latency_max_ms", "worker_idle_gap_p50_ms", "worker_idle_gap_p90_ms", "worker_idle_gap_max_ms", "queue_wait_p50_ms", "queue_wait_p90_ms", "queue_wait_max_ms", "dispatch_samples", "idle_gap_samples", "queue_wait_samples")
    DISPATCH_LATENCY_P50_MS_FIELD_NUMBER: _ClassVar[int]
    DISPATCH_LATENCY_P90_MS_FIELD_NUMBER: _ClassVar[int]
    DISPATCH_LATENCY_MAX_MS_FIELD_NUMBER: _ClassVar[int]
    WORKER_IDLE_GAP_P50_MS_FIELD_NUMBER: _ClassVar[int]
    WORKER_IDLE_GAP_P90_MS_FIELD_NUMBER: _ClassVar[int]
    WORKER_IDLE_GAP_MAX_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_WAIT_P50_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_WAIT_P90_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_WAIT_MAX_MS_FIELD_NUMBER: _ClassVar[int]
    DISPATCH_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    IDLE_GAP_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    QUEUE_WAIT_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    dispatch_latency_p50_ms: float
    dispatch_latency_p90_ms: float
    dispatch_latency_max_ms: float
    worker_idle_gap_p50_ms: float
    worker_idle_gap_p90_ms: float
    worker_idle_gap_max_ms: float
    queue_wait_p50_ms: float
    queue_wait_p90_ms: float
    queue_wait_max_ms: float
    dispatch_samples: int
    idle_gap_samples: int
    queue_wait_samples: int
    def __init__(self, dispatch_latency_p50_ms: _Optional[float] = ..., dispatch_latency_p90_ms: _Optional[float] = ..., dispatch_latency_max_ms: _Optional[float] = ..., worker_idle_gap_p50_ms: _Optional[float] = ..., worker_idle_gap_p90_ms: _Optional[float] = ..., worker_idle_gap_max_ms: _Optional[float] = ..., queue_wait_p50_ms: _Optional[float] = ..., queue_wait_p90_ms: _Optional[float] = ..., queue_wait_max_ms: _Optional[float] = ..., dispatch_samples: _Optional[int] = ..., idle_gap_samples: _Optional[int] = ..., queue_wait_samples: _Optional[int] = ...) -> None: ...

class GetResultsResponse(_message.Message):
    __slots__ = ("complete", "results", "wall_clock_ms", "runtime_metrics")
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    WALL_CLOCK_MS_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_METRICS_FIELD_NUMBER: _ClassVar[int]
    complete: bool
    results: _containers.RepeatedCompositeFieldContainer[TrainingResult]
    wall_clock_ms: int
    runtime_metrics: GenerationRuntimeMetrics
    def __init__(self, complete: bool = ..., results: _Optional[_Iterable[_Union[TrainingResult, _Mapping]]] = ..., wall_clock_ms: _Optional[int] = ..., runtime_metrics: _Optional[_Union[GenerationRuntimeMetrics, _Mapping]] = ...) -> None: ...
