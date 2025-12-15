from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class MetricsType(Enum):
    SYSTEM_API = "system_api"
    LLM_API = "llm_api"


@dataclass
class LatencyMetrics:
    p50: float
    p75: float
    p90: float
    p99: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RequestMetrics:
    request_id: str
    record_id: str
    latency_ms: float
    start_time: datetime
    end_time: datetime
    success: bool
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'record_id': self.record_id,
            'latency_ms': self.latency_ms,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'success': self.success,
            'error_message': self.error_message,
            'status_code': self.status_code,
        }


@dataclass
class LLMRequestMetrics(RequestMetrics):
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    tokens_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'ttft_ms': self.ttft_ms,
            'tpot_ms': self.tpot_ms,
            'tokens_generated': self.tokens_generated,
        })
        return data


@dataclass
class WorkerMetrics:
    worker_count: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    latency_metrics: LatencyMetrics
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    total_time_ms: float
    throughput_rps: float
    individual_requests: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_count': self.worker_count,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'latency_metrics': self.latency_metrics.to_dict(),
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'avg_latency_ms': self.avg_latency_ms,
            'total_time_ms': self.total_time_ms,
            'throughput_rps': self.throughput_rps,
            'individual_requests': self.individual_requests,
        }


@dataclass
class LLMWorkerMetrics(WorkerMetrics):
    avg_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    total_tokens_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'avg_ttft_ms': self.avg_ttft_ms,
            'avg_tpot_ms': self.avg_tpot_ms,
            'total_tokens_generated': self.total_tokens_generated,
        })
        return data


@dataclass
class BenchmarkResult:
    benchmark_id: str
    benchmark_type: MetricsType
    timestamp: datetime
    duration_seconds: float
    worker_metrics: List[Dict[str, Any]]
    endpoint: Optional[str] = None
    org_id: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    min_load: int = 0
    max_load: int = 0
    dataset_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_id': self.benchmark_id,
            'benchmark_type': self.benchmark_type.value,
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
            'worker_metrics': self.worker_metrics,
            'endpoint': self.endpoint,
            'org_id': self.org_id,
            'provider': self.provider,
            'model_name': self.model_name,
            'min_load': self.min_load,
            'max_load': self.max_load,
            'dataset_size': self.dataset_size,
        }
