"""
Enterprise Benchmarking Suite for System and LLM APIs.

This package provides comprehensive benchmarking capabilities for:
- System APIs: Test HTTP endpoints with various load levels
- LLM APIs: Benchmark OpenAI, Together AI, and Anthropic models

All results are persisted in MongoDB for analysis and comparison.
"""

from mongo_client import mongo_db, MongoDBSingleton
from models import (
    MetricsType,
    LatencyMetrics,
    RequestMetrics,
    LLMRequestMetrics,
    WorkerMetrics,
    LLMWorkerMetrics,
    BenchmarkResult,
)
from system_api_benchmark import (
    SystemAPIConfig,
    SystemAPIBenchmark,
    run_system_benchmark,
)
from llm_api_benchmark import (
    LLMAPIConfig,
    LLMAPIBenchmark,
    run_llm_benchmark,
)
from analyzer import BenchmarkAnalyzer

__all__ = [
    'mongo_db',
    'MongoDBSingleton',
    'MetricsType',
    'LatencyMetrics',
    'RequestMetrics',
    'LLMRequestMetrics',
    'WorkerMetrics',
    'LLMWorkerMetrics',
    'BenchmarkResult',
    'SystemAPIConfig',
    'SystemAPIBenchmark',
    'run_system_benchmark',
    'LLMAPIConfig',
    'LLMAPIBenchmark',
    'run_llm_benchmark',
    'BenchmarkAnalyzer',
]

__version__ = '1.0.0'
