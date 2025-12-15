import asyncio
import httpx
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from jsonpath_ng import parse as jsonpath_parse
import statistics
from tqdm.asyncio import tqdm

from models import (
    LatencyMetrics,
    RequestMetrics,
    WorkerMetrics,
    BenchmarkResult,
    MetricsType,
)
from mongo_client import mongo_db

logger = logging.getLogger(__name__)


def get_percentile(sorted_list: List[float], percentile: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * percentile)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


@dataclass
class SystemAPIConfig:
    endpoint: str
    auth_token: str
    json_path: str
    org_id: str
    benchmark_run_id: str
    max_load: int
    min_load: int
    dataset: List[Dict[str, Any]]
    ramp_duration_seconds: int = 60
    enable_ramping: bool = True
    timeout: int = 30
    headers: Optional[Dict[str, str]] = None
    max_retries: int = 3
    retry_backoff: float = 1.0
    store_results: bool = True
    output_file: Optional[str] = None
    method: str = "POST"


class SystemAPIBenchmark:
    def __init__(self, config: SystemAPIConfig):
        self.config = config
        self.benchmark_id = str(uuid.uuid4())
        self.db = mongo_db.get_database() if config.store_results else None
        self.request_times: Dict[int, List[float]] = {}
        self.request_details: Dict[int, List[Dict[str, Any]]] = {}
        self.request_metrics: Dict[int, List[RequestMetrics]] = {}
        self._current_target_workers: int = self.config.min_load
        self._completed = 0
        self._total = len(config.dataset)
        self._pbar: Optional[tqdm] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        if self.config.headers:
            headers.update(self.config.headers)
        return headers

    def _extract_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            jsonpath_expr = jsonpath_parse(self.config.json_path)
            matches = jsonpath_expr.find(data)
            if matches:
                return matches[0].value
            return data
        except Exception as e:
            logger.warning(f"JSONPath extraction failed: {e}")
            return data

    async def _single_request(
        self,
        client: httpx.AsyncClient,
        record_id: str,
        payload: Dict[str, Any]
    ) -> RequestMetrics:
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        start_ns = time.perf_counter()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                if self.config.method.upper() == "GET":
                    response = await client.get(
                        self.config.endpoint,
                        params=payload,
                        headers=self._get_headers(),
                    )
                else:
                    response = await client.request(
                        self.config.method.upper(),
                        self.config.endpoint,
                        json=payload,
                        headers=self._get_headers(),
                    )

                end_ns = time.perf_counter()
                end_time = datetime.now()
                latency_ms = (end_ns - start_ns) * 1000

                success = 200 <= response.status_code < 300
                self._completed += 1
                if self._pbar:
                    self._pbar.update(1)

                return RequestMetrics(
                    request_id=request_id,
                    record_id=record_id,
                    latency_ms=latency_ms,
                    start_time=start_time,
                    end_time=end_time,
                    success=success,
                    status_code=response.status_code,
                    error_message=None if success else response.text[:200],
                )

            except httpx.TimeoutException:
                last_error = "Request timeout"
            except httpx.ConnectError as e:
                last_error = f"Connection error: {str(e)[:100]}"
            except Exception as e:
                last_error = str(e)[:100]

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_backoff * (2 ** attempt))

        end_ns = time.perf_counter()
        end_time = datetime.now()
        self._completed += 1
        if self._pbar:
            self._pbar.update(1)

        return RequestMetrics(
            request_id=request_id,
            record_id=record_id,
            latency_ms=(end_ns - start_ns) * 1000,
            start_time=start_time,
            end_time=end_time,
            success=False,
            error_message=f"Max retries exceeded: {last_error}",
        )

    async def _worker_task(
        self,
        worker_id: int,
        worker_count: int,
        client: httpx.AsyncClient,
        task_queue: asyncio.Queue,
    ):
        if worker_count not in self.request_times:
            self.request_times[worker_count] = []
            self.request_details[worker_count] = []

        while True:
            try:
                record_id, payload = await asyncio.wait_for(task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                break

            metrics = await self._single_request(client, record_id, payload)
            self.request_times[worker_count].append(metrics.latency_ms)
            self.request_details[worker_count].append(metrics.to_dict())
            task_queue.task_done()

    async def _run_benchmark_with_workers(self, worker_count: int) -> WorkerMetrics:
        logger.info(f"Starting benchmark with {worker_count} workers...")

        task_queue: asyncio.Queue = asyncio.Queue()
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            payload = self._extract_payload(record)
            await task_queue.put((record_id, payload))

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            workers = [self._worker_task(i, worker_count, client, task_queue) for i in range(worker_count)]
            await asyncio.gather(*workers)

        latencies = self.request_times.get(worker_count, [])
        if not latencies:
            raise ValueError(f"No completed requests for worker count {worker_count}")

        return self._compute_worker_metrics(worker_count, latencies)

    def _compute_worker_metrics(self, worker_count: int, latencies: List[float]) -> WorkerMetrics:
        latencies_sorted = sorted(latencies)
        total_requests = len(latencies)
        failed = len([r for r in self.request_details[worker_count] if not r['success']])
        successful = total_requests - failed

        latency_metrics = LatencyMetrics(
            p50=get_percentile(latencies_sorted, 0.50),
            p75=get_percentile(latencies_sorted, 0.75),
            p90=get_percentile(latencies_sorted, 0.90),
            p99=get_percentile(latencies_sorted, 0.99),
        )

        details = self.request_details[worker_count]
        max_end = max(r['end_time'] for r in details)
        min_start = min(r['start_time'] for r in details)
        elapsed = (max_end - min_start).total_seconds()
        throughput = total_requests / elapsed if elapsed > 0 else 0

        return WorkerMetrics(
            worker_count=worker_count,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            latency_metrics=latency_metrics,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_latency_ms=statistics.mean(latencies),
            total_time_ms=sum(latencies),
            throughput_rps=throughput,
            individual_requests=details,
        )

    async def _run_benchmark_with_time_based_ramping(self) -> List[WorkerMetrics]:
        logger.info(f"Starting ramp {self.config.min_load} -> {self.config.max_load} over {self.config.ramp_duration_seconds}s")

        mean_load = (self.config.min_load + self.config.max_load) / 2
        ramp_duration = self.config.ramp_duration_seconds
        phase_1_duration = ramp_duration / 2

        task_queue = asyncio.Queue()
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            payload = self._extract_payload(record)
            await task_queue.put((record_id, payload))

        worker_semaphore = asyncio.Semaphore(self.config.min_load)
        benchmark_start = time.perf_counter()
        active_permits = self.config.min_load

        def _get_target_workers(elapsed: float) -> float:
            if elapsed < phase_1_duration:
                return self.config.min_load + (elapsed / phase_1_duration) * (self.config.max_load - self.config.min_load)
            else:
                progress = (elapsed - phase_1_duration) / phase_1_duration
                return self.config.max_load - progress * (self.config.max_load - mean_load)

        async def _worker_task_with_semaphore(worker_id: int, client: httpx.AsyncClient):
            while True:
                try:
                    record_id, payload = await asyncio.wait_for(task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break

                async with worker_semaphore:
                    metrics = await self._single_request(client, record_id, payload)
                    bucket = int(self._current_target_workers)
                    self.request_metrics.setdefault(bucket, []).append(metrics)
                    detail = metrics.to_dict()
                    detail['target_worker_count'] = bucket
                    self.request_details.setdefault(bucket, []).append(detail)
                    task_queue.task_done()

        async def _ramp_manager():
            nonlocal active_permits
            phase_2_end = benchmark_start + ramp_duration
            last_count = self.config.min_load

            try:
                while time.perf_counter() < phase_2_end and not task_queue.empty():
                    elapsed = time.perf_counter() - benchmark_start
                    target = _get_target_workers(elapsed)
                    target = max(self.config.min_load, min(target, self.config.max_load))
                    self._current_target_workers = int(round(target))

                    diff = int(target) - active_permits
                    if diff > 0:
                        for _ in range(diff):
                            worker_semaphore.release()
                        active_permits += diff
                    elif diff < 0:
                        for _ in range(-diff):
                            try:
                                await asyncio.wait_for(worker_semaphore.acquire(), timeout=0.1)
                                active_permits -= 1
                            except asyncio.TimeoutError:
                                break

                    if int(target) != int(last_count):
                        logger.info(f"Ramp: {int(target)} workers at {elapsed:.1f}s")
                    last_count = target
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                pass

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            workers = [asyncio.create_task(_worker_task_with_semaphore(i, client)) for i in range(self.config.max_load)]
            ramp_task = asyncio.create_task(_ramp_manager())

            try:
                await asyncio.wait_for(asyncio.gather(*workers), timeout=ramp_duration * 1.5)
            except asyncio.TimeoutError:
                logger.warning("Ramp benchmark timeout")
                for w in workers:
                    w.cancel()
            finally:
                ramp_task.cancel()

        all_worker_metrics = []
        for wc in sorted(self.request_metrics.keys()):
            metrics_list = self.request_metrics[wc]
            if not metrics_list:
                continue

            latencies = sorted([m.latency_ms for m in metrics_list])
            total = len(metrics_list)
            successful = sum(1 for m in metrics_list if m.success)

            latency_metrics = LatencyMetrics(
                p50=get_percentile(latencies, 0.50),
                p75=get_percentile(latencies, 0.75),
                p90=get_percentile(latencies, 0.90),
                p99=get_percentile(latencies, 0.99),
            )

            elapsed = (max(m.end_time for m in metrics_list) - min(m.start_time for m in metrics_list)).total_seconds()

            wm = WorkerMetrics(
                worker_count=wc,
                total_requests=total,
                successful_requests=successful,
                failed_requests=total - successful,
                latency_metrics=latency_metrics,
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                avg_latency_ms=statistics.mean(latencies),
                total_time_ms=sum(latencies),
                throughput_rps=total / elapsed if elapsed > 0 else 0,
                individual_requests=self.request_details.get(wc, []),
            )
            all_worker_metrics.append(wm)
            logger.info(f"Ramp {wc} workers: p50={latency_metrics.p50:.1f}ms, p99={latency_metrics.p99:.1f}ms, rps={wm.throughput_rps:.1f}")

        if not all_worker_metrics:
            raise ValueError("No completed requests during ramping")
        return all_worker_metrics

    async def run(self) -> BenchmarkResult:
        logger.info(f"Starting System API benchmark: {self.benchmark_id}")

        if not self.config.dataset:
            raise ValueError("Dataset is empty")

        start_time = datetime.now()
        all_worker_metrics = []

        self._pbar = tqdm(total=self._total, desc="Requests", unit="req")

        try:
            if self.config.enable_ramping:
                worker_metrics = await self._run_benchmark_with_time_based_ramping()
                for wm in worker_metrics:
                    all_worker_metrics.append(wm.to_dict())
            else:
                for worker_count in range(self.config.min_load, self.config.max_load + 1):
                    self.request_times.clear()
                    self.request_details.clear()
                    worker_metrics = await self._run_benchmark_with_workers(worker_count)
                    all_worker_metrics.append(worker_metrics.to_dict())
        finally:
            self._pbar.close()

        end_time = datetime.now()

        result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            benchmark_type=MetricsType.SYSTEM_API,
            timestamp=start_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            worker_metrics=all_worker_metrics,
            endpoint=self.config.endpoint,
            org_id=self.config.org_id,
            min_load=self.config.min_load,
            max_load=self.config.max_load,
            dataset_size=len(self.config.dataset),
        )

        if self.config.store_results and self.db:
            await self._store_result(result)

        if self.config.output_file:
            self._save_to_file(result)

        logger.info(f"System API benchmark completed: {self.benchmark_id}")
        return result

    async def _store_result(self, result: BenchmarkResult):
        try:
            collection = self.db['benchmark_results']
            await collection.insert_one(result.to_dict())
            logger.info(f"Stored result in MongoDB: {result.benchmark_id}")
        except Exception as e:
            logger.error(f"Failed to store result: {e}")

    def _save_to_file(self, result: BenchmarkResult):
        import json
        with open(self.config.output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved result to {self.config.output_file}")


async def run_system_benchmark(config: SystemAPIConfig) -> BenchmarkResult:
    benchmark = SystemAPIBenchmark(config)
    return await benchmark.run()
