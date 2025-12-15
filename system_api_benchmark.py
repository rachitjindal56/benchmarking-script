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

from models import (
    LatencyMetrics,
    RequestMetrics,
    WorkerMetrics,
    BenchmarkResult,
    MetricsType,
)
from mongo_client import mongo_db

logger = logging.getLogger(__name__)


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


class SystemAPIBenchmark:    
    def __init__(self, config: SystemAPIConfig):
        self.config = config
        self.benchmark_id = str(uuid.uuid4())
        self.db = mongo_db.get_database()
        self.request_times: Dict[int, List[float]] = {}
        self.request_details: Dict[int, List[Dict[str, Any]]] = {}
        self.request_metrics: Dict[int, List[RequestMetrics]] = {}
        self._current_target_workers: int = self.config.min_load
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {self.config.auth_token}"} if self.config.auth_token else {})
        }
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
            logger.warning(f"Failed to extract payload with JSONPath: {e}")
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
        
        try:
            response = await client.post(
                self.config.endpoint,
                json=payload,
                headers=self._get_headers(),
                timeout=self.config.timeout,
            )
            
            end_ns = time.perf_counter()
            end_time = datetime.now()
            latency_ms = (end_ns - start_ns) * 1000
            
            success = 200 <= response.status_code < 300
            
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
            
        except asyncio.TimeoutError:
            end_ns = time.perf_counter()
            end_time = datetime.now()
            latency_ms = (end_ns - start_ns) * 1000
            
            return RequestMetrics(
                request_id=request_id,
                record_id=record_id,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message="Request timeout",
            )
            
        except Exception as e:
            end_ns = time.perf_counter()
            end_time = datetime.now()
            latency_ms = (end_ns - start_ns) * 1000
            
            return RequestMetrics(
                request_id=request_id,
                record_id=record_id,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error_message=str(e)[:200],
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
                record_id, payload = await asyncio.wait_for(
                    task_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                break
            
            metrics = await self._single_request(client, record_id, payload)
            self.request_times[worker_count].append(metrics.latency_ms)
            self.request_details[worker_count].append(metrics.to_dict())
            task_queue.task_done()
    
    async def _run_benchmark_with_workers(
        self,
        worker_count: int,
    ) -> WorkerMetrics:
        logger.info(f"Starting benchmark with {worker_count} workers...")
        
        task_queue: asyncio.Queue = asyncio.Queue()
        
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            payload = self._extract_payload(record)
            await task_queue.put((record_id, payload))
        
        async with httpx.AsyncClient() as client:
            workers = [
                self._worker_task(i, worker_count, client, task_queue)
                for i in range(worker_count)
            ]
            
            await asyncio.gather(*workers)
        
        latencies = self.request_times.get(worker_count, [])
        if not latencies:
            raise ValueError(f"No successful requests for worker count {worker_count}")
        
        latencies_sorted = sorted(latencies)
        total_requests = len(latencies_sorted)
        successful_requests = total_requests
        failed_requests = len([
            r for r in self.request_details[worker_count]
            if not r['success']
        ])
        successful_requests = total_requests - failed_requests
        
        latency_metrics = LatencyMetrics(
            p50=statistics.median(latencies_sorted),
            p75=latencies_sorted[int(len(latencies_sorted) * 0.75)],
            p90=latencies_sorted[int(len(latencies_sorted) * 0.90)],
            p99=latencies_sorted[int(len(latencies_sorted) * 0.99)],
        )
        
        total_time_ms = sum(latencies)
        avg_latency_ms = statistics.mean(latencies)
        min_latency_ms = min(latencies)
        max_latency_ms = max(latencies)
        
        max_end_time = max(
            r['end_time'] for r in self.request_details[worker_count]
        )
        min_start_time = min(
            r['start_time'] for r in self.request_details[worker_count]
        )
        elapsed_seconds = (max_end_time - min_start_time).total_seconds()
        throughput_rps = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0
        
        metrics = WorkerMetrics(
            worker_count=worker_count,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            latency_metrics=latency_metrics,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            avg_latency_ms=avg_latency_ms,
            total_time_ms=total_time_ms,
            throughput_rps=throughput_rps,
            individual_requests=self.request_details[worker_count],
        )
        
        logger.info(
            f"Completed benchmark with {worker_count} workers: "
            f"avg_latency={avg_latency_ms:.2f}ms, "
            f"p50={latency_metrics.p50:.2f}ms, "
            f"p99={latency_metrics.p99:.2f}ms, "
            f"throughput={throughput_rps:.2f} req/s, "
            f"success_rate={successful_requests}/{total_requests}"
        )
        return metrics
    
    async def _run_benchmark_with_time_based_ramping(self) -> WorkerMetrics:
        logger.info(
            f"Starting smooth ramp from {self.config.min_load} to {self.config.max_load} "
            f"to {(self.config.min_load + self.config.max_load) / 2:.1f} "
            f"over {self.config.ramp_duration_seconds}s"
        )
        
        mean_load = (self.config.min_load + self.config.max_load) / 2
        ramp_duration = self.config.ramp_duration_seconds
        
        task_queue = asyncio.Queue()
        
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            payload = self._extract_payload(record)
            await task_queue.put((record_id, payload))
        
        phase_1_duration = ramp_duration / 2
        phase_2_duration = ramp_duration / 2
        
        current_workers = []
        worker_semaphore = asyncio.Semaphore(self.config.min_load)
        benchmark_start = time.perf_counter()
        
        def _get_target_workers(elapsed_seconds: float) -> float:
            if elapsed_seconds < phase_1_duration:
                progress = elapsed_seconds / phase_1_duration
                return self.config.min_load + progress * (self.config.max_load - self.config.min_load)
            else:
                progress = (elapsed_seconds - phase_1_duration) / phase_2_duration
                return self.config.max_load - progress * (self.config.max_load - mean_load)
        
        async def _worker_task_with_semaphore(worker_id: int):
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                while True:
                    try:
                        record_id, payload = await asyncio.wait_for(
                            task_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        break
                    
                    async with worker_semaphore:
                        metrics = await self._single_request(client, record_id, payload)
                        current_worker_count = int(self._current_target_workers)
                        self.request_metrics.setdefault(current_worker_count, []).append(metrics)
                        detail = metrics.to_dict()
                        detail['target_worker_count'] = current_worker_count
                        self.request_details.setdefault(current_worker_count, []).append(detail)
                        task_queue.task_done()
        
        async def _ramp_manager():
            phase_2_end = benchmark_start + ramp_duration
            last_worker_count = self.config.min_load
            
            try:
                while time.perf_counter() < phase_2_end and not task_queue.empty():
                    current_time = time.perf_counter()
                    elapsed = current_time - benchmark_start
                    
                    target_workers = _get_target_workers(elapsed)
                    target_workers = max(self.config.min_load, min(target_workers, self.config.max_load))

                    self._current_target_workers = int(round(target_workers))
                    current_permits = worker_semaphore._value
                    if target_workers > current_permits:
                        permits_to_add = int(target_workers - current_permits)
                        for _ in range(permits_to_add):
                            worker_semaphore.release()
                        if int(target_workers) != int(last_worker_count):
                            logger.info(f"Ramp: {int(target_workers)} workers at {elapsed:.2f}s")
                    elif target_workers < current_permits:
                        permits_to_remove = int(current_permits - target_workers)
                        for _ in range(permits_to_remove):
                            try:
                                await asyncio.wait_for(worker_semaphore.acquire(), timeout=0.1)
                            except asyncio.TimeoutError:
                                pass
                        if int(target_workers) != int(last_worker_count):
                            logger.info(f"Ramp: {int(target_workers)} workers at {elapsed:.2f}s")
                    
                    last_worker_count = target_workers
                    await asyncio.sleep(0.2)
            
            except asyncio.CancelledError:
                pass
        
        for i in range(self.config.max_load):
            worker = asyncio.create_task(_worker_task_with_semaphore(i))
            current_workers.append(worker)
        
        ramp_task = asyncio.create_task(_ramp_manager())
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*current_workers),
                timeout=ramp_duration * 1.5
            )
        except asyncio.TimeoutError:
            logger.warning("Ramp benchmark timeout")
            for worker in current_workers:
                worker.cancel()
        finally:
            ramp_task.cancel()
        
        all_metrics = []
        all_worker_metrics = []
        
        for worker_count in sorted(self.request_metrics.keys()):
            metrics_list = self.request_metrics[worker_count]
            
            if not metrics_list:
                continue
            
            latencies = [m.latency_ms for m in metrics_list]
            latencies_sorted = sorted(latencies)
            total_requests = len(metrics_list)
            successful_requests = sum(1 for m in metrics_list if m.success)
            failed_requests = total_requests - successful_requests
            
            latency_metrics = LatencyMetrics(
                p50=statistics.median(latencies_sorted),
                p75=latencies_sorted[int(len(latencies_sorted) * 0.75)],
                p90=latencies_sorted[int(len(latencies_sorted) * 0.90)],
                p99=latencies_sorted[int(len(latencies_sorted) * 0.99)],
            )
            
            avg_latency_ms = statistics.mean(latencies)
            min_latency_ms = min(latencies)
            max_latency_ms = max(latencies)
            
            max_end_time = max(m.end_time for m in metrics_list)
            min_start_time = min(m.start_time for m in metrics_list)
            elapsed_seconds = (max_end_time - min_start_time).total_seconds()
            throughput_rps = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0
            
            worker_metric = WorkerMetrics(
                worker_count=worker_count,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                latency_metrics=latency_metrics,
                min_latency_ms=min_latency_ms,
                max_latency_ms=max_latency_ms,
                avg_latency_ms=avg_latency_ms,
                total_time_ms=sum(latencies),
                throughput_rps=throughput_rps,
                individual_requests=self.request_details[worker_count],
            )
            all_worker_metrics.append(worker_metric)
            all_metrics.extend(metrics_list)
            
            logger.info(
                f"Ramp: {worker_count} workers - "
                f"avg_latency={avg_latency_ms:.2f}ms, "
                f"p50={latency_metrics.p50:.2f}ms, "
                f"p99={latency_metrics.p99:.2f}ms, "
                f"throughput={throughput_rps:.2f} req/s, "
                f"requests={total_requests}, "
                f"success_rate={successful_requests}/{total_requests}"
            )
        
        if not all_worker_metrics:
            raise ValueError("No completed requests during ramping")

        return all_worker_metrics
    
    async def run(self) -> BenchmarkResult:
        logger.info(f"Starting System API benchmark: {self.benchmark_id}")
        
        start_time = datetime.now()
        all_worker_metrics = []
        
        if self.config.enable_ramping:
            try:
                worker_metrics = await self._run_benchmark_with_time_based_ramping()
                if isinstance(worker_metrics, list):
                    for wm in worker_metrics:
                        all_worker_metrics.append(wm.to_dict())
                else:
                    all_worker_metrics.append(worker_metrics.to_dict())
            except Exception as e:
                logger.error(f"Ramping benchmark failed: {e}")
                raise
        else:
            for worker_count in range(self.config.min_load, self.config.max_load + 1):
                try:
                    worker_metrics = await self._run_benchmark_with_workers(worker_count)
                    all_worker_metrics.append(worker_metrics.to_dict())
                except Exception as e:
                    logger.error(f"Benchmark failed for {worker_count} workers: {e}")
                    raise
        
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        
        result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            benchmark_type=MetricsType.SYSTEM_API,
            timestamp=start_time,
            duration_seconds=duration_seconds,
            worker_metrics=all_worker_metrics,
            endpoint=self.config.endpoint,
            org_id=self.config.org_id,
            min_load=self.config.min_load,
            max_load=self.config.max_load,
            dataset_size=len(self.config.dataset),
        )
        await self._store_result(result)
        logger.info(f"System API benchmark completed: {self.benchmark_id}")
        return result
    
    async def _store_result(self, result: BenchmarkResult):
        try:
            collection = self.db['benchmark_results']
            document = result.to_dict()
            await collection.insert_one(document)
            logger.info(f"Stored benchmark result in MongoDB: {result.benchmark_id}")
        except Exception as e:
            logger.error(f"Failed to store benchmark result: {e}")
            raise


async def run_system_benchmark(config: SystemAPIConfig) -> BenchmarkResult:
    benchmark = SystemAPIBenchmark(config)
    return await benchmark.run()
