import asyncio
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from openai._exceptions import APIError

from models import (
    LatencyMetrics,
    LLMRequestMetrics,
    LLMWorkerMetrics,
    BenchmarkResult,
    MetricsType,
)
from mongo_client import mongo_db

logger = logging.getLogger(__name__)


class LLMProvider:    
    def __init__(
        self,
        model_name: str,
        auth_token: str,
        base_url: str,
        timeout: int = 60
    ):
        self.model_name = model_name
        self.auth_token = auth_token
        self.base_url = base_url
        self.timeout = timeout
        self.client = AsyncOpenAI(api_key=auth_token, base_url=base_url, timeout=timeout)
    
    async def make_request(
        self,
        prompt: str
    ) -> Tuple[bool, float, float, int, Optional[str]]:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):    
    def __init__(self, model_name: str, auth_token: str, timeout: int = 60):
        super().__init__(
            model_name,
            auth_token,
            "https://api.openai.com/v1",
            timeout
        )
    
    async def make_request(
        self,
        prompt: str
    ) -> Tuple[bool, float, float, int, Optional[str]]:
        request_start = time.perf_counter()
        ttft_time = None
        token_times = []
        tokens_generated = 0
        
        try:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                stream=True,
            )
            
            async for event in stream:
                current_time = time.perf_counter()
                if hasattr(event, 'choices') and len(event.choices) > 0:
                    choice = event.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        if choice.delta.content:
                            if ttft_time is None:
                                ttft_time = current_time - request_start
                            else:
                                token_times.append(current_time)
                            
                            tokens_generated += 1
            
            ttft_ms = ttft_time * 1000 if ttft_time else 0
            if len(token_times) > 1:
                token_intervals = [
                    (token_times[i] - token_times[i-1]) * 1000
                    for i in range(1, len(token_times))
                ]
                tpot_ms = statistics.mean(token_intervals) if token_intervals else 0
            else:
                tpot_ms = 0
            
            return True, ttft_ms, tpot_ms, tokens_generated, None
            
        except APITimeoutError:
            return False, 0.0, 0.0, 0, "Request timeout"
        except APIConnectionError as e:
            return False, 0.0, 0.0, 0, f"Connection error: {str(e)[:100]}"
        except APIError as e:
            return False, 0.0, 0.0, 0, f"API error: {str(e)[:100]}"
        except Exception as e:
            return False, 0.0, 0.0, 0, str(e)[:100]


class TogetherAIProvider(LLMProvider):    
    def __init__(self, model_name: str, auth_token: str, timeout: int = 60):
        super().__init__(
            model_name,
            auth_token,
            "https://api.together.xyz/v1",
            timeout
        )
    
    async def make_request(
        self,
        prompt: str
    ) -> Tuple[bool, float, float, int, Optional[str]]:
        request_start = time.perf_counter()
        ttft_time = None
        token_times = []
        tokens_generated = 0
        
        try:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                stream=True,
            )
            
            async for event in stream:
                current_time = time.perf_counter()
                
                if hasattr(event, 'choices') and len(event.choices) > 0:
                    choice = event.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        if choice.delta.content:
                            if ttft_time is None:
                                ttft_time = current_time - request_start
                            else:
                                token_times.append(current_time)
                            
                            tokens_generated += 1
            
            ttft_ms = ttft_time * 1000 if ttft_time else 0
            if len(token_times) > 1:
                token_intervals = [
                    (token_times[i] - token_times[i-1]) * 1000
                    for i in range(1, len(token_times))
                ]
                tpot_ms = statistics.mean(token_intervals) if token_intervals else 0
            else:
                tpot_ms = 0
            
            return True, ttft_ms, tpot_ms, tokens_generated, None
            
        except APITimeoutError:
            return False, 0.0, 0.0, 0, "Request timeout"
        except APIConnectionError as e:
            return False, 0.0, 0.0, 0, f"Connection error: {str(e)[:100]}"
        except APIError as e:
            return False, 0.0, 0.0, 0, f"API error: {str(e)[:100]}"
        except Exception as e:
            return False, 0.0, 0.0, 0, str(e)[:100]


class AnthropicProvider(LLMProvider):    
    def __init__(self, model_name: str, auth_token: str, timeout: int = 60):
        super().__init__(
            model_name,
            auth_token,
            "https://api.anthropic.com/v1",
            timeout
        )
    
    async def make_request(
        self,
        prompt: str
    ) -> Tuple[bool, float, float, int, Optional[str]]:
        request_start = time.perf_counter()
        ttft_time = None
        token_times = []
        tokens_generated = 0
        
        try:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                stream=True,
            )
            
            async for event in stream:
                current_time = time.perf_counter()
                
                if hasattr(event, 'choices') and len(event.choices) > 0:
                    choice = event.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        if choice.delta.content:
                            if ttft_time is None:
                                ttft_time = current_time - request_start
                            else:
                                token_times.append(current_time)
                            
                            tokens_generated += 1
            
            ttft_ms = ttft_time * 1000 if ttft_time else 0
            
            if len(token_times) > 1:
                token_intervals = [
                    (token_times[i] - token_times[i-1]) * 1000
                    for i in range(1, len(token_times))
                ]
                tpot_ms = statistics.mean(token_intervals) if token_intervals else 0
            else:
                tpot_ms = 0
            
            return True, ttft_ms, tpot_ms, tokens_generated, None
            
        except APITimeoutError:
            return False, 0.0, 0.0, 0, "Request timeout"
        except APIConnectionError as e:
            return False, 0.0, 0.0, 0, f"Connection error: {str(e)[:100]}"
        except APIError as e:
            return False, 0.0, 0.0, 0, f"API error: {str(e)[:100]}"
        except Exception as e:
            return False, 0.0, 0.0, 0, str(e)[:100]


def create_provider(
    provider_name: str,
    model_name: str,
    auth_token: str,
    timeout: int = 60
) -> LLMProvider:
    provider_map = {
        'openai': OpenAIProvider,
        'togetherai': TogetherAIProvider,
        'anthropic': AnthropicProvider,
    }
    
    provider_class = provider_map.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class(model_name, auth_token, timeout)


@dataclass
class LLMAPIConfig:
    provider: str
    model_name: str
    auth_token: str
    max_load: int
    min_load: int
    dataset: List[Dict[str, Any]]
    ramp_duration_seconds: int = 60
    enable_ramping: bool = True
    timeout: int = 60


class LLMAPIBenchmark:
    def __init__(self, config: LLMAPIConfig):
        self.config = config
        self.benchmark_id = str(uuid.uuid4())
        self.db = mongo_db.get_database()
        self.provider = create_provider(
            config.provider,
            config.model_name,
            config.auth_token,
            config.timeout
        )
        self.request_metrics: Dict[int, List[LLMRequestMetrics]] = {}
        self.request_details: Dict[int, List[Dict[str, Any]]] = {}
        self._current_target_workers: int = self.config.min_load
    
    async def _single_request(
        self,
        record_id: str,
        prompt: str,
    ) -> LLMRequestMetrics:
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        request_start_time = time.perf_counter()
        
        success, ttft_ms, tpot_ms, tokens_generated, error_msg = (
            await self.provider.make_request(prompt)
        )
        
        request_end_time = time.perf_counter()
        end_time = datetime.now()
        latency_ms = (request_end_time - request_start_time) * 1000
        
        return LLMRequestMetrics(
            request_id=request_id,
            record_id=record_id,
            latency_ms=latency_ms,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error_message=error_msg,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            tokens_generated=tokens_generated,
        )
    
    async def _worker_task(
        self,
        worker_id: int,
        worker_count: int,
        task_queue: asyncio.Queue,
    ):
        if worker_count not in self.request_metrics:
            self.request_metrics[worker_count] = []
            self.request_details[worker_count] = []
        
        while True:
            try:
                record_id, prompt = await asyncio.wait_for(
                    task_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                break
            
            metrics = await self._single_request(record_id, prompt)
            self.request_metrics[worker_count].append(metrics)
            self.request_details[worker_count].append(metrics.to_dict())
            task_queue.task_done()
    
    async def _run_benchmark_with_workers(
        self,
        worker_count: int,
    ) -> LLMWorkerMetrics:
        logger.info(f"Starting LLM benchmark with {worker_count} workers...")
        
        task_queue: asyncio.Queue = asyncio.Queue()
        
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            prompt = record.get('prompt', '')
            await task_queue.put((record_id, prompt))
        
        workers = [
            self._worker_task(i, worker_count, task_queue)
            for i in range(worker_count)
        ]
        
        await asyncio.gather(*workers)
        
        metrics_list = self.request_metrics.get(worker_count, [])
        if not metrics_list:
            raise ValueError(f"No completed requests for worker count {worker_count}")
        
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
        
        avg_ttft_ms = statistics.mean([m.ttft_ms for m in metrics_list if m.success]) if successful_requests > 0 else 0
        avg_tpot_ms = statistics.mean([m.tpot_ms for m in metrics_list if m.success and m.tokens_generated > 0]) if successful_requests > 0 else 0
        total_tokens_generated = sum(m.tokens_generated for m in metrics_list)
        
        total_time_ms = sum(latencies)
        avg_latency_ms = statistics.mean(latencies)
        min_latency_ms = min(latencies)
        max_latency_ms = max(latencies)
        
        max_end_time = max(m.end_time for m in metrics_list)
        min_start_time = min(m.start_time for m in metrics_list)
        elapsed_seconds = (max_end_time - min_start_time).total_seconds()
        throughput_rps = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0
        
        metrics = LLMWorkerMetrics(
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
            avg_ttft_ms=avg_ttft_ms,
            avg_tpot_ms=avg_tpot_ms,
            total_tokens_generated=total_tokens_generated,
            individual_requests=self.request_details[worker_count],
        )
        
        logger.info(
            f"Completed LLM benchmark with {worker_count} workers: "
            f"avg_latency={avg_latency_ms:.2f}ms, "
            f"p50={latency_metrics.p50:.2f}ms, "
            f"p99={latency_metrics.p99:.2f}ms, "
            f"throughput={throughput_rps:.2f} req/s, "
            f"avg TTFT: {avg_ttft_ms:.2f}ms, "
            f"avg TPOT: {avg_tpot_ms:.2f}ms, "
            f"success_rate={successful_requests}/{total_requests}"
        )
        return metrics
    
    async def _run_benchmark_with_time_based_ramping(self) -> LLMWorkerMetrics:
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
            prompt = record.get('prompt', '')
            await task_queue.put((record_id, prompt))
        
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
            if self._current_target_workers not in self.request_metrics:
                self.request_metrics[self._current_target_workers] = []
                self.request_details[self._current_target_workers] = []
            
            while True:
                try:
                    record_id, prompt = await asyncio.wait_for(
                        task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    break
                
                async with worker_semaphore:
                    metrics = await self._single_request(record_id, prompt)
                    current_bucket = int(self._current_target_workers)
                    self.request_metrics.setdefault(current_bucket, []).append(metrics)
                    detail = metrics.to_dict()
                    detail['target_worker_count'] = current_bucket
                    self.request_details.setdefault(current_bucket, []).append(detail)
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
        all_worker_metrics: List[LLMWorkerMetrics] = []

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

            avg_ttft_ms = statistics.mean([m.ttft_ms for m in metrics_list if m.success]) if successful_requests > 0 else 0
            avg_tpot_ms = statistics.mean([m.tpot_ms for m in metrics_list if m.success and m.tokens_generated > 0]) if successful_requests > 0 else 0
            total_tokens_generated = sum(m.tokens_generated for m in metrics_list)

            total_time_ms = sum(latencies)
            avg_latency_ms = statistics.mean(latencies)
            min_latency_ms = min(latencies)
            max_latency_ms = max(latencies)

            max_end_time = max(m.end_time for m in metrics_list)
            min_start_time = min(m.start_time for m in metrics_list)
            elapsed_seconds = (max_end_time - min_start_time).total_seconds()
            throughput_rps = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0

            worker_metric = LLMWorkerMetrics(
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
                avg_ttft_ms=avg_ttft_ms,
                avg_tpot_ms=avg_tpot_ms,
                total_tokens_generated=total_tokens_generated,
                individual_requests=[m.to_dict() for m in metrics_list],
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
        logger.info(f"Starting LLM API benchmark: {self.benchmark_id}")
        
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
            benchmark_type=MetricsType.LLM_API,
            timestamp=start_time,
            duration_seconds=duration_seconds,
            worker_metrics=all_worker_metrics,
            provider=self.config.provider,
            model_name=self.config.model_name,
            min_load=self.config.min_load,
            max_load=self.config.max_load,
            dataset_size=len(self.config.dataset),
        )
        
        await self._store_result(result)
        
        logger.info(f"LLM API benchmark completed: {self.benchmark_id}")
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


async def run_llm_benchmark(config: LLMAPIConfig) -> BenchmarkResult:
    benchmark = LLMAPIBenchmark(config)
    return await benchmark.run()
