import asyncio
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import statistics

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from openai._exceptions import APIError
import anthropic
from tqdm.asyncio import tqdm

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
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 1.0
    ):
        self.model_name = model_name
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    async def make_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        raise NotImplementedError

    async def _retry_request(self, func, *args, **kwargs) -> Tuple[bool, float, float, int, Optional[str]]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except (APITimeoutError, APIConnectionError) as e:
                last_error = str(e)[:100]
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff * (2 ** attempt))
            except Exception as e:
                return False, 0.0, 0.0, 0, str(e)[:100]
        return False, 0.0, 0.0, 0, f"Max retries exceeded: {last_error}"


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str, auth_token: str, timeout: int = 60, use_chat: bool = True, **kwargs):
        super().__init__(model_name, auth_token, timeout, **kwargs)
        self.use_chat = use_chat
        self.client = AsyncOpenAI(api_key=auth_token, timeout=timeout)

    async def make_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        return await self._retry_request(self._do_request, prompt)

    async def _do_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        request_start = time.perf_counter()
        ttft_time = None
        token_times = []
        tokens_generated = 0

        if self.use_chat:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for event in stream:
                current_time = time.perf_counter()
                if event.choices and event.choices[0].delta.content:
                    if ttft_time is None:
                        ttft_time = current_time - request_start
                    else:
                        token_times.append(current_time)
                    tokens_generated += 1
        else:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                stream=True,
            )
            async for event in stream:
                current_time = time.perf_counter()
                if event.choices and event.choices[0].text:
                    if ttft_time is None:
                        ttft_time = current_time - request_start
                    else:
                        token_times.append(current_time)
                    tokens_generated += 1

        ttft_ms = ttft_time * 1000 if ttft_time else 0
        tpot_ms = 0
        if len(token_times) > 1:
            intervals = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
            tpot_ms = statistics.mean(intervals) if intervals else 0

        return True, ttft_ms, tpot_ms, tokens_generated, None


class TogetherAIProvider(LLMProvider):
    def __init__(self, model_name: str, auth_token: str, timeout: int = 60, use_chat: bool = True, **kwargs):
        super().__init__(model_name, auth_token, timeout, **kwargs)
        self.use_chat = use_chat
        self.client = AsyncOpenAI(
            api_key=auth_token,
            base_url="https://api.together.xyz/v1",
            timeout=timeout
        )

    async def make_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        return await self._retry_request(self._do_request, prompt)

    async def _do_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        request_start = time.perf_counter()
        ttft_time = None
        token_times = []
        tokens_generated = 0

        if self.use_chat:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for event in stream:
                current_time = time.perf_counter()
                if event.choices and event.choices[0].delta.content:
                    if ttft_time is None:
                        ttft_time = current_time - request_start
                    else:
                        token_times.append(current_time)
                    tokens_generated += 1
        else:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                stream=True,
            )
            async for event in stream:
                current_time = time.perf_counter()
                if event.choices and event.choices[0].text:
                    if ttft_time is None:
                        ttft_time = current_time - request_start
                    else:
                        token_times.append(current_time)
                    tokens_generated += 1

        ttft_ms = ttft_time * 1000 if ttft_time else 0
        tpot_ms = 0
        if len(token_times) > 1:
            intervals = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
            tpot_ms = statistics.mean(intervals) if intervals else 0

        return True, ttft_ms, tpot_ms, tokens_generated, None


class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str, auth_token: str, timeout: int = 60, **kwargs):
        super().__init__(model_name, auth_token, timeout, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=auth_token, timeout=timeout)

    async def make_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        return await self._retry_request(self._do_request, prompt)

    async def _do_request(self, prompt: str) -> Tuple[bool, float, float, int, Optional[str]]:
        request_start = time.perf_counter()
        ttft_time = None
        token_times = []
        tokens_generated = 0

        try:
            async with self.client.messages.stream(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    current_time = time.perf_counter()
                    if text:
                        if ttft_time is None:
                            ttft_time = current_time - request_start
                        else:
                            token_times.append(current_time)
                        tokens_generated += len(text.split())

            ttft_ms = ttft_time * 1000 if ttft_time else 0
            tpot_ms = 0
            if len(token_times) > 1:
                intervals = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
                tpot_ms = statistics.mean(intervals) if intervals else 0

            return True, ttft_ms, tpot_ms, tokens_generated, None

        except anthropic.APITimeoutError:
            return False, 0.0, 0.0, 0, "Request timeout"
        except anthropic.APIConnectionError as e:
            return False, 0.0, 0.0, 0, f"Connection error: {str(e)[:100]}"
        except anthropic.APIError as e:
            return False, 0.0, 0.0, 0, f"API error: {str(e)[:100]}"


def create_provider(
    provider_name: str,
    model_name: str,
    auth_token: str,
    timeout: int = 60,
    use_chat: bool = True,
    max_retries: int = 3
) -> LLMProvider:
    providers = {
        'openai': lambda: OpenAIProvider(model_name, auth_token, timeout, use_chat, max_retries=max_retries),
        'togetherai': lambda: TogetherAIProvider(model_name, auth_token, timeout, use_chat, max_retries=max_retries),
        'anthropic': lambda: AnthropicProvider(model_name, auth_token, timeout, max_retries=max_retries),
    }
    factory = providers.get(provider_name.lower())
    if not factory:
        raise ValueError(f"Unknown provider: {provider_name}")
    return factory()


def get_percentile(sorted_list: List[float], percentile: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * percentile)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


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
    use_chat: bool = True
    max_retries: int = 3
    store_results: bool = True
    output_file: Optional[str] = None


class LLMAPIBenchmark:
    def __init__(self, config: LLMAPIConfig):
        self.config = config
        self.benchmark_id = str(uuid.uuid4())
        self.db = mongo_db.get_database() if config.store_results else None
        self.provider = create_provider(
            config.provider,
            config.model_name,
            config.auth_token,
            config.timeout,
            config.use_chat,
            config.max_retries
        )
        self.request_metrics: Dict[int, List[LLMRequestMetrics]] = {}
        self.request_details: Dict[int, List[Dict[str, Any]]] = {}
        self._current_target_workers: int = self.config.min_load
        self._completed = 0
        self._total = len(config.dataset)
        self._pbar: Optional[tqdm] = None

    async def _single_request(self, record_id: str, prompt: str) -> LLMRequestMetrics:
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        request_start_time = time.perf_counter()

        success, ttft_ms, tpot_ms, tokens_generated, error_msg = await self.provider.make_request(prompt)

        request_end_time = time.perf_counter()
        end_time = datetime.now()
        latency_ms = (request_end_time - request_start_time) * 1000

        self._completed += 1
        if self._pbar:
            self._pbar.update(1)

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

    async def _worker_task(self, worker_id: int, worker_count: int, task_queue: asyncio.Queue):
        if worker_count not in self.request_metrics:
            self.request_metrics[worker_count] = []
            self.request_details[worker_count] = []

        while True:
            try:
                record_id, prompt = await asyncio.wait_for(task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                break

            metrics = await self._single_request(record_id, prompt)
            self.request_metrics[worker_count].append(metrics)
            self.request_details[worker_count].append(metrics.to_dict())
            task_queue.task_done()

    async def _run_benchmark_with_workers(self, worker_count: int) -> LLMWorkerMetrics:
        logger.info(f"Starting LLM benchmark with {worker_count} workers...")

        task_queue: asyncio.Queue = asyncio.Queue()
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            prompt = record.get('prompt', '')
            await task_queue.put((record_id, prompt))

        workers = [self._worker_task(i, worker_count, task_queue) for i in range(worker_count)]
        await asyncio.gather(*workers)

        return self._compute_worker_metrics(worker_count)

    def _compute_worker_metrics(self, worker_count: int) -> LLMWorkerMetrics:
        metrics_list = self.request_metrics.get(worker_count, [])
        if not metrics_list:
            raise ValueError(f"No completed requests for worker count {worker_count}")

        latencies = [m.latency_ms for m in metrics_list]
        latencies_sorted = sorted(latencies)

        total_requests = len(metrics_list)
        successful_requests = sum(1 for m in metrics_list if m.success)
        failed_requests = total_requests - successful_requests

        latency_metrics = LatencyMetrics(
            p50=get_percentile(latencies_sorted, 0.50),
            p75=get_percentile(latencies_sorted, 0.75),
            p90=get_percentile(latencies_sorted, 0.90),
            p99=get_percentile(latencies_sorted, 0.99),
        )

        successful_metrics = [m for m in metrics_list if m.success]
        avg_ttft_ms = statistics.mean([m.ttft_ms for m in successful_metrics]) if successful_metrics else 0
        tokens_with_tpot = [m for m in successful_metrics if m.tokens_generated > 0]
        avg_tpot_ms = statistics.mean([m.tpot_ms for m in tokens_with_tpot]) if tokens_with_tpot else 0
        total_tokens_generated = sum(m.tokens_generated for m in metrics_list)

        max_end_time = max(m.end_time for m in metrics_list)
        min_start_time = min(m.start_time for m in metrics_list)
        elapsed_seconds = (max_end_time - min_start_time).total_seconds()
        throughput_rps = total_requests / elapsed_seconds if elapsed_seconds > 0 else 0

        return LLMWorkerMetrics(
            worker_count=worker_count,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            latency_metrics=latency_metrics,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            avg_latency_ms=statistics.mean(latencies),
            total_time_ms=sum(latencies),
            throughput_rps=throughput_rps,
            avg_ttft_ms=avg_ttft_ms,
            avg_tpot_ms=avg_tpot_ms,
            total_tokens_generated=total_tokens_generated,
            individual_requests=self.request_details.get(worker_count, []),
        )

    async def _run_benchmark_with_time_based_ramping(self) -> List[LLMWorkerMetrics]:
        logger.info(f"Starting ramp {self.config.min_load} -> {self.config.max_load} over {self.config.ramp_duration_seconds}s")

        mean_load = (self.config.min_load + self.config.max_load) / 2
        ramp_duration = self.config.ramp_duration_seconds
        phase_1_duration = ramp_duration / 2

        task_queue = asyncio.Queue()
        for record in self.config.dataset:
            record_id = record.get('id', str(uuid.uuid4()))
            prompt = record.get('prompt', '')
            await task_queue.put((record_id, prompt))

        worker_semaphore = asyncio.Semaphore(self.config.min_load)
        benchmark_start = time.perf_counter()
        active_permits = self.config.min_load

        def _get_target_workers(elapsed: float) -> float:
            if elapsed < phase_1_duration:
                return self.config.min_load + (elapsed / phase_1_duration) * (self.config.max_load - self.config.min_load)
            else:
                progress = (elapsed - phase_1_duration) / phase_1_duration
                return self.config.max_load - progress * (self.config.max_load - mean_load)

        async def _worker_task_with_semaphore(worker_id: int):
            while True:
                try:
                    record_id, prompt = await asyncio.wait_for(task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break

                async with worker_semaphore:
                    metrics = await self._single_request(record_id, prompt)
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

        workers = [asyncio.create_task(_worker_task_with_semaphore(i)) for i in range(self.config.max_load)]
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

            successful_list = [m for m in metrics_list if m.success]
            avg_ttft = statistics.mean([m.ttft_ms for m in successful_list]) if successful_list else 0
            tokens_list = [m for m in successful_list if m.tokens_generated > 0]
            avg_tpot = statistics.mean([m.tpot_ms for m in tokens_list]) if tokens_list else 0

            elapsed = (max(m.end_time for m in metrics_list) - min(m.start_time for m in metrics_list)).total_seconds()

            wm = LLMWorkerMetrics(
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
                avg_ttft_ms=avg_ttft,
                avg_tpot_ms=avg_tpot,
                total_tokens_generated=sum(m.tokens_generated for m in metrics_list),
                individual_requests=[m.to_dict() for m in metrics_list],
            )
            all_worker_metrics.append(wm)
            logger.info(f"Ramp {wc} workers: p50={latency_metrics.p50:.1f}ms, p99={latency_metrics.p99:.1f}ms, rps={wm.throughput_rps:.1f}")

        if not all_worker_metrics:
            raise ValueError("No completed requests during ramping")
        return all_worker_metrics

    async def run(self) -> BenchmarkResult:
        logger.info(f"Starting LLM API benchmark: {self.benchmark_id}")

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
                    self.request_metrics.clear()
                    self.request_details.clear()
                    worker_metrics = await self._run_benchmark_with_workers(worker_count)
                    all_worker_metrics.append(worker_metrics.to_dict())
        finally:
            self._pbar.close()

        end_time = datetime.now()

        result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            benchmark_type=MetricsType.LLM_API,
            timestamp=start_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            worker_metrics=all_worker_metrics,
            provider=self.config.provider,
            model_name=self.config.model_name,
            min_load=self.config.min_load,
            max_load=self.config.max_load,
            dataset_size=len(self.config.dataset),
        )

        if self.config.store_results and self.db:
            await self._store_result(result)

        if self.config.output_file:
            self._save_to_file(result)

        logger.info(f"LLM API benchmark completed: {self.benchmark_id}")
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


async def run_llm_benchmark(config: LLMAPIConfig) -> BenchmarkResult:
    benchmark = LLMAPIBenchmark(config)
    return await benchmark.run()
