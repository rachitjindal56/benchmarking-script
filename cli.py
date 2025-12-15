import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from system_api_benchmark import SystemAPIConfig, run_system_benchmark
from llm_api_benchmark import LLMAPIConfig, run_llm_benchmark
from models import MetricsType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ENV_TOKEN_MAP = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'togetherai': 'TOGETHER_API_KEY',
}


def resolve_auth_token(token: str, provider: Optional[str] = None) -> str:
    if token and token.startswith('$env:'):
        env_var = token[5:]
        return os.getenv(env_var, '')
    if not token and provider:
        env_var = ENV_TOKEN_MAP.get(provider.lower())
        if env_var:
            return os.getenv(env_var, '')
    return token or ''


class BenchmarkCLI:
    def __init__(self):
        self.config = None
        self.benchmark_type = None

    def prompt_system_api(self) -> SystemAPIConfig:
        print("\nConfiguration\n")

        endpoint = input("Enter API endpoint: ").strip()
        auth_token = input("Enter authentication token (or $env:VAR_NAME): ").strip()
        auth_token = resolve_auth_token(auth_token)
        json_path = input("Enter JSONPath for payload extraction (e.g., $.data): ").strip() or "$."
        org_id = input("Enter organization ID: ").strip()
        benchmark_run_id = input("Enter benchmark run ID: ").strip()

        min_load = int(input("Enter minimum concurrent workers (e.g., 1): "))
        max_load = int(input("Enter maximum concurrent workers (e.g., 10): "))

        if min_load > max_load:
            raise ValueError("min_load must be <= max_load")

        enable_ramping = input("Enable time-based load ramping? (y/n, default=y): ").strip().lower() != 'n'
        ramp_duration = 60
        if enable_ramping:
            ramp_duration = int(input("Enter ramp duration in seconds (default=60): ") or "60")

        dataset_path = input("Enter path to dataset JSON file: ").strip()
        dataset = self._load_dataset(dataset_path)

        return SystemAPIConfig(
            endpoint=endpoint,
            auth_token=auth_token,
            json_path=json_path,
            org_id=org_id,
            benchmark_run_id=benchmark_run_id,
            max_load=max_load,
            min_load=min_load,
            dataset=dataset,
            enable_ramping=enable_ramping,
            ramp_duration_seconds=ramp_duration,
        )

    def prompt_llm_api(self) -> LLMAPIConfig:
        print("\n=== LLM API Benchmarking Configuration ===\n")

        print("Supported providers: openai, togetherai, anthropic")
        provider = input("Enter LLM provider: ").strip().lower()

        if provider not in ['openai', 'togetherai', 'anthropic']:
            raise ValueError(f"Unsupported provider: {provider}")

        model_name = input("Enter model name: ").strip()
        auth_token = input("Enter authentication token (or $env:VAR_NAME, leave empty for auto): ").strip()
        auth_token = resolve_auth_token(auth_token, provider)

        if not auth_token:
            raise ValueError(f"No auth token provided and {ENV_TOKEN_MAP.get(provider)} not set")

        min_load = int(input("Enter minimum concurrent workers (e.g., 1): "))
        max_load = int(input("Enter maximum concurrent workers (e.g., 10): "))

        if min_load > max_load:
            raise ValueError("min_load must be <= max_load")

        enable_ramping = input("Enable time-based load ramping? (y/n, default=y): ").strip().lower() != 'n'
        ramp_duration = 60
        if enable_ramping:
            ramp_duration = int(input("Enter ramp duration in seconds (default=60): ") or "60")
        
        dataset_path = input("Enter path to dataset JSON file (with 'prompt' field): ").strip()
        dataset = self._load_dataset(dataset_path)
        
        return LLMAPIConfig(
            provider=provider,
            model_name=model_name,
            auth_token=auth_token,
            max_load=max_load,
            min_load=min_load,
            dataset=dataset,
            enable_ramping=enable_ramping,
            ramp_duration_seconds=ramp_duration,
        )

    def prompt_benchmark_type(self) -> str:
        print("\n=== Benchmark Type Selection ===\n")
        print("1. System API Benchmark")
        print("2. LLM API Benchmark")

        choice = input("\nSelect benchmark type (1 or 2): ").strip()

        if choice == "1":
            return "system_api"
        elif choice == "2":
            return "llm_api"
        else:
            raise ValueError("Invalid choice. Please select 1 or 2.")

    def _load_dataset(self, path: str) -> list:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("Dataset must be a JSON object or array")

        if not data:
            raise ValueError("Dataset is empty")

        logger.info(f"Loaded dataset with {len(data)} records from {path}")
        return data

    async def run_benchmark(self, output_file: Optional[str] = None, store_results: bool = True):
        benchmark_type = self.prompt_benchmark_type()

        try:
            if benchmark_type == "system_api":
                config = self.prompt_system_api()
                config.output_file = output_file
                config.store_results = store_results
                logger.info(f"Starting System API benchmark to {config.endpoint}")
                result = await run_system_benchmark(config)
            else:
                config = self.prompt_llm_api()
                config.output_file = output_file
                config.store_results = store_results
                logger.info(f"Starting LLM API benchmark for {config.model_name} on {config.provider}")
                result = await run_llm_benchmark(config)

            self._print_summary(result)

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def _print_summary(self, result: Dict[str, Any]):
        print("\n=== Benchmark Summary ===\n")
        print(f"Benchmark ID: {result.benchmark_id}")
        print(f"Type: {result.benchmark_type}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")
        print(f"Dataset Size: {result.dataset_size}")

        if result.benchmark_type == MetricsType.SYSTEM_API:
            print(f"Endpoint: {result.endpoint}")
            print(f"Org ID: {result.org_id}")
        else:
            print(f"Provider: {result.provider}")
            print(f"Model: {result.model_name}")

        print(f"\n=== Worker Metrics ===\n")
        for metrics in result.worker_metrics:
            self._print_worker_metrics(metrics)

    def _print_worker_metrics(self, metrics: Dict[str, Any]):
        worker_count = metrics['worker_count']
        print(f"\n--- {worker_count} Worker(s) ---")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Successful: {metrics['successful_requests']}")
        print(f"Failed: {metrics['failed_requests']}")
        print(f"Throughput: {metrics['throughput_rps']:.2f} req/s")

        latency = metrics['latency_metrics']
        print(f"Latency P50: {latency['p50']:.2f}ms")
        print(f"Latency P75: {latency['p75']:.2f}ms")
        print(f"Latency P90: {latency['p90']:.2f}ms")
        print(f"Latency P99: {latency['p99']:.2f}ms")
        print(f"Avg Latency: {metrics['avg_latency_ms']:.2f}ms")

        if 'avg_ttft_ms' in metrics:
            print(f"Avg TTFT: {metrics['avg_ttft_ms']:.2f}ms")
            print(f"Avg TPOT: {metrics['avg_tpot_ms']:.2f}ms")
            print(f"Total Tokens: {metrics['total_tokens_generated']}")

    def parse_config_file(
        self,
        config_path: str,
        benchmark_type: str = None,
        output_file: Optional[str] = None,
        store_results: bool = True
    ):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        if not benchmark_type:
            benchmark_type = config_data.get('benchmark_type')
            if not benchmark_type:
                raise ValueError("benchmark_type must be specified in config file or --type argument")

        logger.info(f"Parsed {benchmark_type} configuration from {config_path}")

        dataset = config_data.get('dataset') or config_data.get('dataset_file')
        if isinstance(dataset, str):
            dataset = self._load_dataset(dataset)
        elif not dataset:
            raise ValueError("dataset or dataset_file must be specified in config")

        if benchmark_type == 'system_api':
            auth_token = config_data.get('auth_token', '')
            auth_token = resolve_auth_token(auth_token)

            if config_data.get('min_load', 1) > config_data.get('max_load', 1):
                raise ValueError("min_load must be <= max_load")

            return SystemAPIConfig(
                endpoint=config_data['endpoint'],
                auth_token=auth_token,
                json_path=config_data.get('json_path', '$.'),
                org_id=config_data['org_id'],
                benchmark_run_id=config_data['benchmark_run_id'],
                max_load=config_data['max_load'],
                min_load=config_data['min_load'],
                dataset=dataset,
                enable_ramping=config_data.get('enable_ramping', True),
                ramp_duration_seconds=config_data.get('ramp_duration_seconds', 60),
                timeout=config_data.get('timeout', 30),
                max_retries=config_data.get('max_retries', 3),
                store_results=store_results,
                output_file=output_file,
                method=config_data.get('method', 'POST'),
            )

        elif benchmark_type == 'llm_api':
            provider = config_data['provider']
            auth_token = config_data.get('auth_token', '')
            auth_token = resolve_auth_token(auth_token, provider)

            if not auth_token:
                raise ValueError(f"No auth token and {ENV_TOKEN_MAP.get(provider.lower())} not set")

            if config_data.get('min_load', 1) > config_data.get('max_load', 1):
                raise ValueError("min_load must be <= max_load")

            return LLMAPIConfig(
                provider=provider,
                model_name=config_data['model_name'],
                auth_token=auth_token,
                max_load=config_data['max_load'],
                min_load=config_data['min_load'],
                dataset=dataset,
                enable_ramping=config_data.get('enable_ramping', True),
                ramp_duration_seconds=config_data.get('ramp_duration_seconds', 60),
                timeout=config_data.get('timeout', 60),
                use_chat=config_data.get('use_chat', True),
                max_retries=config_data.get('max_retries', 3),
                store_results=store_results,
                output_file=output_file,
            )

        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def main():
    parser = argparse.ArgumentParser(
        description='Enterprise Benchmarking Suite for System and LLM APIs'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--type',
        choices=['system_api', 'llm_api'],
        help='Benchmark type'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results JSON (skips MongoDB if --no-store)'
    )
    parser.add_argument(
        '--no-store',
        action='store_true',
        help='Skip storing results in MongoDB'
    )

    args = parser.parse_args()

    cli = BenchmarkCLI()
    store_results = not args.no_store

    try:
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = cli.parse_config_file(args.config, args.type, args.output, store_results)
            if isinstance(config, SystemAPIConfig):
                logger.info(f"Starting System API benchmark to {config.endpoint}")
                result = asyncio.run(run_system_benchmark(config))
            else:
                logger.info(f"Starting LLM API benchmark for {config.model_name} on {config.provider}")
                result = asyncio.run(run_llm_benchmark(config))

            cli._print_summary(result)
        else:
            asyncio.run(cli.run_benchmark(args.output, store_results))

    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
