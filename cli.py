import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

from system_api_benchmark import SystemAPIConfig, run_system_benchmark
from llm_api_benchmark import LLMAPIConfig, run_llm_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkCLI:    
    def __init__(self):
        self.config = None
        self.benchmark_type = None
    
    def prompt_system_api(self) -> SystemAPIConfig:
        print("\nConfiguration\n")
        
        endpoint = input("Enter API endpoint (e.g., https://api.example.com/v1/endpoint): ").strip()
        auth_token = input("Enter authentication token: ").strip()
        json_path = input("Enter JSONPath for payload extraction (e.g., $.data): ").strip()
        org_id = input("Enter organization ID: ").strip()
        benchmark_run_id = input("Enter benchmark run ID: ").strip()
        
        min_load = int(input("Enter minimum concurrent workers (e.g., 1): "))
        max_load = int(input("Enter maximum concurrent workers (e.g., 10): "))
        
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
        print("\nLLM API Benchmarking Configuration\n")
        
        print("Supported providers: openai, togetherai, anthropic")
        provider = input("Enter LLM provider: ").strip().lower()
        
        if provider not in ['openai', 'togetherai', 'anthropic']:
            raise ValueError(f"Unsupported provider: {provider}")
        
        model_name = input("Enter model name: ").strip()
        auth_token = input("Enter authentication token: ").strip()
        
        min_load = int(input("Enter minimum concurrent workers (e.g., 1): "))
        max_load = int(input("Enter maximum concurrent workers (e.g., 10): "))
        
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
        print("\nBenchmark Type Selection\n")
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
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("Dataset must be a JSON object or array")
            
            logger.info(f"Loaded dataset with {len(data)} records from {path}")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset file: {e}")
    
    async def run_benchmark(self):
        benchmark_type = self.prompt_benchmark_type()
        
        try:
            if benchmark_type == "system_api":
                config = self.prompt_system_api()
                logger.info(f"Starting System API benchmark to {config.endpoint}")
                result = await run_system_benchmark(config)
            else:
                config = self.prompt_llm_api()
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
        
        if result.benchmark_type == "system_api":
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
    
    def parse_config_file(self, config_path: str, benchmark_type: str = None):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            if not benchmark_type:
                benchmark_type = config_data.get('benchmark_type')
                if not benchmark_type:
                    raise ValueError("benchmark_type must be specified in config file or --type argument")
            
            logger.info(f"Parsed {benchmark_type} configuration from {config_path}")
            
            if benchmark_type == 'system_api':
                dataset = config_data.get('dataset', [])
                if isinstance(dataset, str):
                    dataset = self._load_dataset(dataset)
                
                return SystemAPIConfig(
                    endpoint=config_data['endpoint'],
                    auth_token=config_data['auth_token'],
                    json_path=config_data.get('json_path', '$.'),
                    org_id=config_data['org_id'],
                    benchmark_run_id=config_data['benchmark_run_id'],
                    max_load=config_data['max_load'],
                    min_load=config_data['min_load'],
                    dataset=dataset,
                    enable_ramping=config_data.get('enable_ramping', True),
                    ramp_duration_seconds=config_data.get('ramp_duration_seconds', 60),
                    timeout=config_data.get('timeout', 30),
                )
            
            elif benchmark_type == 'llm_api':
                dataset = config_data.get('dataset', [])
                if isinstance(dataset, str):
                    dataset = self._load_dataset(dataset)
                
                return LLMAPIConfig(
                    provider=config_data['provider'],
                    model_name=config_data['model_name'],
                    auth_token=config_data['auth_token'],
                    max_load=config_data['max_load'],
                    min_load=config_data['min_load'],
                    dataset=dataset,
                    enable_ramping=config_data.get('enable_ramping', True),
                    ramp_duration_seconds=config_data.get('ramp_duration_seconds', 60),
                    timeout=config_data.get('timeout', 60),
                )
            
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required config field: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse config file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Enterprise Benchmarking Suite for System and LLM APIs'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file (alternative to interactive mode)'
    )
    parser.add_argument(
        '--type',
        choices=['system_api', 'llm_api'],
        help='Benchmark type (system_api or llm_api)'
    )
    
    args = parser.parse_args()
    
    cli = BenchmarkCLI()
    
    try:
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = cli.parse_config_file(args.config, args.type)
            if isinstance(config, SystemAPIConfig):
                logger.info(f"Starting System API benchmark to {config.endpoint}")
                result = asyncio.run(run_system_benchmark(config))
            else:
                logger.info(f"Starting LLM API benchmark for {config.model_name} on {config.provider}")
                result = asyncio.run(run_llm_benchmark(config))
            
            cli._print_summary(result)
        else:
            asyncio.run(cli.run_benchmark())
    
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
