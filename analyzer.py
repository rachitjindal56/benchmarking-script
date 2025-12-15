from typing import Dict, Any, List

from mongo_client import mongo_db


class BenchmarkAnalyzer:
    def __init__(self):
        self.db = mongo_db.get_database()
        self.collection = self.db['benchmark_results']

    async def get_result_by_id(self, benchmark_id: str) -> Dict[str, Any]:
        return await self.collection.find_one({'benchmark_id': benchmark_id})

    async def get_all_results(
        self,
        benchmark_type: str = None,
        provider: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = {}
        if benchmark_type:
            query['benchmark_type'] = benchmark_type
        if provider:
            query['provider'] = provider

        cursor = self.collection.find(query).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_comparison(self, benchmark_ids: List[str]) -> Dict[str, Any]:
        results = []
        for bid in benchmark_ids:
            result = await self.get_result_by_id(bid)
            if result:
                results.append(result)

        if not results:
            raise ValueError("No benchmarks found to compare")

        comparison = {
            'benchmarks': benchmark_ids,
            'count': len(results),
            'timestamps': [r.get('timestamp') for r in results],
            'summary': {}
        }

        for result in results:
            bid = result.get('benchmark_id')
            comparison['summary'][bid] = {
                'type': result.get('benchmark_type'),
                'duration_seconds': result.get('duration_seconds'),
                'dataset_size': result.get('dataset_size'),
                'worker_metrics': result.get('worker_metrics', [])
            }

        return comparison

    async def get_statistics(self, benchmark_id: str) -> Dict[str, Any]:
        result = await self.get_result_by_id(benchmark_id)
        if not result:
            raise ValueError(f"Benchmark not found: {benchmark_id}")

        worker_metrics = result.get('worker_metrics', [])
        stats = {
            'benchmark_id': benchmark_id,
            'total_workers': len(worker_metrics),
            'total_requests': sum(m.get('total_requests', 0) for m in worker_metrics),
            'total_successful': sum(m.get('successful_requests', 0) for m in worker_metrics),
            'total_failed': sum(m.get('failed_requests', 0) for m in worker_metrics),
            'avg_throughput_rps': sum(m.get('throughput_rps', 0) for m in worker_metrics) / len(worker_metrics) if worker_metrics else 0,
            'best_latency_p50': min((m['latency_metrics']['p50'] for m in worker_metrics), default=0),
            'worst_latency_p99': max((m['latency_metrics']['p99'] for m in worker_metrics), default=0),
        }

        if result.get('benchmark_type') == 'llm_api':
            llm_metrics = [m for m in worker_metrics if 'avg_ttft_ms' in m]
            stats.update({
                'avg_ttft_ms': sum(m.get('avg_ttft_ms', 0) for m in llm_metrics) / len(llm_metrics) if llm_metrics else 0,
                'avg_tpot_ms': sum(m.get('avg_tpot_ms', 0) for m in llm_metrics) / len(llm_metrics) if llm_metrics else 0,
                'total_tokens': sum(m.get('total_tokens_generated', 0) for m in worker_metrics),
            })

        return stats
