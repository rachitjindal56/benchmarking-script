import json
import sys
from pathlib import Path

from mongo_client import mongo_db
from llm_api_benchmark import LLMAPIConfig
from system_api_benchmark import SystemAPIConfig


def test_mongo_connection():
    try:
        import asyncio
        asyncio.run(mongo_db.connect_to_database())
        print("MongoDB connection successful")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False


def test_system_api_config():
    try:
        config = SystemAPIConfig(
            endpoint="http://localhost:8000/api/test",
            auth_token="test_token",
            json_path="$.data",
            org_id="test_org",
            benchmark_run_id="test_run",
            max_load=2,
            min_load=1,
            dataset=[
                {"id": "1", "data": {"query": "test"}},
                {"id": "2", "data": {"query": "test2"}},
            ],
        )
        print("System API configuration valid")
        return True
    except Exception as e:
        print(f"System API configuration failed: {e}")
        return False


def test_llm_api_config():
    try:
        config = LLMAPIConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            auth_token="test_token",
            min_load=1,
            max_load=2,
            dataset=[
                {"id": "1", "prompt": "What is AI?"},
                {"id": "2", "prompt": "Explain ML"},
            ],
        )
        print("LLM API configuration valid")
        return True
    except Exception as e:
        print(f"LLM API configuration failed: {e}")
        return False


def test_dataset_loading():
    try:
        dataset_path = Path(__file__).parent / "dataset_example.json"
        if not dataset_path.exists():
            print("Dataset example file not found")
            return False
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        print(f"Dataset loading successful ({len(data)} records)")
        return True
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return False


def main():
    print("Benchmarking Suite Setup Test\n")
    
    tests = [
        ("MongoDB Connection", test_mongo_connection),
        ("System API Config", test_system_api_config),
        ("LLM API Config", test_llm_api_config),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        results.append(test_func())
    
    print("\nTest Summary")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\nAll tests passed! Ready to run benchmarks.")
        return 0
    else:
        print("\nSome tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
