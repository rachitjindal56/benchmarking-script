# Benchmarking Suite

Async benchmarking tool for System APIs and LLM providers with time-based load ramping and performance metrics.

## Architecture

- **Semaphore-Based Ramping**: Smooth permit-based concurrency control
- **Async Request Execution**: All I/O non-blocking via asyncio
- **Worker Bucketing**: Requests grouped by concurrent worker count
- **Per-Bucket Aggregation**: Independent metrics for each worker level

## Functionalities

- **Multi-Protocol Support**: Benchmark HTTP system APIs and LLM providers (OpenAI, Together AI, Anthropic)
- **Time-Based Load Ramping**: Smoothly ramp from minimum to maximum concurrent workers over a configurable duration
- **Per-Worker Metrics**: Track performance metrics for each worker level independently
- **Async-First Architecture**: Built with asyncio and Motor for non-blocking database operations
- **Comprehensive Metrics**: Latency percentiles (p50, p75, p99), throughput, success rates, and detailed request logs
- **MongoDB Storage**: Persistent storage of benchmark results with query capabilities
- **LLM Streaming**: Special support for LLM streaming with TTFT (Time To First Token) and TPOT (Time Per Output Token) tracking

## Installation

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set MongoDB URI (optional, defaults to localhost:27017)
export MONGODB_URI="mongodb://localhost:27017"
export DATABASE_NAME="benchmarking"
```

## Example Workflow

1. **Create Configuration**
   ```bash
   # Edit test_ramping_config.json with your endpoint
   ```

2. **Generate Large Dataset** (Optional)
   ```bash
   python generate_dataset.py -n 1000 -o large_dataset.json
   ```

3. **Run Benchmark**
   ```bash
   python cli.py --config test_ramping_config.json
   ```

4. **View Results**
   ```bash
   python analyzer.py --query latest
   ```


## Configuration

### System API Benchmark Config

Create a JSON file (e.g., `my_benchmark.json`):

```json
{
  "benchmark_type": "system_api",
  "endpoint": "https://api.example.com/v1/endpoint",
  "auth_token": "your-auth-token",
  "json_path": "$.data",
  "org_id": "my_org",
  "benchmark_run_id": "run_001",
  "min_load": 1,
  "max_load": 10,
  "enable_ramping": true,
  "ramp_duration_seconds": 120,
  "timeout": 30,
  "dataset": "test_dataset_large.json"
}
```

**Configuration Fields:**
- `endpoint`: Target API endpoint URL
- `auth_token`: Authentication token (if required)
- `json_path`: JSONPath to extract response data (e.g., `$.data` or `$.result.value`)
- `org_id`: Organization identifier for tracking
- `benchmark_run_id`: Unique run identifier
- `min_load`: Starting number of concurrent workers
- `max_load`: Maximum number of concurrent workers
- `enable_ramping`: Enable time-based load ramping (true/false)
- `ramp_duration_seconds`: Time to ramp from min to max workers
- `timeout`: Request timeout in seconds
- `dataset`: Path to JSON dataset file


## Example Usage

### Test with Local Dataset

```bash
# Run a 10-worker ramping test over 120 seconds
python cli.py --config test_ramping_config.json
```

### Custom Endpoint

Edit `test_ramping_config.json` or create a new config:

```json
{
  "benchmark_type": "system_api",
  "endpoint": "http://localhost:3000/api/test",
  "auth_token": "Bearer token123",
  "json_path": "$",
  "org_id": "test",
  "benchmark_run_id": "local_test_001",
  "min_load": 1,
  "max_load": 5,
  "enable_ramping": true,
  "ramp_duration_seconds": 60,
  "dataset": "test_dataset_large.json"
}
```

Then run:

```bash
python cli.py --config test_ramping_config.json
```

## Output & Results

## Available Scripts

| Script | Purpose |
|--------|---------|
| `cli.py` | Main interface for running benchmarks |
| `system_api_benchmark.py` | System API benchmarking implementation |
| `llm_api_benchmark.py` | LLM API benchmarking implementation |
| `analyzer.py` | Query and export benchmark results |
| `generate_dataset.py` | Generate large synthetic datasets |
| `models.py` | Data models for metrics |
| `mongo_client.py` | MongoDB connection management |

## Performance Metrics Explained

| Metric | Definition |
|--------|-----------|
| **p50 Latency** | 50th percentile (median) response time |
| **p75 Latency** | 75th percentile response time |
| **p99 Latency** | 99th percentile response time |
| **Avg Latency** | Mean response time |
| **Throughput (req/s)** | Requests processed per second |
| **Success Rate** | Percentage of successful requests |
| **TTFT** | Time To First Token (LLM only) |
| **TPOT** | Time Per Output Token (LLM only) |