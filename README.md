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

## Key Assumptions

The benchmarking suite makes the following assumptions about your API and setup:

### General Assumptions
1. **Monotonic Worker Increase**: Workers only increase during ramping
   - Workers start at `min_load` and increase to `max_load`
   - Requests are bucketed by the stable worker count at request time
   - No worker reduction during benchmark

2. **Dataset Reusability**: Dataset cycles if exhausted
   - If dataset is smaller than total requests, records are reused in order
   - No shuffling or randomization of dataset order

3. **Timeout Applies Per Request**: Each request has individual timeout
   - `timeout` field applies to each individual request
   - Total benchmark time = actual request processing time, not affected by timeout value

4. **MongoDB Connection**: Already established at benchmark start
   - MongoDB URI and database name are read from environment or defaults
   - No validation that MongoDB is running before benchmark starts
   - Results stored asynchronously; failures are logged but don't halt benchmark

5. **Latency Calculation**: Based on `time.perf_counter()`
   - Uses monotonic system time for nanosecond-precision latency
   - Latency includes full network round-trip time (request to response)
   - Each worker's latency measured independently; no inter-worker latency correlation
   - p50, p75, p99 calculated using `statistics` module quantiles

### System API Assumptions
1. **Authentication Method**: Bearer token authentication
   - Auth token is passed in `Authorization` header as `Bearer {auth_token}`
   - If `auth_token` is empty, no authorization header is sent
   - If your API uses a different auth method, you can pass custom headers via `headers` config field

2. **Request Format**: POST requests with JSON body
   - All requests are sent as POST to the endpoint
   - Payload is JSON-formatted based on dataset records
   - Dataset records are passed directly as JSON body

3. **Success Status Codes**: HTTP 200-299 (2xx range)
   - Any response with status code in the 200-299 range is considered successful
   - All other status codes (3xx, 4xx, 5xx) are marked as failed

4. **Payload Extraction**: JSONPath for dynamic payload creation
   - `json_path` field uses JSONPath syntax to extract nested data from dataset records
   - Example: `$.data` extracts the `data` field from record
   - If extraction fails, the entire record is sent as payload

5. **Dataset Structure**: Array of objects
   - Dataset must be a JSON array of objects
   - Each object represents one request payload
   - All fields in each object are sent to the API

### LLM API Assumptions
1. **Streaming Response**: OpenAI-compatible streaming API
   - LLM requests expect streaming responses with token-by-token output
   - TTFT (Time To First Token) is measured from request start to first token chunk
   - TPOT (Time Per Output Token) is calculated from token intervals

2. **Prompt Field**: Dataset records must have `prompt` field
   - Each dataset record should contain a `prompt` field with the text to send to LLM
   - Other fields in the record are ignored

3. **AsyncOpenAI Compatibility**: Providers must be OpenAI-compatible
   - All LLM providers use AsyncOpenAI client with configurable base URL
   - Supported: OpenAI (https://api.openai.com/v1), Together AI, Anthropic-compatible endpoints


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