# LLM Benchmark Suite

This LLM benchmarking tool is based on the [Fireworks.ai benchmark suite](https://github.com/fw-ai/benchmark) and has been restructured and refined for use with our experiment system. It provides comprehensive load testing capabilities for Large Language Models (LLMs) with support for multiple providers and detailed performance metrics collection.

## Features

- **Multi-provider support**: OpenAI API compatible endpoints, VLLM, Anyscale Endpoints, Together.ai, Text Generation Inference (TGI), and NVIDIA Triton servers
- **Flexible load patterns**: Fixed QPS, burst mode, and continuous load testing
- **Comprehensive metrics**: Latency, throughput, token generation rates, time-to-first-token, and per-token latency
- **Streaming support**: Both streaming and non-streaming API modes
- **Configurable parameters**: Customizable prompt lengths, response lengths, temperature, and generation parameters
- **CSV export**: Results can be exported to CSV for further analysis

## Usage

### Building the Docker Image

Before using this benchmark with our ansible-enabled experiment routines, you need to build the Docker image:

```bash
cd llm-benchmark
docker build -t llm-benchmark:latest .
```

### Running with Ansible Experiments

The benchmark is designed to be used as a containerized workload within our Kubernetes-based experiment system. It's automatically deployed and configured through Ansible playbooks.

### Direct Usage (Standalone)

For standalone testing outside the experiment system:

```bash
# Using the built Docker image
docker run --rm llm-benchmark:latest \
  -H http://your-llm-service:8000 \
  --provider vllm \
  --model your-model \
  --qps 5 \
  -t 300s \
  -u 10 \
  --stream \
  --chat
```

#### Command Line Options

- `-H, --host`: Target LLM service URL
- `--provider`: LLM provider type (vllm, fireworks, openai, together, etc.)
- `--model`: Model name to benchmark
- `--qps`: Target queries per second
- `-t, --run-time`: Test duration (e.g., "300s")
- `-u, --users`: Number of concurrent users
- `-r, --spawn-rate`: User spawn rate
- `--stream`: Enable streaming API
- `--chat`: Use chat completions endpoint
- `--summary-file`: Export results to CSV file
- `-p, --prompt-tokens`: Prompt length in tokens
- `-o, --max-tokens`: Response length in tokens
- `--temperature`: Generation temperature
- `--logprobs`: Enable logprobs collection

## Metrics Collected

- **Latency metrics**: Total latency, time-to-first-token, per-token latency
- **Throughput**: Queries per second (QPS), requests per second
- **Token metrics**: Generated tokens, prompt tokens, token generation rate
- **Quality metrics**: Logprobs, response quality indicators

## Integration with Experiment System

This benchmark tool is tightly integrated with our ansible-based experiment orchestration system:

1. **Containerized deployment**: Runs as Kubernetes pods with persistent storage
2. **Automated result collection**: Results are automatically fetched and stored
3. **Telemetry integration**: Works alongside observability tools (Prometheus, Grafana)
4. **Configurable experiments**: Supports parameter sweeps and iterative testing
5. **Result aggregation**: Collects and consolidates results across multiple test runs

The benchmark results are stored in the `ansible/fetched/` directory and can be analyzed using the provided data processing tools in the experiment system.
