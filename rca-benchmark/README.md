# RCA Benchmark

This repository contains an enhanced version of [RCAEval](https://github.com/phamquiluan/RCAEval), a comprehensive benchmark for Root Cause Analysis (RCA) of microservice systems. This version has been specifically adapted and refined for our experiments, incorporating several improvements and additional capabilities, for use with our LLM experiment system.
Here, we excluded some parts of the original repository (CI/CD management, docs, etc.) to keep it slim, so kindly check out the original repository if you are interested in these.

## Overview

RCAEval is a benchmark for evaluating root cause analysis methods on microservice systems with telemetry data. Our enhanced version includes:

- **Bug fixes and stability improvements** from the original RCAEval implementation
- **Docker containerization** for reproducible and isolated execution environments
- **Enhanced parameterization** for flexible experiment configuration
- **Integration of three additional RCA methods**: `microrca`, `microscope`, and `monitorrank`
- **Integration of custom telemetry datasets** obtained from LLM inference deployments

## Key Features

### Supported RCA Methods

Our implementation supports a wide range of root cause analysis methods:

**Graph-based Methods:**
- PC (Peter-Clark) with PageRank and Random Walk
- FCI (Fast Causal Inference) with PageRank and Random Walk
- Granger Causality with PageRank and Random Walk
- LiNGAM with PageRank and Random Walk
- GES (Greedy Equivalence Search) with PageRank
- CMLP with PageRank

**Specialized RCA Methods:**
- BARO (Bayesian Online Change Point Detection)
- CausalRCA
- CIRCA
- MicroCause
- E-Diagnosis
- RCD (Root Cause Detection)
- NSigma
- CausalAI

**Trace-based Methods:**
- MicroRank
- TraceRCA

**Multi-source Methods:**
- MMBaro (Multi-Modal BARO)
- PDiagnose

**Additional Methods implemented and included by us:**
- MicroRCA
- MicroScope
- MonitorRank

### Datasets

The benchmark supports multiple microservice datasets, however, we include additional datasets:
- Custom LLM inference deployment datasets (`llm-ref-stack-dp`)

## Usage

### Docker-based Execution (Recommended)

The easiest way to run experiments is using Docker:

```bash
# Build the Docker image
docker build -t rca-benchmark .

# Run all experiments
./execute_experiments.sh python3.8 # some methods require python3.8
./execute_experiments.sh python3.12 # but most methods use python3.12
```
This assumes that the LLM inference deployment datasets are present in a local directory `ctn_data/data_raw/`.

## Experiment Configuration

The `execute_experiments.py` script, which is executed within the docker container, runs three categories of experiments:

1. **Metric-based methods**: Methods that work with telemetry metrics
2. **Trace-based methods**: Methods that analyze distributed traces
3. **Multi-source methods**: Methods that combine multiple data sources

## Output

Results are saved in the `output/` directory (subdirectory of `ctn_data`) with:
- Individual method results in JSON format
- Comprehensive evaluation reports in CSV format
- Performance metrics including accuracy and speed

## Custom Datasets

To use your own telemetry datasets:

1. Place your data in the expected directory structure
2. Ensure data includes:
   - Time series metrics in CSV format
   - Injection time information in `inject_time.txt`
   - Proper column naming conventions

## Dependencies

Key dependencies include:
- Python 3.8 or 3.12
- PyTorch (for neural network-based methods)
- Causal-learn (for causal inference)
- NetworkX (for graph operations)
- Pandas and NumPy (for data processing)
