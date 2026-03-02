# Experiment Data and Artifacts

This directory contains the experimental data and artifacts from our experiments.

## Unpacking Artifacts

To extract the experimental data, you'll need to unpack the `artifacts.tar.gz` file:

```bash
# Extract the tar.gz file
tar -xzvf artifacts.tar.gz

# Verify the extraction
ls -la
```

## Directory Structure

The extracted data is organized into the following directories:

### smoketests
Contains experimental data from experiments focusing on testing capability limits and general observability. This includes:
- Performance metrics from distributed LLM inference
- Resource utilization data
- Throughput and latency measurements

### rcaevaluation
Contains experimental data from experiments focusing on RCA when used for LLM inference. This includes:
- Chaos injection results and system behavior under failure conditions
- Performance metrics from distributed LLM inference
- Resource utilization data
- Throughput and latency measurements
- RCA algorithm performance evaluations

## Data Formats

The experimental data is typically stored in the following formats:
- **JSON**: Configuration files, experiment metadata, and structured results
- **CSV**: Time-series metrics, performance data, and measurement logs
- **Log files**: Detailed execution logs and debugging information
- **Configuration files**: YAML/JSON files used for experiment setup

## Usage

Refer to the other directories of this repository for detailed information about specific experimental data and analysis procedures.
