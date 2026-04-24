# Root Cause Analysis for Large Language Model Inference Systems

Our work has been accepted to the [QualITA](https://qualitawg.github.io/) workshop co-located with the [ICPE 2026](https://icpe2026.spec.org/) conference.

- **Publication:** [Beyond Microservices: Testing Web-Scale RCA Methods on GPU-Driven LLM Workloads](https://dl.acm.org/doi/10.1145/3777911.3800637)
- **Preprint:** Find it [here](https://arxiv.org/abs/2603.02057)

This repository contains the complete experimental framework and evaluation suite for our research on **Root Cause Analysis (RCA) in Large Language Model (LLM) inference systems**. Our work addresses the critical challenge of diagnosing performance anomalies and failures in distributed LLM deployments through comprehensive telemetry analysis and automated root cause detection.

## Research Contributions

Our research makes several key contributions to the field of LLM system reliability and observability:

1. **Comprehensive RCA Evaluation Framework**: We present the first systematic evaluation of root cause analysis methods specifically tailored for LLM inference systems, testing 20+ different RCA algorithms across various failure scenarios.

2. **Real-world LLM Telemetry Datasets**: We provide novel telemetry datasets collected from actual distributed LLM deployments under various stress conditions, including GPU stress, memory pressure, and network chaos scenarios.

3. **Chaos Engineering for LLM Systems**: We develop and validate chaos engineering techniques specifically designed for LLM inference workloads, enabling controlled failure injection and system behavior analysis.

4. **Multi-modal Observability Integration**: We demonstrate how to effectively combine metrics, logs, traces, and network data for comprehensive system diagnosis in LLM deployments.

## Repository Structure

### 🔬 **`ansible/`** - Experiment Orchestration Framework
The core infrastructure automation system that enables reproducible, large-scale experiments on Kubernetes clusters.

- **Key Components:**
  - **Infrastructure Setup**: Automated deployment of k3s, NVIDIA GPU operator, Ray clusters, and comprehensive observability stack
  - **Experiment Execution**: Configurable experiment workflows with support for multiple LLM models and failure scenarios
  - **Chaos Engineering**: Automated anomaly injection for CPU, memory, GPU, and network stress testing
  - **Data Collection**: Automated telemetry gathering from Prometheus, Grafana, Loki, OpenTelemetry, and DeepFlow

- **Supported Models**: Falcon, Llama, Mistral, Gemma, Phi, Qwen (7B-11B parameter range); many others can be easily added
- **Experiment Types**: Baseline performance, stress testing, chaos injection, and RCA evaluation

### 🔬 **`llm-benchmark/`** - Load Testing and Performance Evaluation
A comprehensive benchmarking suite for evaluating LLM inference performance under various load conditions.

- **Features:**
  - Multi-provider support (VLLM, OpenAI API, Together.ai, Anyscale, TGI, Triton)
  - Configurable load patterns (fixed QPS, burst mode, continuous load)
  - Comprehensive metrics collection (latency, throughput, token generation rates)
  - Streaming and non-streaming API support

### 🔬 **`rca-benchmark/`** - Root Cause Analysis Evaluation Suite
An enhanced version of RCAEval specifically adapted for LLM inference systems, featuring 20+ RCA algorithms and custom LLM telemetry datasets.

- **RCA Methods Evaluated:**
  - **Graph-based**: PC, FCI, Granger Causality, LiNGAM, GES, CMLP with PageRank/Random Walk
  - **Specialized**: BARO, CausalRCA, CIRCA, MicroCause, E-Diagnosis, RCD, NSigma, CausalAI
  - **Trace-based**: MicroRank, TraceRCA
  - **Multi-source**: MMBaro, PDiagnose
  - **Custom Implementations**: MicroRCA, MicroScope, MonitorRank

- **Datasets**: Custom LLM inference deployment telemetry data with controlled failure injections

### 📊 **`data/`** - Experimental Results and Artifacts
Contains the complete experimental data from our evaluation, including:
- **Smoke Tests**: Performance baselines and capability limits
- **RCA Evaluation**: Chaos injection results and RCA algorithm performance
- **Telemetry Data**: Comprehensive metrics, logs, traces, and network data

## Quick Start

### 1. **Understanding the Research Scope**
- Start with `ansible/README.md` for the complete experimental framework overview
- Review `rca-benchmark/README.md` for RCA evaluation methodology
- Check `llm-benchmark/README.md` for load testing capabilities

### 2. **Exploring Experimental Data**
```bash
# Extract experimental results
cd data/
tar -xzvf artifacts.tar.gz
# Browse smoketests/ and rcaevaluation/ directories
```

### 3. **Reproducing Experiments**
```bash
# Set up infrastructure (requires Kubernetes cluster)
cd ansible/
ansible-playbook -i inventories/your-cluster.ini install-k3s.yaml
# Check the ansible directory for more information, these commands are not complete
ansible-playbook -i inventories/your-cluster.ini install-observability.yaml

# Run a baseline experiment
ansible-playbook -i inventories/your-cluster.ini execute-experiment.yaml \
  -e "model_id=falcon-h1-7b-instruct" \
  -e "exp_type_identifier=baseline-test"
```

### 4. **RCA Evaluation**
```bash
# Run RCA benchmark on collected telemetry data
cd rca-benchmark/
docker build -t rca-benchmark .
./execute_experiments.sh python3.12
```

## Technical Highlights

- **Infrastructure**: Kubernetes-based distributed system with Ray for model serving
- **Observability**: Comprehensive telemetry stack (Prometheus, Grafana, Loki, OpenTelemetry, DeepFlow)
- **Chaos Engineering**: Controlled failure injection for realistic failure scenarios
- **Evaluation**: Systematic comparison of 20+ RCA algorithms across multiple failure types
- **Reproducibility**: Complete automation of experiment setup, execution, and data collection

## How to Cite

Publication:

```bibtex
@inproceedings{10.1145/3777911.3800637,
      author = {Scheinert, Dominik and Acker, Alexander and Wittkopp, Thorsten and Becker, Soeren and Yous, Hamza and Reddy, Karnakar and Farhat, Ibrahim and Hacid, Hakim and Kao, Odej},
      title = {Beyond Microservices: Testing Web-Scale RCA Methods on GPU-Driven LLM Workloads},
      year = {2026},
      isbn = {9798400723261},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3777911.3800637},
      doi = {10.1145/3777911.3800637},
      abstract = {Large language model (LLM) services have become an integral part of search, assistance, and decision-making applications. However, unlike traditional web or microservices, the hardware and software stack enabling LLM inference deployment is of higher complexity and far less field-tested, making it more susceptible to failures that are difficult to resolve. Keeping outage costs and quality of service degradations in check depends on shortening mean time to repair, which in practice is gated by how quickly the fault is identified, located, and diagnosed. Automated root cause analysis (RCA) accelerates failure localization by identifying the system component that failed and tracing how the failure propagated. Numerous RCA methods have been developed for traditional services, using request path tracing, resource metric and log data analysis. Yet, existing RCA methods have not been designed for LLM deployments that present distinct runtime characteristics. In this study, we evaluate the effectiveness of RCA methods on a best-practice LLM inference deployment under controlled failure injections. Across 24 methods—20 metric-based, two trace-based, and two multi-source—we find that multi-source approaches achieve the highest accuracy, metric-based methods show fault-type-dependent performance, and trace-based methods largely fail. These results reveal that existing RCA tools do not generalize to LLM systems, motivating tailored analysis techniques and enhanced observability, for which we formulate guidelines.},
      booktitle = {Companion of the 17th ACM/SPEC International Conference on Performance Engineering},
      pages = {163–172},
      numpages = {10},
      keywords = {distributed systems, reliability engineering, root cause analysis, aiops, large language models},
      location = {Italy},
      series = {ICPE Companion '26}
}
```

Preprint:

```bibtex
@misc{scheinert2026microservicestestingwebscalerca,
      title={Beyond Microservices: Testing Web-Scale RCA Methods on GPU-Driven LLM Workloads},
      author={Dominik Scheinert and Alexander Acker and Thorsten Wittkopp and Soeren Becker and Hamza Yous and Karnakar Reddy and Ibrahim Farhat and Hakim Hacid and Odej Kao},
      year={2026},
      eprint={2603.02057},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2603.02057},
}
```
