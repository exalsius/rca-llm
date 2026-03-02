# Ansible Infrastructure Automation for LLM Evaluation

This Ansible project provides automated infrastructure setup and experiment execution for large language model (LLM) evaluation in Kubernetes environments. The system supports distributed model deployment, load testing, chaos engineering, and comprehensive observability data collection for root cause analysis (RCA) evaluation.

## Project Structure

```markdown
ansible/
├── configs/                          # Model and experiment configurations
│   ├── config_base.yaml             # Base configuration parameters
│   ├── config_vllm.yaml             # VLLM-specific settings
│   └── config_*.yaml                # Model-specific configurations
├── roles/                           # Ansible roles for different components
│   ├── experiment/                  # Experiment execution roles
│   │   ├── chaos/                   # Chaos engineering and anomaly injection
│   │   ├── load/                    # Load testing and benchmarking
│   │   └── model/                   # Model deployment and management
│   ├── observability/               # Observability stack components
│   │   ├── cilium/                  # Cilium CNI and monitoring
│   │   ├── deepflow/                # DeepFlow network observability
│   │   ├── istio/                   # Istio service mesh
│   │   ├── loki/                    # Log aggregation
│   │   ├── opentelemetry/           # Distributed tracing
│   │   └── prometheus/              # Metrics collection
│   └── ray/                         # Ray cluster management
├── inventories/                     # Infrastructure inventory files
├── experiments-*.yaml               # Predefined experiment configurations
├── execute-experiment.yaml          # Main experiment execution playbook
├── install-*.yaml                   # Infrastructure installation playbooks
└── fetched/                         # Experiment results and artifacts
```

## Prerequisites

- **Python 3.12+** with pip/uv package manager
- **Ansible 2.17+** and ansible-core 2.17+
- **Kubernetes cluster** (k3s, GKE, EKS, or similar)
- **NVIDIA GPU support** for model inference
- **kubectl** configured for cluster access
- **Helm 3.x** for package management

## Installation

1. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

2. **Configure inventory:**
   ```bash
   cp example.inventory.ini inventories/your-cluster.ini
   # Edit the inventory file with your cluster details
   ```

3. **Verify cluster connectivity:**
   ```bash
   ansible all -i inventories/your-cluster.ini -m ping
   ```

## Infrastructure Setup

### 1. Install Kubernetes (k3s)
```bash
ansible-playbook -i inventories/your-cluster.ini install-k3s.yaml
```

### 2. Install NVIDIA GPU Operator
```bash
ansible-playbook -i inventories/your-cluster.ini install-nvidia-operator.yaml
```

### 3. Install Observability Stack
```bash
ansible-playbook -i inventories/your-cluster.ini install-observability.yaml
```

### 4. Install Ray Cluster
```bash
ansible-playbook -i inventories/your-cluster.ini install-ray.yaml
```

## Running Experiments

### Basic Experiment Execution

Execute a single experiment using the main playbook:

```bash
ansible-playbook -i inventories/your-cluster.ini execute-experiment.yaml \
  -e "model_id=falcon-h1-7b-instruct" \
  -e "exp_type_identifier=baseline-test" \
  -e "exp_global_data_parallelism=1"
```

### Predefined Experiment Suites

**Smoke Tests:**
```bash
ansible-playbook -i inventories/your-cluster.ini experiments-smoke-tests.yaml
```

**RCA Anomaly Injection Experiments:**
```bash
ansible-playbook -i inventories/your-cluster.ini experiments-rca-anomaly-injections.yaml
```

## Configuration

### Model Configuration

Each model requires a specific configuration file in `configs/`:

- `config_base.yaml`: Common parameters for all experiments
- `config_vllm.yaml`: VLLM-specific settings
- `config_<model-name>.yaml`: Model-specific parameters

### Key Configuration Parameters

**Resource Allocation:**
- `ray_service_cpu_per_pod`: CPU cores per Ray pod
- `ray_service_gpu_per_pod`: GPUs per Ray pod
- `ray_service_memory_per_pod`: Memory allocation per pod
- `exp_global_data_parallelism`: Number of model replicas

**Load Testing:**
- `llm_benchmark_iterations`: Number of benchmark iterations
- `llm_benchmark_duration`: Duration per iteration (seconds)
- `llm_benchmark_target_qps_values`: Target queries per second
- `llm_benchmark_concurrent_workers`: Concurrent load generators

**Chaos Engineering:**
- `chaos_experiment_scenario`: Type of anomaly to inject
- `chaos_anomaly_injection_duration`: Duration of anomaly injection
- `chaos_anomaly_injection_target_container`: Target container for injection

### Experiment Types

1. **Baseline Tests**: Standard performance evaluation without anomalies
2. **Stress Tests**: CPU, memory, or GPU stress injection
3. **Network Chaos**: Network delay and packet loss simulation
4. **Resource Exhaustion**: Memory or CPU resource limitation

## Observability and Data Collection

The system automatically collects comprehensive telemetry data:

- **Metrics**: Prometheus metrics from all components
- **Logs**: Structured logs via Loki and Fluent Bit
- **Traces**: Distributed traces via OpenTelemetry
- **Network Data**: DeepFlow network observability
- **Custom Metrics**: Ray cluster and model-specific metrics

### Data Export

All experiment data is automatically exported to `fetched/<experiment-id>/`:
- `results/`: Performance metrics and benchmark results
- `grafana-configs/`: Grafana dashboard configurations
- `diverse/`: Additional telemetry data and artifacts

## Advanced Usage

### Custom Experiment Configuration

Create custom experiment configurations by extending the base playbook:

```yaml
---
- name: "Custom Experiment"
  ansible.builtin.import_playbook: execute-experiment.yaml
  vars:
    model_id: "your-model"
    exp_type_identifier: "custom-test"
    exp_global_data_parallelism: 2
    # Add custom parameters
    custom_parameter: "value"
```

### Tag-based Execution

Execute specific parts of the experiment:

```bash
# Deploy only the model
ansible-playbook execute-experiment.yaml --tags deploy

# Run only benchmarking
ansible-playbook execute-experiment.yaml --tags benchmark

# Collect only telemetry data
ansible-playbook execute-experiment.yaml --tags collect-telemetry-data
```

### Multi-Model Evaluation

Run experiments across multiple models:

```bash
for model in falcon-h1-7b-instruct llama-3dot1-8b-instruct mistral-0dot3-7b-instruct; do
  ansible-playbook -i inventories/your-cluster.ini execute-experiment.yaml \
    -e "model_id=$model" \
    -e "exp_type_identifier=baseline-$model"
done
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**: Ensure NVIDIA GPU operator is properly installed
2. **Memory Issues**: Adjust `ray_service_memory_per_pod` based on model requirements
3. **Network Connectivity**: Verify cluster networking and service mesh configuration
4. **Resource Constraints**: Monitor cluster resources and adjust parallelism settings

### Debug Mode

Enable verbose output for debugging:

```bash
ansible-playbook execute-experiment.yaml -vvv
```

### Log Collection

Check experiment logs in the `fetched/` directory or via Grafana dashboards.

## Security Considerations

- All sensitive data should be stored in Ansible Vault
- Use proper RBAC configurations for Kubernetes
- Ensure secure communication between cluster components
- Regularly update dependencies and base images

## Contributing

When extending the system:

1. Follow Ansible best practices for role organization
2. Add appropriate configuration validation
3. Update documentation for new features
4. Test changes across different model configurations
