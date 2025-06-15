# Vision Arena Benchmark for Qwen2.5-VL with API Server Count Analysis

This directory contains scripts and tools to benchmark the **Vision Arena dataset** serving performance with **Qwen2.5-VL** model using different `--api-server-count` values to analyze the impact on overall performance.

## Overview

The benchmark tests the effect of running multiple API server processes on:
- **Request throughput** (requests/second)
- **Token throughput** (tokens/second) 
- **Time to First Token (TTFT)**
- **Time Per Output Token (TPOT)**
- **End-to-end latency**

## Files

- `run_vision_arena_benchmark.py` - Comprehensive Python script with full automation
- `run_vision_arena_simple.sh` - Simple bash script for quick testing
- `README_vision_arena_benchmark.md` - This documentation file

## Quick Start

### Method 1: Simple Bash Script (Recommended for Quick Testing)

```bash
# Make script executable
chmod +x run_vision_arena_simple.sh

# Run the benchmark
./run_vision_arena_simple.sh
```

### Method 2: Python Script (More Features)

```bash
# Install dependencies
pip install requests psutil

# Run with default settings
python run_vision_arena_benchmark.py

# Run with custom model
python run_vision_arena_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct

# Run with custom parameters
python run_vision_arena_benchmark.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --num-prompts 1000 \
    --api-server-counts 1 2 4 \
    --request-rates 1 4 8 16 inf
```

## Configuration

### Default Settings

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` | Vision-language model |
| Dataset | `lmarena-ai/VisionArena-Chat` | HuggingFace dataset |
| Prompts | 500 | Number of test prompts |
| API Server Counts | [1, 2, 4, 8] | API server processes to test |
| Request Rates | [1, 4, 8, 16, "inf"] | Queries per second |
| Port | 8000 | Base server port |

### Customizing Parameters

Edit the configuration section in either script:

**Bash Script:**
```bash
# Configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH="lmarena-ai/VisionArena-Chat"
NUM_PROMPTS=500
API_SERVER_COUNTS=(1 2 4 8)
REQUEST_RATES=(1 4 8 16 "inf")
```

**Python Script:**
```python
# Use command line arguments
python run_vision_arena_benchmark.py \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --num-prompts 1000 \
    --api-server-counts 1 2 4 8 16 \
    --request-rates 1 2 4 8 16 32 inf
```

## Requirements

### System Requirements
- **GPU**: NVIDIA GPU with sufficient VRAM (16GB+ recommended for 7B model)
- **RAM**: 32GB+ system memory recommended
- **Storage**: 50GB+ for model weights
- **OS**: Linux (recommended) or macOS

### Software Requirements
- Python 3.8+
- vLLM (latest version)
- CUDA toolkit
- Dependencies: `requests`, `psutil`, `jq` (optional, for result parsing)

### Installation

```bash
# Install vLLM
pip install vllm

# Install additional dependencies
pip install requests psutil

# Install jq for result parsing (optional)
# Ubuntu/Debian:
sudo apt-get install jq
# macOS:
brew install jq
```

## Understanding the Results

### Key Metrics

1. **Request Throughput** - Requests processed per second
2. **Output Token Throughput** - Tokens generated per second
3. **Mean TTFT** - Average time to first token (latency)
4. **Mean TPOT** - Average time per output token (generation speed)

### Expected Behavior

- **Low QPS (1-4)**: Single API server should be sufficient
- **Medium QPS (8-16)**: Multiple API servers may improve throughput
- **High QPS (inf)**: Maximum throughput difference should be most apparent

### Result Files

Results are saved in `vision_arena_results/` directory:
- `asc{N}_qps{R}.json` - Individual benchmark results
- `vision_arena_report_{timestamp}.md` - Consolidated report (Python script only)

## Example Output

```
Starting Vision Arena benchmark for Qwen/Qwen2.5-VL-3B-Instruct
API server counts: 1 2 4 8
Request rates: 1 4 8 16 inf
Number of prompts: 500

============================================================
Testing with 1 API server(s)
============================================================
Starting vLLM server with 1 API server(s)...
Server on port 8000 is ready!

Testing request rate: 1
Running benchmark: API servers=1, QPS=1
Benchmark completed successfully

Testing request rate: 4
Running benchmark: API servers=1, QPS=4
Benchmark completed successfully

...

Summary of results:
API Servers: 1
  QPS 1: Request throughput=0.98 req/s, Token throughput=125.3 tok/s
  QPS 4: Request throughput=3.89 req/s, Token throughput=487.2 tok/s
  ...
```

## Manual Commands

### Start Server Manually

```bash
# Single API server
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8000 \
    --disable-log-requests \
    --trust-remote-code

# Multiple API servers (requires VLLM_USE_V1=1)
VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8000 \
    --api-server-count 4 \
    --disable-log-requests \
    --trust-remote-code
```

### Run Benchmark Manually

```bash
python benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --endpoint "/v1/chat/completions" \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 500 \
    --request-rate 8 \
    --port 8000 \
    --save-result \
    --result-dir vision_arena_results \
    --result-filename test_result.json
```

## Troubleshooting

### Common Issues

1. **Server fails to start**
   - Check GPU memory availability
   - Ensure no other vLLM processes are running
   - Verify model name and availability

2. **Benchmark fails**
   - Check server is responding: `curl http://localhost:8000/health`
   - Verify dataset path is correct
   - Check network connectivity

3. **Out of memory errors**
   - Reduce `--gpu-memory-utilization` (default 0.9)
   - Use smaller model variant
   - Reduce `--max-model-len`

### Debug Commands

```bash
# Check running processes
ps aux | grep vllm

# Check GPU memory
nvidia-smi

# Test server health
curl http://localhost:8000/health

# View server logs
tail -f /path/to/vllm/logs
```

## Advanced Usage

### Custom Model Support

To test other multimodal models, modify the model name:

```bash
# For other Qwen variants
MODEL="Qwen/Qwen2-VL-7B-Instruct"

# For LLaVA models
MODEL="llava-hf/llava-1.5-7b-hf"

# Note: Some models may require different prompt formats
```

### Different Datasets

```bash
# Use different vision arena dataset
DATASET_PATH="lmarena-ai/vision-arena-bench-v0.1"

# Use LLaVA-OneVision dataset
DATASET_PATH="lmms-lab/LLaVA-OneVision-Data"
```

### Performance Tuning

```bash
# Optimize for throughput
vllm serve $MODEL \
    --api-server-count 8 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --max-model-len 4096

# Optimize for latency
vllm serve $MODEL \
    --api-server-count 1 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 32 \
    --max-model-len 8192
```

## Contributing

To extend this benchmark:

1. **Add new models**: Modify model configuration
2. **Add new datasets**: Update dataset parsing logic
3. **Add new metrics**: Extend result analysis
4. **Add visualizations**: Create plotting scripts for results

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Vision Arena Dataset](https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [vLLM Serving Guide](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 