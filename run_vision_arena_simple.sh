#!/bin/bash

# Vision Arena Benchmark Script for Qwen2.5-VL with Different API Server Counts
# Usage: ./run_vision_arena_simple.sh

set -e

# Configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_PATH="lmarena-ai/VisionArena-Chat"
NUM_PROMPTS=500
BASE_PORT=8000
RESULTS_DIR="vision_arena_results"

# API server counts to test
API_SERVER_COUNTS=(1 2 4 8)

# Request rates to test
REQUEST_RATES=(1 4 8 16 "inf")

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout=300
    local start_time=$(date +%s)
    
    echo "Waiting for server on port $port to be ready..."
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            echo "Server failed to start within $timeout seconds"
            return 1
        fi
        
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "Server on port $port is ready!"
            return 0
        fi
        
        sleep 2
    done
}

# Function to kill vLLM processes
kill_vllm_processes() {
    echo "Killing existing vLLM processes..."
    pkill -f "vllm serve" || true
    pkill -f "api_server" || true
    sleep 5
}

# Function to start server
start_server() {
    local api_server_count=$1
    
    echo "Starting vLLM server with $api_server_count API server(s)..."
    
    # Set environment variable for V1 if multiple API servers
    if [ $api_server_count -gt 1 ]; then
        export VLLM_USE_V1=1
    else
        unset VLLM_USE_V1
    fi
    
    # Start server
    vllm serve $MODEL \
        --port $BASE_PORT \
        --api-server-count $api_server_count \
        --disable-log-requests \
        --disable-log-stats \
        --gpu-memory-utilization 0.9 \
        --max-model-len 8192 \
        --trust-remote-code &
    
    local server_pid=$!
    
    # Wait for server to be ready
    if ! wait_for_server $BASE_PORT; then
        kill $server_pid || true
        return 1
    fi
    
    echo $server_pid
}

# Function to run benchmark
run_benchmark() {
    local api_server_count=$1
    local request_rate=$2
    
    echo "Running benchmark: API servers=$api_server_count, QPS=$request_rate"
    
    python benchmarks/benchmark_serving.py \
        --backend openai-chat \
        --model $MODEL \
        --endpoint "/v1/chat/completions" \
        --dataset-name hf \
        --dataset-path $DATASET_PATH \
        --hf-split train \
        --num-prompts $NUM_PROMPTS \
        --request-rate $request_rate \
        --port $BASE_PORT \
        --save-result \
        --result-dir $RESULTS_DIR \
        --result-filename "asc${api_server_count}_qps${request_rate}.json" \
        --seed 42
}

# Create results directory
mkdir -p $RESULTS_DIR

# Main benchmark loop
echo "Starting Vision Arena benchmark for $MODEL"
echo "API server counts: ${API_SERVER_COUNTS[*]}"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Number of prompts: $NUM_PROMPTS"
echo "Results directory: $RESULTS_DIR"

for api_server_count in "${API_SERVER_COUNTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Testing with $api_server_count API server(s)"
    echo "============================================================"
    
    # Kill any existing processes
    kill_vllm_processes
    
    # Start server
    if server_pid=$(start_server $api_server_count); then
        echo "Server started with PID: $server_pid"
        
        # Run benchmarks with different request rates
        for request_rate in "${REQUEST_RATES[@]}"; do
            echo ""
            echo "Testing request rate: $request_rate"
            
            if run_benchmark $api_server_count $request_rate; then
                echo "Benchmark completed successfully"
            else
                echo "Benchmark failed for API servers=$api_server_count, QPS=$request_rate"
            fi
            
            sleep 2  # Brief pause between benchmarks
        done
        
        # Kill server
        echo "Stopping server (PID: $server_pid)"
        kill $server_pid || true
        
    else
        echo "Failed to start server with $api_server_count API servers"
    fi
    
    # Clean up
    kill_vllm_processes
    sleep 5  # Wait between different API server count tests
done

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETED"
echo "============================================================"
echo "Results saved in: $RESULTS_DIR"

# Generate summary
echo ""
echo "Summary of results:"
for api_server_count in "${API_SERVER_COUNTS[@]}"; do
    echo "API Servers: $api_server_count"
    for request_rate in "${REQUEST_RATES[@]}"; do
        result_file="$RESULTS_DIR/asc${api_server_count}_qps${request_rate}.json"
        if [ -f "$result_file" ]; then
            # Extract key metrics using jq if available
            if command -v jq &> /dev/null; then
                throughput=$(jq -r '.request_throughput // "N/A"' "$result_file")
                token_throughput=$(jq -r '.output_throughput // "N/A"' "$result_file")
                echo "  QPS $request_rate: Request throughput=$throughput req/s, Token throughput=$token_throughput tok/s"
            else
                echo "  QPS $request_rate: Result file exists"
            fi
        else
            echo "  QPS $request_rate: No result file"
        fi
    done
    echo ""
done

echo "To analyze results in detail, check the JSON files in $RESULTS_DIR" 