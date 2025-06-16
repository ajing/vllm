#!/usr/bin/env python3
"""
Vision Arena Serving Benchmark for Qwen2.5-VL with Different API Server Counts

This script benchmarks the effect of --api-server-count on the overall performance
when serving Qwen2.5-VL on the Vision Arena dataset.

Usage:
    python run_vision_arena_benchmark.py --model Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil


class VisionArenaBenchmark:
    def __init__(
        self,
        model: str,
        num_prompts: int = 500,
        dataset_path: str = "lmarena-ai/VisionArena-Chat",
        base_port: int = 8000,
        results_dir: str = "vision_arena_results"
    ):
        self.model = model
        self.num_prompts = num_prompts
        self.dataset_path = dataset_path
        self.base_port = base_port
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Default API server counts to test
        self.api_server_counts = [2, 4, 8]
        
        # Default request rates (QPS) to test
        self.request_rates = [8, 16]
        
    def wait_for_server(self, port: int, timeout: int = 1200) -> bool:
        """Wait for vLLM server to be ready"""
        import requests
        
        print(f"Waiting for server on port {port} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Server on port {port} is ready!")
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(2)
        
        print(f"Server on port {port} failed to start within {timeout} seconds")
        return False
    
    def kill_vllm_processes(self):
        """Kill any existing vLLM processes"""
        print("Killing existing vLLM processes...")
        try:
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
            subprocess.run(["pkill", "-f", "api_server"], check=False)
            time.sleep(5)  # Wait for processes to die
        except Exception as e:
            print(f"Error killing processes: {e}")
    
    def start_server(self, api_server_count: int) -> subprocess.Popen:
        """Start vLLM server with specified API server count"""
        print(f"Starting vLLM server with {api_server_count} API server(s)...")
        
        # Server arguments
        server_args = [
            "vllm", "serve", self.model,
            "--port", str(self.base_port),
            "--api-server-count", str(api_server_count)
        ]
        
        # Add V1 engine if using multiple API servers
        env = os.environ.copy()
        if api_server_count > 1:
            env["VLLM_USE_V1"] = "1"
        
        print(f"Server command: {' '.join(server_args)}")
        
        # Start server process
        process = subprocess.Popen(
            server_args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        if not self.wait_for_server(self.base_port):
            # Get error output before terminating
            try:
                stdout, stderr = process.communicate(timeout=5)
                print(f"Server stdout: {stdout[-1000:]}")  # Last 1000 chars
                print(f"Server stderr: {stderr[-1000:]}")  # Last 1000 chars
            except:
                pass
            process.terminate()
            raise RuntimeError("Failed to start server")
        
        return process
    
    def run_benchmark(
        self,
        api_server_count: int,
        request_rate: str,
        additional_args: Optional[List[str]] = None
    ) -> Dict:
        """Run benchmark with specified parameters"""
        
        # Client arguments
        client_args = [
            "python", "benchmarks/benchmark_serving.py",
            "--backend", "openai-chat",
            "--model", self.model,
            "--endpoint", "/v1/chat/completions",
            "--dataset-name", "hf",
            "--dataset-path", self.dataset_path,
            "--hf-split", "train",
            "--num-prompts", str(self.num_prompts),
            "--request-rate", str(request_rate),
            "--port", str(self.base_port),
            "--save-result",
            "--result-dir", str(self.results_dir),
            "--result-filename", f"asc{api_server_count}_qps{request_rate}.json",
            "--seed", "42",
        ]
        
        if additional_args:
            client_args.extend(additional_args)
        
        print(f"Running benchmark: API servers={api_server_count}, QPS={request_rate}")
        print(f"Client command: {' '.join(client_args)}")
        
        # Run benchmark
        start_time = time.time()
        result = subprocess.run(
            client_args,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr}")
            return None
        
        print(f"Benchmark completed in {duration:.2f} seconds")
        print("Output:", result.stdout[-500:])  # Print last 500 chars
        
        # Parse results from JSON file
        result_file = self.results_dir / f"asc{api_server_count}_qps{request_rate}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                benchmark_data = json.load(f)
            return benchmark_data
        
        return None
    
    def run_all_benchmarks(self, custom_args: Optional[List[str]] = None) -> Dict:
        """Run benchmarks for all API server counts and request rates"""
        results = {}
        
        for api_server_count in self.api_server_counts:
            print(f"\n{'='*60}")
            print(f"Testing with {api_server_count} API server(s)")
            print(f"{'='*60}")
            
            # Kill any existing processes
            self.kill_vllm_processes()
            
            server_process = None
            try:
                # Start server
                server_process = self.start_server(api_server_count)
                
                # Run benchmarks with different request rates
                api_results = {}
                for request_rate in self.request_rates:
                    try:
                        benchmark_result = self.run_benchmark(
                            api_server_count, 
                            request_rate,
                            custom_args
                        )
                        if benchmark_result:
                            api_results[str(request_rate)] = benchmark_result
                    except Exception as e:
                        print(f"Error running benchmark for QPS {request_rate}: {e}")
                
                results[api_server_count] = api_results
                
            except Exception as e:
                print(f"Error with {api_server_count} API servers: {e}")
            
            finally:
                # Clean up server process
                if server_process is not None:
                    try:
                        server_process.terminate()
                        server_process.wait(timeout=10)
                    except:
                        try:
                            server_process.kill()
                        except:
                            pass
                
                self.kill_vllm_processes()
                time.sleep(5)  # Wait between tests
        
        return results
    
    def save_consolidated_results(self, results: Dict) -> str:
        """Save all benchmark results to a single consolidated JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"consolidated_benchmark_results_{timestamp}.json"
        
        # Create consolidated structure
        consolidated = {
            "benchmark_info": {
                "model": self.model,
                "dataset": self.dataset_path,
                "num_prompts": self.num_prompts,
                "timestamp": timestamp,
                "api_server_counts_tested": self.api_server_counts,
                "request_rates_tested": self.request_rates,
                "total_configurations": sum(len(api_results) for api_results in results.values())
            },
            "results_by_api_server_count": results,
            "summary": {
                "best_performance": {
                    "request_throughput": {"api_server_count": 0, "request_rate": 0, "value": 0, "unit": "requests/sec"},
                    "token_throughput": {"api_server_count": 0, "request_rate": 0, "value": 0, "unit": "tokens/sec"},
                    "lowest_ttft": {"api_server_count": 0, "request_rate": 0, "value": float('inf'), "unit": "ms"},
                    "lowest_tpot": {"api_server_count": 0, "request_rate": 0, "value": float('inf'), "unit": "ms"}
                },
                "performance_comparison": []
            }
        }
        
        # Process each result to find best performance
        for api_count, api_results in results.items():
            for request_rate, data in api_results.items():
                if not data:
                    continue
                    
                # Extract metrics safely
                req_throughput = data.get('request_throughput', 0)
                token_throughput = data.get('output_throughput', 0)
                ttft = data.get('mean_ttft_ms', float('inf'))
                tpot = data.get('mean_tpot_ms', float('inf'))
                
                # Add to comparison data
                consolidated["summary"]["performance_comparison"].append({
                    "api_server_count": api_count,
                    "request_rate": request_rate,
                    "request_throughput": req_throughput,
                    "token_throughput": token_throughput,
                    "mean_ttft_ms": ttft,
                    "mean_tpot_ms": tpot
                })
                
                # Update best metrics
                if req_throughput > consolidated["summary"]["best_performance"]["request_throughput"]["value"]:
                    consolidated["summary"]["best_performance"]["request_throughput"].update({
                        "api_server_count": api_count,
                        "request_rate": request_rate,
                        "value": req_throughput
                    })
                
                if token_throughput > consolidated["summary"]["best_performance"]["token_throughput"]["value"]:
                    consolidated["summary"]["best_performance"]["token_throughput"].update({
                        "api_server_count": api_count,
                        "request_rate": request_rate,
                        "value": token_throughput
                    })
                
                if ttft < consolidated["summary"]["best_performance"]["lowest_ttft"]["value"]:
                    consolidated["summary"]["best_performance"]["lowest_ttft"].update({
                        "api_server_count": api_count,
                        "request_rate": request_rate,
                        "value": ttft
                    })
                
                if tpot < consolidated["summary"]["best_performance"]["lowest_tpot"]["value"]:
                    consolidated["summary"]["best_performance"]["lowest_tpot"].update({
                        "api_server_count": api_count,
                        "request_rate": request_rate,
                        "value": tpot
                    })
        
        # Sort performance comparison by request throughput (descending)
        consolidated["summary"]["performance_comparison"].sort(
            key=lambda x: x.get("request_throughput", 0), reverse=True
        )
        
        # Save consolidated results
        os.makedirs(self.results_dir, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        print(f"\n🎉 Consolidated results saved to: {filename}")
        return str(filename)

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive report from benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"vision_arena_report_{timestamp}.md"
        
        report = f"""# Vision Arena Benchmark Report

**Model:** {self.model}
**Dataset:** {self.dataset_path}
**Number of Prompts:** {self.num_prompts}
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| API Servers | Request Rate | Request Throughput (req/s) | Output Token Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|--------------|----------------------------|----------------------------------|----------------|----------------|
"""
        
        for api_count in sorted(results.keys()):
            for qps in self.request_rates:
                qps_str = str(qps)
                if qps_str in results[api_count]:
                    data = results[api_count][qps_str]
                    
                    # Extract metrics safely
                    req_throughput = data.get('request_throughput', 'N/A')
                    token_throughput = data.get('output_throughput', 'N/A')
                    ttft = data.get('mean_ttft_ms', 'N/A')
                    tpot = data.get('mean_tpot_ms', 'N/A')
                    
                    report += f"| {api_count} | {qps} | {req_throughput} | {token_throughput} | {ttft} | {tpot} |\n"
        
        report += f"""

## Detailed Results

"""
        
        for api_count in sorted(results.keys()):
            report += f"### {api_count} API Server{'s' if api_count > 1 else ''}\n\n"
            
            for qps in self.request_rates:
                qps_str = str(qps)
                if qps_str in results[api_count]:
                    data = results[api_count][qps_str]
                    report += f"#### Request Rate: {qps}\n\n"
                    report += f"```json\n{json.dumps(data, indent=2)}\n```\n\n"
        
        # Write report to file
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report generated: {report_file}")
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Vision Arena serving with different API server counts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model to benchmark (default: Qwen/Qwen2.5-VL-3B-Instruct)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="Number of prompts to test (default: 500)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="lmarena-ai/VisionArena-Chat",
        help="Dataset path (default: lmarena-ai/VisionArena-Chat)"
    )
    parser.add_argument(
        "--api-server-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="API server counts to test (default: [1, 2, 4, 8])"
    )
    parser.add_argument(
        "--request-rates",
        nargs="+",
        default=[1, 4, 8, 16, "inf"],
        help="Request rates to test (default: [1, 4, 8, 16, inf])"
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="Base port for vLLM server (default: 8000)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="vision_arena_results",
        help="Directory to save results (default: vision_arena_results)"
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = VisionArenaBenchmark(
        model=args.model,
        num_prompts=args.num_prompts,
        dataset_path=args.dataset_path,
        base_port=args.base_port,
        results_dir=args.results_dir
    )
    
    # Set custom parameters
    benchmark.api_server_counts = args.api_server_counts
    benchmark.request_rates = args.request_rates
    
    print(f"Starting Vision Arena benchmark for {args.model}")
    print(f"API server counts: {args.api_server_counts}")
    print(f"Request rates: {args.request_rates}")
    print(f"Number of prompts: {args.num_prompts}")
    
    # Run benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Save consolidated results to single file
    consolidated_file = benchmark.save_consolidated_results(results)
    
    # Generate report
    report = benchmark.generate_report(results)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)
    print(f"📊 Consolidated results: {consolidated_file}")
    print(f"📋 Detailed report: {benchmark.results_dir}/vision_arena_report_*.md")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(report[:1000] + "..." if len(report) > 1000 else report)


if __name__ == "__main__":
    main() 