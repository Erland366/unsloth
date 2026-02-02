#!/usr/bin/env python3
"""
Shared utilities for GLM 4.7 Flash benchmarks.

Contains common dataclasses, memory utilities, dataset preparation,
and timing callback used by both unsloth and HuggingFace benchmarks.
"""

import gc
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import TrainerCallback


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Model settings
    model_name: str = "unsloth/GLM-4.7-Flash"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Attention settings
    attn_implementation: str = "flash_attention_2"  # "flash_attention_2", "sdpa", "eager"

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_steps: int = 50
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    compile_warmup_steps: int = 5  # Steps to measure compile overhead

    # Dataset
    dataset_name: str = "unsloth/OpenMathReasoning-mini"
    dataset_split: str = "cot"

    # Chat template markers (model-specific)
    instruction_part: str = "[gMASK]<sop><|user|>"
    response_part: str = "<|assistant|><think>"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "glm47_benchmark"
    run_name: Optional[str] = None
    logging_steps: int = 1

    # Benchmark options
    seed: int = 3407


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    # Memory metrics
    peak_memory_gb: float = 0.0
    peak_memory_training_gb: float = 0.0
    memory_percentage: float = 0.0

    # Timing metrics
    total_time_seconds: float = 0.0
    avg_step_time_seconds: float = 0.0

    # Compile overhead metrics (primarily for unsloth)
    compile_warmup_time_seconds: float = 0.0
    compile_warmup_steps: int = 0
    steady_state_time_seconds: float = 0.0
    steady_state_steps: int = 0
    avg_compile_step_time_seconds: float = 0.0
    avg_steady_step_time_seconds: float = 0.0
    compile_overhead_seconds: float = 0.0

    # Throughput metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    steady_samples_per_second: float = 0.0
    steady_tokens_per_second: float = 0.0

    # Training metrics
    final_loss: float = 0.0
    total_steps: int = 0
    total_samples: int = 0
    total_tokens: int = 0

    # Metadata
    timestamp: str = ""
    git_commit: str = ""
    attn_implementation: str = ""
    framework: str = ""
    status: str = "ok"  # "ok" or "OOM" or "error"

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        framework_label = self.framework.upper() if self.framework else "BENCHMARK"
        return f"""
═══════════════════════════════════════════════════════════════════
                     {framework_label} RESULTS
═══════════════════════════════════════════════════════════════════
Framework:              {self.framework}
Attention:              {self.attn_implementation}

Memory Usage:
  Peak VRAM:              {self.peak_memory_gb:.2f} GB
  Peak VRAM (training):   {self.peak_memory_training_gb:.2f} GB
  Memory % of max:        {self.memory_percentage:.1f}%

Compile Overhead (first {self.compile_warmup_steps} steps):
  Compile phase time:     {self.compile_warmup_time_seconds:.2f}s ({self.avg_compile_step_time_seconds*1000:.1f} ms/step)
  Steady-state time:      {self.steady_state_time_seconds:.2f}s ({self.avg_steady_step_time_seconds*1000:.1f} ms/step)
  Est. compile overhead:  {self.compile_overhead_seconds:.2f}s

Timing (overall):
  Total time:             {self.total_time_seconds:.2f}s ({self.total_time_seconds/60:.2f} min)
  Avg step time:          {self.avg_step_time_seconds*1000:.1f} ms

Throughput:
  Overall:                {self.samples_per_second:.2f} samples/s | {self.tokens_per_second:.1f} tok/s
  Steady-state:           {self.steady_samples_per_second:.2f} samples/s | {self.steady_tokens_per_second:.1f} tok/s

Training:
  Total steps:            {self.total_steps}
  Total samples:          {self.total_samples}
  Total tokens:           {self.total_tokens:,}
  Final loss:             {self.final_loss:.4f}
═══════════════════════════════════════════════════════════════════
"""


class TimingCallback(TrainerCallback):
    """Callback to measure per-step timing and compile overhead."""

    def __init__(self, compile_warmup_steps: int = 5):
        self.compile_warmup_steps = compile_warmup_steps
        self.step_times: List[float] = []
        self.step_start_time: Optional[float] = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is not None:
            step_time = time.perf_counter() - self.step_start_time
            self.step_times.append(step_time)
            self.step_start_time = None

    def get_compile_stats(self) -> dict:
        """Calculate compile overhead statistics."""
        if not self.step_times:
            return {}

        warmup_steps = min(self.compile_warmup_steps, len(self.step_times))
        warmup_times = self.step_times[:warmup_steps]
        steady_times = self.step_times[warmup_steps:]

        avg_warmup = sum(warmup_times) / len(warmup_times) if warmup_times else 0
        avg_steady = sum(steady_times) / len(steady_times) if steady_times else avg_warmup

        compile_overhead = sum(warmup_times) - (avg_steady * warmup_steps) if steady_times else 0

        return {
            "compile_warmup_time": sum(warmup_times),
            "compile_warmup_steps": warmup_steps,
            "steady_state_time": sum(steady_times),
            "steady_state_steps": len(steady_times),
            "avg_compile_step_time": avg_warmup,
            "avg_steady_step_time": avg_steady,
            "compile_overhead": max(0, compile_overhead),
        }


def clear_memory():
    """Clear GPU memory cache and reset peak stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {
            "gpu_name": "N/A",
            "max_memory_gb": 0.0,
            "reserved_gb": 0.0,
            "allocated_gb": 0.0,
        }
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory_gb = gpu_stats.total_memory / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "gpu_name": gpu_stats.name,
        "max_memory_gb": max_memory_gb,
        "reserved_gb": reserved_gb,
        "allocated_gb": allocated_gb,
    }


def get_git_commit(cwd: Optional[Path] = None) -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd or Path(__file__).parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def prepare_dataset(tokenizer, config: BenchmarkConfig):
    """Load and prepare the dataset. Returns (dataset, avg_tokens_per_sample)."""
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    def generate_conversation(examples):
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
        return {"conversations": conversations}

    dataset = dataset.map(generate_conversation, batched=True)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Calculate actual token counts (sample first 1000 examples for efficiency)
    sample_size = min(1000, len(dataset))
    sample_texts = dataset.select(range(sample_size))["text"]
    token_counts = [
        min(len(tokenizer.encode(text)), config.max_seq_length)
        for text in sample_texts
    ]
    avg_tokens = sum(token_counts) / len(token_counts)
    print(f"Average tokens per sample: {avg_tokens:.1f} (sampled {sample_size} examples)")

    return dataset, avg_tokens


def save_result(result: BenchmarkResult, filepath: Path):
    """Save benchmark result to JSON file."""
    data = result.to_dict()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Result saved to: {filepath}")


def load_result(filepath: Path) -> Optional[dict]:
    """Load benchmark result from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def compare_results(result: BenchmarkResult, baseline: dict):
    """Print comparison between current result and baseline."""
    def pct_change(new, old):
        if old == 0:
            return 0
        return ((new - old) / old) * 100

    def fmt_change(new, old, lower_is_better=True):
        pct = pct_change(new, old)
        direction = "↓" if pct < 0 else "↑"
        color = "better" if (pct < 0) == lower_is_better else "worse"
        return f"{new:.2f} ({direction}{abs(pct):.1f}% {color})"

    # Use steady-state metrics if available
    baseline_samples_per_sec = baseline.get("steady_samples_per_second", baseline.get("samples_per_second", 0))
    baseline_tokens_per_sec = baseline.get("steady_tokens_per_second", baseline.get("tokens_per_second", 0))
    baseline_step_time = baseline.get("avg_steady_step_time_seconds", baseline.get("avg_step_time_seconds", 0))

    result_samples_per_sec = result.steady_samples_per_second or result.samples_per_second
    result_tokens_per_sec = result.steady_tokens_per_second or result.tokens_per_second
    result_step_time = result.avg_steady_step_time_seconds or result.avg_step_time_seconds

    print(f"""
═══════════════════════════════════════════════════════════════════
                  COMPARISON WITH BASELINE
═══════════════════════════════════════════════════════════════════
Baseline from: {baseline.get('timestamp', 'unknown')} (commit: {baseline.get('git_commit', 'unknown')})
Baseline framework: {baseline.get('framework', 'unknown')} → Current: {result.framework}

Memory:
  Peak VRAM:        {fmt_change(result.peak_memory_gb, baseline['peak_memory_gb'], lower_is_better=True)}

Timing:
  Avg step time:    {fmt_change(result_step_time*1000, baseline_step_time*1000, lower_is_better=True)} ms

Throughput:
  Samples/sec:      {fmt_change(result_samples_per_sec, baseline_samples_per_sec, lower_is_better=False)}
  Tokens/sec:       {fmt_change(result_tokens_per_sec, baseline_tokens_per_sec, lower_is_better=False)}
═══════════════════════════════════════════════════════════════════
""")
