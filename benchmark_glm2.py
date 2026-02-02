#!/usr/bin/env python3
"""
GLM 4.7 Flash Benchmark Script

Measures VRAM usage and throughput for GLM 4.7 Flash fine-tuning.
Designed for fast iteration when testing optimizations.

Usage:
    python benchmark_glm.py --max_steps 20 --batch_size 2
    python benchmark_glm.py --max_steps 60 --batch_size 4 --wandb
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import set_seed

load_dotenv()

BASELINE_FILE = Path(__file__).parent / "benchmark_baseline.json"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Model settings
    model_name: str = "unsloth/GLM-4.7-Flash"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False

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

    # Logging
    use_wandb: bool = False
    wandb_project: str = "glm47_benchmark"
    run_name: Optional[str] = None
    logging_steps: int = 1

    # Benchmark options
    warmup_iters: int = 3  # Warmup iterations before timing
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

    # Compile overhead metrics
    compile_warmup_time_seconds: float = 0.0  # Time for first N steps (includes compile)
    compile_warmup_steps: int = 0
    steady_state_time_seconds: float = 0.0  # Time for remaining steps
    steady_state_steps: int = 0
    avg_compile_step_time_seconds: float = 0.0  # Avg step time during compile phase
    avg_steady_step_time_seconds: float = 0.0  # Avg step time after compile
    compile_overhead_seconds: float = 0.0  # Estimated compile overhead

    # Throughput metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    steady_samples_per_second: float = 0.0  # Throughput after compile warmup
    steady_tokens_per_second: float = 0.0

    # Training metrics
    final_loss: float = 0.0
    total_steps: int = 0
    total_samples: int = 0
    total_tokens: int = 0

    # Metadata
    timestamp: str = ""
    git_commit: str = ""

    def to_dict(self):
        return {
            "peak_memory_gb": self.peak_memory_gb,
            "peak_memory_training_gb": self.peak_memory_training_gb,
            "memory_percentage": self.memory_percentage,
            "total_time_seconds": self.total_time_seconds,
            "avg_step_time_seconds": self.avg_step_time_seconds,
            "compile_warmup_time_seconds": self.compile_warmup_time_seconds,
            "compile_warmup_steps": self.compile_warmup_steps,
            "steady_state_time_seconds": self.steady_state_time_seconds,
            "steady_state_steps": self.steady_state_steps,
            "avg_compile_step_time_seconds": self.avg_compile_step_time_seconds,
            "avg_steady_step_time_seconds": self.avg_steady_step_time_seconds,
            "compile_overhead_seconds": self.compile_overhead_seconds,
            "samples_per_second": self.samples_per_second,
            "tokens_per_second": self.tokens_per_second,
            "steady_samples_per_second": self.steady_samples_per_second,
            "steady_tokens_per_second": self.steady_tokens_per_second,
            "final_loss": self.final_loss,
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
        }

    def __str__(self):
        return f"""
═══════════════════════════════════════════════════════════════════
                     BENCHMARK RESULTS
═══════════════════════════════════════════════════════════════════
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


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_stats():
    """Get current GPU memory statistics."""
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


def get_git_commit():
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_baseline(result: BenchmarkResult, filepath: Path = BASELINE_FILE):
    """Save benchmark result as baseline for comparison."""
    data = result.to_dict()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nBaseline saved to: {filepath}")


def load_baseline(filepath: Path = BASELINE_FILE) -> Optional[dict]:
    """Load baseline benchmark result."""
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def compare_with_baseline(result: BenchmarkResult, baseline: dict):
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

    print(f"""
═══════════════════════════════════════════════════════════════════
                  COMPARISON WITH BASELINE
═══════════════════════════════════════════════════════════════════
Baseline from: {baseline.get('timestamp', 'unknown')} (commit: {baseline.get('git_commit', 'unknown')})

Memory:
  Peak VRAM:        {fmt_change(result.peak_memory_gb, baseline['peak_memory_gb'], lower_is_better=True)}

Compile Overhead:
  Compile time:     {fmt_change(result.compile_overhead_seconds, baseline.get('compile_overhead_seconds', 0), lower_is_better=True)}
  Steady step time: {fmt_change(result.avg_steady_step_time_seconds*1000, baseline.get('avg_steady_step_time_seconds', 0)*1000, lower_is_better=True)} ms

Throughput (steady-state):
  Samples/sec:      {fmt_change(result.steady_samples_per_second, baseline.get('steady_samples_per_second', 0), lower_is_better=False)}
  Tokens/sec:       {fmt_change(result.steady_tokens_per_second, baseline.get('steady_tokens_per_second', 0), lower_is_better=False)}
═══════════════════════════════════════════════════════════════════
""")


def prepare_dataset(tokenizer, config: BenchmarkConfig):
    """Load and prepare the dataset."""
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
    return dataset


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the benchmark with the given configuration."""
    from transformers import TrainerCallback

    # Create proper TrainerCallback class
    class TimingCallback(TrainerCallback):
        def __init__(self, compile_warmup_steps: int):
            self.compile_warmup_steps = compile_warmup_steps
            self.step_times = []
            self.step_start_time = None

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start_time = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            if self.step_start_time is not None:
                step_time = time.perf_counter() - self.step_start_time
                self.step_times.append(step_time)
                self.step_start_time = None

        def get_compile_stats(self):
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

    # Set seed for reproducibility
    set_seed(config.seed)

    # Clear any existing memory
    clear_memory()

    # Record initial memory
    initial_memory = get_memory_stats()
    print(f"GPU: {initial_memory['gpu_name']}")
    print(f"Max GPU memory: {initial_memory['max_memory_gb']:.2f} GB")

    # Setup wandb if requested
    if config.use_wandb:
        import wandb
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        os.environ["WANDB_PROJECT"] = config.wandb_project

    # Import unsloth after setting up environment
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer, SFTConfig

    print(f"\nLoading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
    )

    # Record memory after model load
    model_load_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Memory after model load: {model_load_memory:.2f} GB")

    # Apply LoRA
    print(f"\nApplying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj", "out_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Record memory after LoRA
    lora_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Memory after LoRA: {lora_memory:.2f} GB")

    # Prepare dataset
    dataset = prepare_dataset(tokenizer, config)

    # Create timing callback
    timing_callback = TimingCallback(config.compile_warmup_steps)

    # Create trainer
    run_name = config.run_name or f"glm47_bs{config.batch_size}_r{config.lora_r}"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=1,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=config.seed,
            report_to="wandb" if config.use_wandb else "none",
            run_name=run_name,
            output_dir="./outputs",
            max_seq_length=config.max_seq_length,
        ),
        callbacks=[timing_callback],
    )

    # Apply train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="[gMASK]<sop><|user|>",
        response_part="<|assistant|><think>",
    )

    # Record pre-training memory
    pre_train_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nMemory before training: {pre_train_memory:.2f} GB")

    # Reset peak memory stats before training
    torch.cuda.reset_peak_memory_stats()

    # Train and measure
    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"  (First {config.compile_warmup_steps} steps measured for compile overhead)")
    start_time = time.perf_counter()

    trainer_stats = trainer.train()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Get compile stats from callback
    compile_stats = timing_callback.get_compile_stats()

    # Collect results
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory_gb = gpu_stats.total_memory / 1024**3
    peak_memory_gb = torch.cuda.max_memory_reserved() / 1024**3
    peak_training_memory_gb = peak_memory_gb - pre_train_memory

    # Calculate throughput
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_samples = config.max_steps * effective_batch_size
    samples_per_second = total_samples / total_time

    # Estimate tokens (using max_seq_length as upper bound)
    estimated_tokens = total_samples * config.max_seq_length
    tokens_per_second = estimated_tokens / total_time

    # Calculate steady-state throughput
    steady_steps = compile_stats.get("steady_state_steps", config.max_steps)
    steady_time = compile_stats.get("steady_state_time", total_time)
    steady_samples = steady_steps * effective_batch_size
    steady_samples_per_second = steady_samples / steady_time if steady_time > 0 else 0
    steady_tokens_per_second = (steady_samples * config.max_seq_length) / steady_time if steady_time > 0 else 0

    result = BenchmarkResult(
        peak_memory_gb=peak_memory_gb,
        peak_memory_training_gb=peak_training_memory_gb,
        memory_percentage=(peak_memory_gb / max_memory_gb) * 100,
        total_time_seconds=total_time,
        avg_step_time_seconds=total_time / config.max_steps,
        compile_warmup_time_seconds=compile_stats.get("compile_warmup_time", 0),
        compile_warmup_steps=compile_stats.get("compile_warmup_steps", 0),
        steady_state_time_seconds=compile_stats.get("steady_state_time", 0),
        steady_state_steps=compile_stats.get("steady_state_steps", 0),
        avg_compile_step_time_seconds=compile_stats.get("avg_compile_step_time", 0),
        avg_steady_step_time_seconds=compile_stats.get("avg_steady_step_time", 0),
        compile_overhead_seconds=compile_stats.get("compile_overhead", 0),
        samples_per_second=samples_per_second,
        tokens_per_second=tokens_per_second,
        steady_samples_per_second=steady_samples_per_second,
        steady_tokens_per_second=steady_tokens_per_second,
        final_loss=trainer_stats.metrics.get("train_loss", 0.0),
        total_steps=config.max_steps,
        total_samples=total_samples,
        total_tokens=estimated_tokens,
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
    )

    # Log final results to wandb
    if config.use_wandb:
        import wandb
        wandb.log({
            "benchmark/peak_memory_gb": result.peak_memory_gb,
            "benchmark/peak_training_memory_gb": result.peak_memory_training_gb,
            "benchmark/memory_percentage": result.memory_percentage,
            "benchmark/samples_per_second": result.samples_per_second,
            "benchmark/tokens_per_second": result.tokens_per_second,
            "benchmark/steady_samples_per_second": result.steady_samples_per_second,
            "benchmark/steady_tokens_per_second": result.steady_tokens_per_second,
            "benchmark/compile_overhead_seconds": result.compile_overhead_seconds,
            "benchmark/avg_step_time_ms": result.avg_step_time_seconds * 1000,
            "benchmark/avg_steady_step_time_ms": result.avg_steady_step_time_seconds * 1000,
        })
        wandb.finish()

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GLM 4.7 Flash fine-tuning")

    # Model settings
    parser.add_argument("--model_name", type=str, default="unsloth/GLM-4.7-Flash")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--compile_warmup_steps", type=int, default=5,
                        help="Steps to measure for compile overhead")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="glm47_benchmark")
    parser.add_argument("--run_name", type=str, default=None)

    # Baseline comparison
    parser.add_argument("--save_baseline", action="store_true",
                        help="Save result as baseline for future comparison")
    parser.add_argument("--compare", action="store_true",
                        help="Compare result with saved baseline")
    parser.add_argument("--baseline_file", type=str, default=None,
                        help="Custom baseline file path")

    # Benchmark options
    parser.add_argument("--seed", type=int, default=3407)

    return parser.parse_args()


def main():
    args = parse_args()

    baseline_path = Path(args.baseline_file) if args.baseline_file else BASELINE_FILE

    config = BenchmarkConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        compile_warmup_steps=args.compile_warmup_steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        seed=args.seed,
    )

    print("=" * 70)
    print("GLM 4.7 Flash Benchmark")
    print("=" * 70)
    print(f"Config:")
    print(f"  Model:           {config.model_name}")
    print(f"  Batch size:      {config.batch_size} x {config.gradient_accumulation_steps} (GA)")
    print(f"  LoRA:            r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Max steps:       {config.max_steps}")
    print(f"  Compile warmup:  {config.compile_warmup_steps} steps")
    print(f"  Wandb:           {'enabled' if config.use_wandb else 'disabled'}")
    if args.compare:
        print(f"  Compare with:    {baseline_path}")
    print("=" * 70)

    result = run_benchmark(config)
    print(result)

    # Save baseline if requested
    if args.save_baseline:
        save_baseline(result, baseline_path)

    # Compare with baseline if requested
    if args.compare:
        baseline = load_baseline(baseline_path)
        if baseline:
            compare_with_baseline(result, baseline)
        else:
            print(f"\nNo baseline found at {baseline_path}")
            print("Run with --save_baseline first to create one.")

    return result


if __name__ == "__main__":
    main()
