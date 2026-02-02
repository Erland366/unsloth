#!/usr/bin/env python3
"""
GLM 4.7 Flash HuggingFace Baseline Benchmark Script

Measures VRAM usage and throughput for GLM 4.7 Flash fine-tuning using
vanilla HuggingFace + PEFT (no unsloth optimizations).

This serves as a baseline to compare against unsloth's optimizations.

Usage:
    python benchmark_glm_hf.py --max_steps 20 --batch_size 2
    python benchmark_glm_hf.py --max_steps 60 --batch_size 4 --wandb
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import set_seed

from benchmark_common import (
    BenchmarkConfig,
    BenchmarkResult,
    TimingCallback,
    clear_memory,
    get_memory_stats,
    get_git_commit,
    prepare_dataset,
    save_result,
    load_result,
    compare_results,
)

load_dotenv()

BASELINE_FILE = Path(__file__).parent / "benchmark_baseline_hf.json"


def run_hf_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the HuggingFace baseline benchmark with the given configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

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

    # Configure quantization if requested
    quantization_config = None
    torch_dtype = torch.bfloat16
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"\nLoading model: {config.model_name}")
    print(f"Attention implementation: {config.attn_implementation}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=config.attn_implementation,
    )

    # Record memory after model load
    model_load_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Memory after model load: {model_load_memory:.2f} GB")

    # Configure LoRA - match unsloth's target modules for GLM
    print(f"\nApplying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")

    # GLM uses MLA (Multi-head Latent Attention) with different module names
    # target_modules: attention only (qkvo)
    target_modules = [
        "q_a_proj", "q_b_proj",
        "kv_a_proj_with_mqa", "kv_b_proj",
        "o_proj",
    ]

    # target_parameters: MoE expert weights (gate_up_proj, down_proj)
    target_parameters = [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    print(f"MoE expert LoRA via target_parameters: {target_parameters}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        target_parameters=target_parameters,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Record memory after LoRA
    lora_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Memory after LoRA: {lora_memory:.2f} GB")

    # Prepare dataset
    dataset, avg_tokens_per_sample = prepare_dataset(tokenizer, config)

    # Create timing callback
    timing_callback = TimingCallback(config.compile_warmup_steps)

    # Create trainer
    run_name = config.run_name or f"glm47_hf_bs{config.batch_size}_r{config.lora_r}"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
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
            output_dir="./outputs_hf",
            max_length=config.max_seq_length,
            bf16=True,
            gradient_checkpointing=True,
        ),
        callbacks=[timing_callback],
    )

    # Record pre-training memory
    pre_train_memory = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nMemory before training: {pre_train_memory:.2f} GB")

    # Reset peak memory stats before training
    torch.cuda.reset_peak_memory_stats()

    # Train and measure
    print(f"\nStarting training for {config.max_steps} steps...")

    # Synchronize GPU before timing for accurate measurement
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    trainer_stats = trainer.train()

    # Synchronize GPU after training to ensure all ops complete
    torch.cuda.synchronize()
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

    # Use actual token counts (based on sampled average)
    actual_tokens = int(total_samples * avg_tokens_per_sample)
    tokens_per_second = actual_tokens / total_time

    # Calculate steady-state throughput
    steady_steps = compile_stats.get("steady_state_steps", config.max_steps)
    steady_time = compile_stats.get("steady_state_time", total_time)
    steady_samples = steady_steps * effective_batch_size
    steady_samples_per_second = steady_samples / steady_time if steady_time > 0 else 0
    steady_tokens_per_second = (steady_samples * avg_tokens_per_sample) / steady_time if steady_time > 0 else 0

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
        total_tokens=actual_tokens,
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
        attn_implementation=config.attn_implementation,
        framework="huggingface",
        status="ok",
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
            "benchmark/avg_step_time_ms": result.avg_step_time_seconds * 1000,
            "benchmark/attn_implementation": result.attn_implementation,
            "benchmark/framework": result.framework,
        })
        wandb.finish()

    return result


# Backward compatibility alias
run_benchmark = run_hf_benchmark


def compare_with_unsloth(result: BenchmarkResult, unsloth_baseline_path: Path):
    """Print comparison between HF result and unsloth baseline."""
    if not unsloth_baseline_path.exists():
        print(f"\nNo unsloth baseline found at {unsloth_baseline_path}")
        return

    unsloth = load_result(unsloth_baseline_path)
    if unsloth is None:
        return

    def pct_change(new, old):
        if old == 0:
            return 0
        return ((new - old) / old) * 100

    def fmt_compare(hf_val, unsloth_val, lower_is_better=True):
        pct = pct_change(hf_val, unsloth_val)
        if lower_is_better:
            status = "worse" if pct > 0 else "better"
        else:
            status = "better" if pct > 0 else "worse"
        direction = "↑" if pct > 0 else "↓"
        return f"HF: {hf_val:.2f} vs Unsloth: {unsloth_val:.2f} ({direction}{abs(pct):.1f}% {status})"

    # Use steady-state metrics from unsloth if available
    unsloth_samples_per_sec = unsloth.get("steady_samples_per_second", unsloth.get("samples_per_second", 0))
    unsloth_tokens_per_sec = unsloth.get("steady_tokens_per_second", unsloth.get("tokens_per_second", 0))
    unsloth_step_time = unsloth.get("avg_steady_step_time_seconds", unsloth.get("avg_step_time_seconds", 0))

    print(f"""
═══════════════════════════════════════════════════════════════════
              COMPARISON: HuggingFace vs Unsloth
═══════════════════════════════════════════════════════════════════
Unsloth baseline from: {unsloth.get('timestamp', 'unknown')}

Memory:
  Peak VRAM:        {fmt_compare(result.peak_memory_gb, unsloth['peak_memory_gb'], lower_is_better=True)}

Timing:
  Avg step time:    {fmt_compare(result.avg_step_time_seconds*1000, unsloth_step_time*1000, lower_is_better=True)} ms

Throughput:
  Samples/sec:      {fmt_compare(result.samples_per_second, unsloth_samples_per_sec, lower_is_better=False)}
  Tokens/sec:       {fmt_compare(result.tokens_per_second, unsloth_tokens_per_sec, lower_is_better=False)}

Summary:
  Memory savings with Unsloth:    {pct_change(result.peak_memory_gb, unsloth['peak_memory_gb']):.1f}% {"more" if pct_change(result.peak_memory_gb, unsloth['peak_memory_gb']) > 0 else "less"} VRAM with HF
  Speed improvement with Unsloth: {pct_change(unsloth_tokens_per_sec, result.tokens_per_second):.1f}% {"faster" if pct_change(unsloth_tokens_per_sec, result.tokens_per_second) > 0 else "slower"}
═══════════════════════════════════════════════════════════════════
""")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GLM 4.7 Flash with HuggingFace (baseline)")

    # Model settings
    parser.add_argument("--model_name", type=str, default="unsloth/GLM-4.7-Flash")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation: flash_attention_2, sdpa, or eager")

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
    parser.add_argument("--wandb_project", type=str, default="glm47_benchmark_hf")
    parser.add_argument("--run_name", type=str, default=None)

    # Baseline comparison
    parser.add_argument("--save_baseline", action="store_true",
                        help="Save result as HF baseline")
    parser.add_argument("--compare_unsloth", action="store_true",
                        help="Compare with unsloth baseline")
    parser.add_argument("--unsloth_baseline", type=str, default=None,
                        help="Path to unsloth baseline file")

    # Benchmark options
    parser.add_argument("--seed", type=int, default=3407)

    return parser.parse_args()


def main():
    args = parse_args()

    unsloth_baseline_path = Path(args.unsloth_baseline) if args.unsloth_baseline else (
        Path(__file__).parent / "benchmark_baseline.json"
    )

    config = BenchmarkConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_implementation=args.attn_implementation,
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
    print("GLM 4.7 Flash HuggingFace Baseline Benchmark")
    print("=" * 70)
    print(f"Config:")
    print(f"  Model:           {config.model_name}")
    print(f"  Attention:       {config.attn_implementation}")
    print(f"  Batch size:      {config.batch_size} x {config.gradient_accumulation_steps} (GA)")
    print(f"  LoRA:            r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Max steps:       {config.max_steps}")
    print(f"  Wandb:           {'enabled' if config.use_wandb else 'disabled'}")
    if args.compare_unsloth:
        print(f"  Compare with:    {unsloth_baseline_path}")
    print("=" * 70)

    result = run_hf_benchmark(config)
    print(result)

    # Save baseline if requested
    if args.save_baseline:
        save_result(result, BASELINE_FILE)

    # Compare with unsloth baseline if requested
    if args.compare_unsloth:
        compare_with_unsloth(result, unsloth_baseline_path)

    return result


if __name__ == "__main__":
    main()
