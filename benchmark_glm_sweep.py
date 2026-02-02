#!/usr/bin/env python3
"""
GLM 4.7 Flash Scalability Benchmark: Unsloth vs HuggingFace

Runs sweep benchmarks across batch sizes and/or sequence lengths,
comparing Unsloth vs HuggingFace performance. Generates PNG plots and CSV results.

IMPORTANT: Each benchmark runs in a separate subprocess to avoid cross-contamination
between unsloth (which patches transformers globally) and vanilla HuggingFace.

Usage:
    # Run both sweeps (default)
    python benchmark_glm_sweep.py

    # Sweep batch sizes only
    python benchmark_glm_sweep.py --sweep batch_size

    # Sweep sequence lengths only
    python benchmark_glm_sweep.py --sweep seq_length

    # Custom values with fewer steps for quick testing
    python benchmark_glm_sweep.py --sweep batch_size --batch_sizes 1,2 --steps 10

    # Full production run
    python benchmark_glm_sweep.py --sweep all --steps 40
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from benchmark_common import (
    get_git_commit,
    get_memory_stats,
)


@dataclass
class SweepConfig:
    """Configuration for benchmark sweeps."""
    model_name: str = "unsloth/GLM-4.7-Flash"
    max_steps: int = 40
    warmup_steps: int = 5
    compile_warmup_steps: int = 5

    # Sweep values
    batch_sizes: Tuple[int, ...] = (1, 2, 4, 8)
    seq_lengths: Tuple[int, ...] = (512, 1024, 2048, 4096)

    # Fixed parameters for sweeps
    fixed_batch_size: int = 2
    fixed_seq_length: int = 2048

    # Other training settings
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16

    # Output
    output_dir: str = "results"

    # Options
    seed: int = 3407
    attn_implementation: str = "flash_attention_2"


def create_error_row(
    dimension: str,
    value: int,
    framework: str,
    batch_size: int,
    seq_length: int,
    status: str = "error",
) -> dict:
    """Create a result row for failed cases."""
    return {
        "dimension": dimension,
        "value": value,
        "framework": framework,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "tokens_per_second": None,
        "training_vram_gb": None,
        "step_time_ms": None,
        "peak_vram_gb": None,
        "final_loss": None,
        "status": status,
    }


def run_benchmark_subprocess(
    framework: str,
    batch_size: int,
    seq_length: int,
    max_steps: int,
    warmup_steps: int,
    compile_warmup_steps: int,
    model_name: str,
    attn_implementation: str,
    lora_r: int,
    lora_alpha: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    dimension: str,
    value: int,
) -> dict:
    """Run a benchmark in a separate subprocess to avoid import contamination."""
    script_dir = Path(__file__).parent

    if framework == "unsloth":
        script = script_dir / "benchmark_glm.py"
    else:
        script = script_dir / "benchmark_glm_hf.py"

    cmd = [
        sys.executable, str(script),
        "--model_name", model_name,
        "--max_seq_length", str(seq_length),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_steps", str(max_steps),
        "--warmup_steps", str(warmup_steps),
        "--compile_warmup_steps", str(compile_warmup_steps),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--learning_rate", str(learning_rate),
        "--attn_implementation", attn_implementation,
        "--seed", str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Running {framework.upper()}: {dimension}={value}")
    print(f"  batch_size={batch_size}, seq_length={seq_length}")
    print(f"{'='*60}")

    # Set environment to avoid xet logging permission issues
    import os
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per benchmark
            cwd=script_dir,
            env=env,
        )

        # Print output for monitoring
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Filter out common warnings
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines:
                if line and not any(skip in line for skip in [
                    'UserWarning',
                    'Please restructure',
                    'xet_logging',
                    'Permission denied (os error 13)',
                ]):
                    print(line, file=sys.stderr)

        if result.returncode != 0:
            # Check for OOM in output
            if "OutOfMemoryError" in result.stderr or "CUDA out of memory" in result.stderr:
                print(f"\n*** OOM: {framework} at {dimension}={value} ***\n")
                return create_error_row(dimension, value, framework, batch_size, seq_length, "OOM")
            else:
                print(f"\n*** ERROR: {framework} at {dimension}={value} (exit code {result.returncode}) ***\n")
                return create_error_row(dimension, value, framework, batch_size, seq_length, f"error:{result.returncode}")

        # Parse results from output
        # Look for the results section in the output
        row = parse_benchmark_output(result.stdout, dimension, value, framework, batch_size, seq_length)
        return row

    except subprocess.TimeoutExpired:
        print(f"\n*** TIMEOUT: {framework} at {dimension}={value} ***\n")
        return create_error_row(dimension, value, framework, batch_size, seq_length, "timeout")

    except Exception as e:
        print(f"\n*** ERROR: {framework} at {dimension}={value}: {e} ***\n")
        return create_error_row(dimension, value, framework, batch_size, seq_length, f"error:{str(e)[:30]}")


def parse_benchmark_output(
    output: str,
    dimension: str,
    value: int,
    framework: str,
    batch_size: int,
    seq_length: int,
) -> dict:
    """Parse benchmark output to extract metrics."""
    row = {
        "dimension": dimension,
        "value": value,
        "framework": framework,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "tokens_per_second": None,
        "training_vram_gb": None,
        "step_time_ms": None,
        "peak_vram_gb": None,
        "final_loss": None,
        "status": "ok",
    }

    lines = output.split('\n')
    for line in lines:
        line = line.strip()

        # Parse Peak VRAM
        if "Peak VRAM:" in line and "training" not in line.lower():
            try:
                val = line.split(":")[-1].strip().replace("GB", "").strip()
                row["peak_vram_gb"] = float(val)
            except (ValueError, IndexError):
                pass

        # Parse Training VRAM
        if "Peak VRAM (training):" in line:
            try:
                val = line.split(":")[-1].strip().replace("GB", "").strip()
                row["training_vram_gb"] = float(val)
            except (ValueError, IndexError):
                pass

        # Parse tokens/sec - look for "Steady-state:" line first, fallback to "Overall:"
        if "Steady-state:" in line and "tok/s" in line:
            try:
                # Format: "Steady-state:           2.05 samples/s | 3979.7 tok/s"
                parts = line.split("|")
                if len(parts) >= 2:
                    tok_part = parts[-1].strip()
                    val = tok_part.replace("tok/s", "").strip()
                    row["tokens_per_second"] = float(val)
            except (ValueError, IndexError):
                pass
        elif "Overall:" in line and "tok/s" in line and row["tokens_per_second"] is None:
            try:
                parts = line.split("|")
                if len(parts) >= 2:
                    tok_part = parts[-1].strip()
                    val = tok_part.replace("tok/s", "").strip()
                    row["tokens_per_second"] = float(val)
            except (ValueError, IndexError):
                pass

        # Parse step time - look for steady-state first
        if "Steady-state time:" in line and "ms/step" in line:
            try:
                # Format: "Steady-state time:      85.21s (1703.2 ms/step)"
                import re
                match = re.search(r'\((\d+\.?\d*)\s*ms/step\)', line)
                if match:
                    row["step_time_ms"] = float(match.group(1))
            except (ValueError, IndexError):
                pass
        elif "Avg step time:" in line and row["step_time_ms"] is None:
            try:
                val = line.split(":")[-1].strip().replace("ms", "").strip()
                row["step_time_ms"] = float(val)
            except (ValueError, IndexError):
                pass

        # Parse final loss
        if "Final loss:" in line:
            try:
                val = line.split(":")[-1].strip()
                row["final_loss"] = float(val)
            except (ValueError, IndexError):
                pass

    return row


def run_sweep(sweep_config: SweepConfig, dimension: str) -> pd.DataFrame:
    """Run a sweep over one dimension, return results DataFrame."""
    results = []

    if dimension == "batch_size":
        values = sweep_config.batch_sizes
        fixed_seq = sweep_config.fixed_seq_length
    else:  # seq_length
        values = sweep_config.seq_lengths
        fixed_batch = sweep_config.fixed_batch_size

    for value in values:
        # Determine config for this point
        if dimension == "batch_size":
            batch_size = value
            seq_length = fixed_seq
        else:
            batch_size = fixed_batch
            seq_length = value

        # Run Unsloth (in subprocess)
        row = run_benchmark_subprocess(
            framework="unsloth",
            batch_size=batch_size,
            seq_length=seq_length,
            max_steps=sweep_config.max_steps,
            warmup_steps=sweep_config.warmup_steps,
            compile_warmup_steps=sweep_config.compile_warmup_steps,
            model_name=sweep_config.model_name,
            attn_implementation=sweep_config.attn_implementation,
            lora_r=sweep_config.lora_r,
            lora_alpha=sweep_config.lora_alpha,
            gradient_accumulation_steps=sweep_config.gradient_accumulation_steps,
            learning_rate=sweep_config.learning_rate,
            seed=sweep_config.seed,
            dimension=dimension,
            value=value,
        )
        results.append(row)

        # Run HuggingFace (in subprocess)
        row = run_benchmark_subprocess(
            framework="huggingface",
            batch_size=batch_size,
            seq_length=seq_length,
            max_steps=sweep_config.max_steps,
            warmup_steps=sweep_config.warmup_steps,
            compile_warmup_steps=sweep_config.compile_warmup_steps,
            model_name=sweep_config.model_name,
            attn_implementation=sweep_config.attn_implementation,
            lora_r=sweep_config.lora_r,
            lora_alpha=sweep_config.lora_alpha,
            gradient_accumulation_steps=sweep_config.gradient_accumulation_steps,
            learning_rate=sweep_config.learning_rate,
            seed=sweep_config.seed,
            dimension=dimension,
            value=value,
        )
        results.append(row)

    return pd.DataFrame(results)


def plot_comparison(df: pd.DataFrame, dimension: str, output_dir: Path):
    """Generate comparison plots for a sweep.

    Generates:
    - Combined 3-panel plot (throughput, VRAM, step time)
    - Individual plots for each metric
    - Both PNG and SVG formats
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Unsloth brand color
    UNSLOTH_GREEN = "#14b789"
    unsloth_color = UNSLOTH_GREEN

    # Filter out failed runs for plotting
    df_valid = df[df["status"] == "ok"].copy()
    if df_valid.empty:
        print(f"No successful runs to plot for {dimension}")
        return

    colors = {"unsloth": unsloth_color, "huggingface": "grey"}
    markers = {"unsloth": "o", "huggingface": "s"}

    metrics = [
        ("tokens_per_second", "Throughput (tokens/sec)", "throughput"),
        ("peak_vram_gb", "Peak VRAM (GB)", "vram"),
        ("step_time_ms", "Step Time (ms)", "step_time"),
    ]

    def configure_axis(ax, dimension: str, df_valid: pd.DataFrame):
        """Configure axis properties (log scale, ticks, grid)."""
        ax.grid(True, alpha=0.3)
        if dimension == "seq_length":
            ax.set_xscale("log", base=2)
            ax.set_xticks(df_valid["value"].unique())
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    def plot_metric(ax, col: str, title: str):
        """Plot a single metric on the given axis."""
        all_values = []
        for framework in ["unsloth", "huggingface"]:
            data = df_valid[df_valid["framework"] == framework].copy()
            if data.empty:
                continue

            data = data[data[col].notna()]
            if data.empty:
                continue

            all_values.extend(data[col].tolist())
            data = data.sort_values("value")
            ax.plot(
                data["value"],
                data[col],
                marker=markers[framework],
                linestyle="-",
                label=framework,
                color=colors[framework],
                markersize=8,
                linewidth=2,
            )

        ax.set_xlabel(dimension.replace("_", " ").title())
        ax.set_ylabel(title)
        ax.legend()
        configure_axis(ax, dimension, df_valid)

        # For VRAM metrics, don't start y-axis from 0 (misleading)
        # Use a sensible range based on actual data
        if "vram" in col.lower() and all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            padding = (max_val - min_val) * 0.1
            # Start from 90% of min value, rounded down to nearest 5
            y_min = max(0, (min_val - padding) // 5 * 5)
            y_max = max_val + padding
            ax.set_ylim(y_min, y_max)

    def save_figure(fig, base_path: Path):
        """Save figure in both PNG and SVG formats."""
        png_path = base_path.with_suffix(".png")
        svg_path = base_path.with_suffix(".svg")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Plot saved: {png_path}")
        print(f"Plot saved: {svg_path}")

    # Generate combined 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (col, title, _) in enumerate(metrics):
        plot_metric(axes[i], col, title)

    plt.suptitle(
        f"Unsloth vs HuggingFace: {dimension.replace('_', ' ').title()} Scaling",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, output_dir / f"comparison_{dimension}")
    plt.close()

    # Generate individual plots for each metric
    for col, title, short_name in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_metric(ax, col, title)
        ax.set_title(
            f"Unsloth vs HuggingFace: {title}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, output_dir / f"{short_name}_{dimension}")
        plt.close()


def generate_summary(
    batch_df: Optional[pd.DataFrame],
    seq_df: Optional[pd.DataFrame],
    output_dir: Path,
    sweep_config: SweepConfig,
):
    """Generate a markdown summary of the benchmark results."""
    summary_path = output_dir / "summary.md"

    with open(summary_path, "w") as f:
        f.write("# GLM 4.7 Flash Benchmark: Unsloth vs HuggingFace\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Git commit:** {get_git_commit()}\n\n")
        f.write(f"**Model:** {sweep_config.model_name}\n\n")
        f.write(f"**Steps per config:** {sweep_config.max_steps}\n\n")

        gpu_stats = get_memory_stats()
        f.write(f"**GPU:** {gpu_stats['gpu_name']} ({gpu_stats['max_memory_gb']:.1f} GB)\n\n")

        if batch_df is not None and not batch_df.empty:
            f.write("## Batch Size Scaling\n\n")
            f.write(f"Fixed sequence length: {sweep_config.fixed_seq_length}\n\n")
            f.write("![Batch Size Comparison](batch_size/comparison_batch_size.png)\n\n")
            f.write("### Results\n\n")
            try:
                f.write(batch_df.to_markdown(index=False))
            except ImportError:
                f.write("```\n")
                f.write(batch_df.to_csv(index=False))
                f.write("```\n")
            f.write("\n\n")

        if seq_df is not None and not seq_df.empty:
            f.write("## Sequence Length Scaling\n\n")
            f.write(f"Fixed batch size: {sweep_config.fixed_batch_size}\n\n")
            f.write("![Sequence Length Comparison](seq_length/comparison_seq_length.png)\n\n")
            f.write("### Results\n\n")
            try:
                f.write(seq_df.to_markdown(index=False))
            except ImportError:
                f.write("```\n")
                f.write(seq_df.to_csv(index=False))
                f.write("```\n")
            f.write("\n\n")

        # Add speedup summary
        f.write("## Summary\n\n")

        for name, df in [("Batch Size", batch_df), ("Seq Length", seq_df)]:
            if df is None or df.empty:
                continue

            df_valid = df[df["status"] == "ok"]
            unsloth_df = df_valid[df_valid["framework"] == "unsloth"]
            hf_df = df_valid[df_valid["framework"] == "huggingface"]

            if unsloth_df.empty or hf_df.empty:
                continue

            # Calculate average speedup
            merged = pd.merge(
                unsloth_df[["value", "tokens_per_second", "peak_vram_gb"]].dropna(),
                hf_df[["value", "tokens_per_second", "peak_vram_gb"]].dropna(),
                on="value",
                suffixes=("_unsloth", "_hf"),
            )

            if not merged.empty:
                avg_speedup = (
                    merged["tokens_per_second_unsloth"] / merged["tokens_per_second_hf"]
                ).mean()
                avg_vram_reduction = (
                    (merged["peak_vram_gb_hf"] - merged["peak_vram_gb_unsloth"])
                    / merged["peak_vram_gb_hf"]
                    * 100
                ).mean()

                f.write(f"### {name} Sweep\n\n")
                f.write(f"- **Average speedup:** {avg_speedup:.2f}x faster with Unsloth\n")
                f.write(f"- **Average VRAM reduction:** {avg_vram_reduction:.1f}% less VRAM with Unsloth\n\n")

    print(f"Summary saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scalability benchmark: Unsloth vs HuggingFace comparison"
    )

    # Sweep selection
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["batch_size", "seq_length", "all"],
        help="Which sweep to run (default: all)",
    )

    # Sweep values
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--seq_lengths",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated sequence lengths to test",
    )

    # Fixed parameters
    parser.add_argument(
        "--fixed_batch_size",
        type=int,
        default=2,
        help="Fixed batch size for seq_length sweep",
    )
    parser.add_argument(
        "--fixed_seq_length",
        type=int,
        default=2048,
        help="Fixed seq length for batch_size sweep",
    )

    # Training settings
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Training steps per configuration",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="LR warmup steps",
    )
    parser.add_argument(
        "--compile_warmup_steps",
        type=int,
        default=5,
        help="Steps to measure for compile overhead",
    )

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/GLM-4.7-Flash",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results",
    )

    parser.add_argument("--seed", type=int, default=3407)

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse sweep values
    batch_sizes = tuple(int(x) for x in args.batch_sizes.split(","))
    seq_lengths = tuple(int(x) for x in args.seq_lengths.split(","))

    sweep_config = SweepConfig(
        model_name=args.model_name,
        max_steps=args.steps,
        warmup_steps=args.warmup_steps,
        compile_warmup_steps=args.compile_warmup_steps,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        fixed_batch_size=args.fixed_batch_size,
        fixed_seq_length=args.fixed_seq_length,
        output_dir=args.output_dir,
        seed=args.seed,
        attn_implementation=args.attn_implementation,
    )

    # Setup output directory
    output_dir = Path(sweep_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GLM 4.7 Flash Scalability Benchmark")
    print("=" * 70)
    print(f"Sweep:             {args.sweep}")
    print(f"Steps per config:  {sweep_config.max_steps}")
    print(f"Batch sizes:       {sweep_config.batch_sizes}")
    print(f"Seq lengths:       {sweep_config.seq_lengths}")
    print(f"Fixed batch size:  {sweep_config.fixed_batch_size}")
    print(f"Fixed seq length:  {sweep_config.fixed_seq_length}")
    print(f"Output dir:        {output_dir}")
    print("=" * 70)

    batch_df = None
    seq_df = None

    # Run batch size sweep
    if args.sweep in ["batch_size", "all"]:
        print("\n" + "=" * 70)
        print("BATCH SIZE SWEEP")
        print("=" * 70)

        batch_dir = output_dir / "batch_size"
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_df = run_sweep(sweep_config, "batch_size")
        batch_df.to_csv(batch_dir / "results.csv", index=False)
        print(f"\nResults saved: {batch_dir / 'results.csv'}")

        plot_comparison(batch_df, "batch_size", batch_dir)

    # Run sequence length sweep
    if args.sweep in ["seq_length", "all"]:
        print("\n" + "=" * 70)
        print("SEQUENCE LENGTH SWEEP")
        print("=" * 70)

        seq_dir = output_dir / "seq_length"
        seq_dir.mkdir(parents=True, exist_ok=True)

        seq_df = run_sweep(sweep_config, "seq_length")
        seq_df.to_csv(seq_dir / "results.csv", index=False)
        print(f"\nResults saved: {seq_dir / 'results.csv'}")

        plot_comparison(seq_df, "seq_length", seq_dir)

    # Generate summary
    generate_summary(batch_df, seq_df, output_dir, sweep_config)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
