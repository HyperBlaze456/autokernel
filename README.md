# AutoKernel

**Autoresearch for GPU kernels.** Give it any PyTorch model, go to sleep, wake up to optimized Triton kernels.

![AutoKernel Progress](progress.png)

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- which demonstrated autonomous AI agents for LLM training research. AutoKernel applies the same philosophy to GPU kernel optimization: agent modifies one file, runs a fixed evaluation, keeps or reverts, repeats forever.

## How It Works

Give AutoKernel any PyTorch model. It will:

1. **Profile** the model to find which GPU kernels are bottlenecks
2. **Extract** each bottleneck as a standalone Triton kernel
3. **Optimize** each kernel autonomously (edit, benchmark, keep/revert -- forever)
4. **Verify** end-to-end correctness and report the total speedup

The agent reads `program.md` -- the "research org code" -- which contains comprehensive instructions for autonomous operation. It edits `kernel.py` one kernel at a time, runs `bench.py` (fixed benchmark with 5-stage correctness checks + roofline analysis), and either keeps or reverts the change. The orchestrator decides when to move to the next kernel using Amdahl's law.

Each experiment takes ~90 seconds. That's ~40 experiments/hour, ~320 overnight, across all kernels.

## Quick Start

**Requirements:** NVIDIA GPU (tested on H100/A100/RTX 4090), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/RightNow-AI/autokernel.git
cd autokernel
uv sync

# One-time setup: test data + baselines
uv run prepare.py

# Profile a model (ships with GPT-2, LLaMA, BERT -- no transformers needed)
uv run profile.py --model models/llama_7b.py --class-name LlamaModel \
 --input-shape 1,512 --dtype float16

# Extract top bottleneck kernels
uv run extract.py --top 5

# Verify benchmark works
uv run bench.py
```

## Running the Agent

Spin up Claude, Codex, or any coding agent in this directory:

```
Read program.md and let's kick off a new experiment. Start with setup.
```

The agent will:
1. Profile your model and present the optimization plan
2. Create a branch (e.g., `autokernel/mar10-llama7b`)
3. Optimize each bottleneck kernel in priority order
4. Verify end-to-end correctness and report total speedup

`program.md` is intentionally comprehensive so the agent can run 10+ hours without getting stuck. It includes a 6-tier optimization playbook, decision framework, crash handling, and Amdahl's law reasoning.

## The Pipeline

```
                 profile.py              extract.py           bench.py (loop)         verify.py
Any PyTorch  ──>  Rank kernels  ──>  Generate baseline  ──>  Optimize each  ──>  End-to-end
   model          by GPU time       Triton kernels          kernel (agent)       verification
```

| Tool | What it does |
|------|-------------|
| `profile.py` | Profiles any PyTorch model with `torch.profiler`, ranks kernels by GPU time, classifies as compute/memory-bound |
| `extract.py` | Extracts top-N bottleneck kernels from profiling results into standalone Triton kernel files |
| `orchestrate.py` | Multi-kernel scheduler: decides which kernel to optimize next using Amdahl's law, tracks aggregate progress |
| `bench.py` | Fixed benchmark: 5-stage correctness (smoke, shape sweep, numerical stability, determinism, edge cases) + performance + roofline |
| `verify.py` | Plugs optimized kernels back into the model, checks end-to-end correctness, reports total speedup |

## Supported Kernels

9 kernel types covering the core operations of modern deep learning:

| Kernel | Description | Key Metric |
|--------|-------------|------------|
| **matmul** | Dense matrix multiplication (M x K) @ (K x N) | TFLOPS |
| **softmax** | Row-parallel numerically stable softmax | GB/s |
| **layernorm** | Layer normalization with affine transform | GB/s |
| **rmsnorm** | RMS normalization (LLaMA-style) | GB/s |
| **flash_attention** | Scaled dot-product attention with causal masking | TFLOPS |
| **fused_mlp** | SwiGLU-style fused MLP (gate + up + down) | TFLOPS |
| **cross_entropy** | Fused cross entropy loss | GB/s |
| **rotary_embedding** | Rotary position embeddings (RoPE) | GB/s |
| **reduce** | Parallel reduction (sum) | GB/s |

Each has a PyTorch reference in `reference.py` and a starter Triton kernel in `kernels/`.

## Example Models

Self-contained model definitions ship with AutoKernel (no `transformers` library needed):

| Model | File | Params | Usage |
|-------|------|--------|-------|
| GPT-2 Small | `models/gpt2.py` | 124M | `--class-name GPT2 --input-shape 1,1024` |
| LLaMA (compact) | `models/llama_7b.py` | 160M | `--class-name LlamaModel --input-shape 1,512` |
| LLaMA 7B | `models/llama_7b.py` | 7B | `--class-name LlamaModel7B --input-shape 1,2048` |
| BERT-base | `models/bert_base.py` | 110M | `--class-name BertModel --input-shape 8,512` |
| Custom | `models/custom.py` | -- | Template for your own model |

For HuggingFace models (`uv sync --extra models`):

```bash
uv run profile.py --module transformers --class-name AutoModelForCausalLM \
 --pretrained meta-llama/Llama-2-7b-hf --input-shape 1,2048 --dtype float16
```

## Project Structure

```
autokernel/
  kernel.py             the file the agent modifies (one kernel at a time)
  program.md            agent instructions -- the "research org code"

  bench.py              fixed benchmark + 5-stage correctness harness
  reference.py          PyTorch reference implementations (ground truth)
  prepare.py            one-time setup: test data, baselines

  profile.py            profile any PyTorch model, rank kernels by GPU time
  extract.py            extract bottleneck kernels into workspace/
  orchestrate.py        multi-kernel scheduler (Amdahl's law)
  verify.py             end-to-end model verification + speedup report
  analysis.py           experiment visualization (generates progress.png)

  kernels/              starter Triton kernels (9 types)
  models/               self-contained model definitions (GPT-2, LLaMA, BERT)
  workspace/            runtime artifacts (gitignored)
```

## Design Choices

**Why Triton.** Readable Python-like syntax the agent can understand and modify without mastering inline PTX or SASS. Well-tuned Triton regularly reaches 80-95% of cuBLAS. The agent needs to iterate fast -- Triton compiles in seconds, not minutes.

**Correctness first.** The benchmark checks kernel output against PyTorch before measuring performance. A fast but wrong kernel is immediately reverted. This prevents the agent from "optimizing" by producing garbage.

**Amdahl's law orchestration.** The orchestrator prioritizes by impact. A 1.5x speedup on a 60% kernel (1.25x end-to-end) beats a 3x speedup on a 5% kernel (1.03x end-to-end). It moves on when diminishing returns set in.

**Single file to modify.** The agent only touches `kernel.py`. Scope stays manageable, diffs reviewable, reverts clean.

**TSV logging.** Results go to a plain `results.tsv` file. Human-readable, git-friendly, trivially parseable, no infrastructure.

## Results Format

Every experiment is logged to `results.tsv` (tab-separated):

| Column | Description |
|--------|-------------|
| `experiment` | Sequential experiment number (0 = baseline) |
| `tag` | Short identifier |
| `kernel_type` | Which kernel (e.g., `matmul`) |
| `throughput_tflops` | Measured throughput (higher is better) |
| `latency_us` | Execution time in microseconds |
| `pct_peak` | Percentage of GPU theoretical peak |
| `speedup_vs_pytorch` | Speedup vs PyTorch/cuBLAS |
| `correctness` | PASS, FAIL, TIMEOUT, or CRASH |
| `peak_vram_mb` | Peak GPU memory usage |
| `description` | What was tried |

## Credits

This project is **autoresearch for GPU kernels** -- directly inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), the original experiment in autonomous AI research agents for LLM training. Karpathy showed that an AI agent can run hundreds of experiments overnight, methodically exploring a search space and logging every result. AutoKernel applies that same loop -- agent edits one file, runs a fixed evaluation, keeps or reverts -- to the domain of GPU kernel optimization with Triton.

Built by the team behind [Forge](https://www.rightnowai.co/forge).

## License

MIT
