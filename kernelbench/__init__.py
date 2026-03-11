"""
AutoKernel KernelBench Integration.

Bridge between AutoKernel's iterative optimization loop and the KernelBench
benchmark suite (ScalingIntelligence/KernelBench). Enables running 50-300+
refinement experiments per problem instead of one-shot LLM generation.

Components:
    bridge.py   -- Load, cache, and set up KernelBench problems
    bench_kb.py -- Evaluate ModelNew vs Model (correctness + speedup)
    scorer.py   -- Batch scoring across levels, compute fast_p metric
"""

__version__ = "1.0.0"
