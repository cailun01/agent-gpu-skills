---
url: https://docs.nvidia.com/dl-cuda-graph/latest/index.html
---

# CUDA Graph Best Practice for PyTorch

Welcome to the comprehensive guide on using CUDA Graphs effectively with PyTorch!

## Overview

CUDA Graphs are a powerful feature that can significantly improve the performance of your PyTorch models by reducing CPU overhead and enabling more efficient GPU utilization. This guide provides best practices, examples, and tips for integrating CUDA Graphs into your deep learning workflows with PyTorch.

## What You’ll Learn

  * 📚 **CUDA Graph Basics** : Get familiar with CUDA Graphs fundamentals—what they are, how they work, and their benefits—without needing deep CUDA expertise

  * 🐍 **PyTorch CUDA Graphs** : Learn PyTorch’s integration, Megatron-LM implementations, and systematic best practices for adopting graphs

  * 💡 **Examples** : Real-world implementations from MLPerf Training benchmarks—Llama 3.1 405B, GPT-3 175B, Stable Diffusion v2, RNN-T, and more

  * 🔧 **Troubleshooting** : Debug capture failures, silent numerical errors, memory issues, and performance problems with comprehensive guides

  * 📖 **Reference** : Links to official documentation, tools, and additional learning resources


## How to Use This Guide

This guide serves different needs depending on where you are in your CUDA Graph journey:

### 📖 Learning Path: Systematic Understanding

If you’re new to CUDA Graphs and want to learn systematically:

  1. **Start with[ CUDA Graph Basics](cuda-graph-basics/introduction.html)**: Understand the problem (launch overhead) and solution (CUDA Graphs) without needing too much CUDA expertise

  2. **Read[ PyTorch Integration](torch-cuda-graph/torch-integration.html)**: Learn how PyTorch implements CUDA Graphs

  3. **Follow[ Best Practices](torch-cuda-graph/best-practices.html)**: Apply systematic approach to adopt CUDA Graphs in your application

  4. **Consult[ Writing Sync-Free Code](torch-cuda-graph/sync-free-code.html)** and **[Handling Dynamic Patterns](torch-cuda-graph/handling-dynamic-patterns.html)** : Learn to write compatible code

  5. **Verify with[ Troubleshooting](troubleshooting/debugging-strategies.html)**: Debug any issues that arise


### 🔍 Quick Reference: Problem Solving

If you’re encountering specific issues or need quick answers:

  * **Pre-capture checklist?** → [Quick Checklist](torch-cuda-graph/quick-checklist.html)

  * **Capture failing?** → [Capture Failures](troubleshooting/capture-failures.html)

  * **Wrong results?** → [Numerical Errors](troubleshooting/numerical-errors.html)

  * **Out of memory or illegal memory access?** → [Memory Issues](troubleshooting/memory-issues.html)

  * **Process hanging?** → [Process Hang](troubleshooting/process-hang.html)

  * **Poor performance?** → [Performance Issues](troubleshooting/performance-issues.html)

  * **Need to handle specific pattern?** → [Handling Dynamic Patterns](torch-cuda-graph/handling-dynamic-patterns.html)


Use the search function (top of page) to find specific topics quickly.

### ⚡ Framework-Specific Paths

**For Megatron-LM/Transformer Engine users** : Skip directly to [Transformer Engine and Megatron-LM CUDA Graphs](torch-cuda-graph/te-megatron-cuda-graphs.html) to learn about `make_graphed_callables` (TE), `CudaGraphManager`, and `FullCudaGraphWrapper`.

**For PyTorch users** : See [PyTorch Integration](torch-cuda-graph/torch-integration.html) for `torch.cuda.graph`, `make_graphed_callables`, and automatic CUDA Graphs with `torch.compile`.

## Quick Start
    
    
    import torch
    
    # Enable CUDA Graph mode in PyTorch
    model = YourModel().cuda()
    static_input = torch.randn(32, 3, 224, 224, device='cuda')
    
    # Warmup
    for _ in range(3):
        _ = model(static_input)
    
    # Capture the graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(static_input)
    
    # Training loop - replay the graph
    for data in dataloader:
        static_input.copy_(data)  # Update input in-place
        g.replay()                # Execute captured operations
    

## Why Use CUDA Graphs?

Tip

Performance Benefits

  * **Reduced CPU overhead** : Eliminate runtime overhead including kernel launch overhead

  * **Better GPU utilization** : Reduce GPU idle time—less waiting for CPU, fewer gaps between kernel executions

  * **Lower latency** : Faster inference and training iterations

  * **Predictable performance** : Consistent execution times with reduced run-to-run and rank-to-rank jitter


Warning

Important Considerations

  * Not all operations are graph-compatible—some require workarounds or must remain in eager mode

  * Adopting CUDA graphs may require effort to make existing workloads compatible

  * Trade-off between development effort and performance gains varies by workload


## Next Steps

CUDA Graph Basics

Understand what CUDA Graphs are and why they matter.

[Introduction](cuda-graph-basics/introduction.html)

PyTorch Integration

Learn how PyTorch integrates CUDA Graphs.

[PyTorch CUDA Graph Integration](torch-cuda-graph/torch-integration.html)

Examples

Real-world CUDA Graph implementations from MLPerf Training.

[Examples](examples/introduction.html)

Troubleshooting

Debug common CUDA graph issues.

[Debugging Strategies](troubleshooting/debugging-strategies.html)