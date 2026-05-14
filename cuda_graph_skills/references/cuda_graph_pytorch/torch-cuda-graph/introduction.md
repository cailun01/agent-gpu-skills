---
url: https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/introduction.html
---

# PyTorch CUDA Graphs

Note

This section provides comprehensive guidance for using CUDA Graphs in PyTorch, covering integration, best practices, and common challenges.

This chapter focuses on **using CUDA Graphs in PyTorch**. While previous chapters covered CUDA graph fundamentals, this section addresses the practical challenges of integrating graphs into real-world PyTorch code: eliminating synchronization, handling dynamic patterns, and navigating PyTorch-specific constraints.

## What This Chapter Covers

🐍 PyTorch Integration

PyTorch’s CUDA graph APIs (`torch.cuda.CUDAGraph`, `torch.cuda.graph()`, `torch.cuda.make_graphed_callables()`) and basic usage patterns.

[PyTorch CUDA Graph Integration](torch-integration.html)

⚡ Transformer Engine & Megatron-LM

Megatron-LM’s built-in CUDA graph support (`CudaGraphManager` and `FullCudaGraphWrapper`) for distributed training.

[Transformer Engine and Megatron-LM CUDA Graph Support](te-megatron-cuda-graphs.html)

✅ Best Practices

Systematic approach to adopting CUDA graphs (quantifying benefits, choosing scope, verifying correctness, and writing compatible code).

[Best Practices for PyTorch CUDA Graphs](best-practices.html)

🔇 Writing Sync-Free Code

Eliminating CPU-GPU synchronization—a prerequisite for CUDA graph capture.

[Writing Sync-Free Code](sync-free-code.html)

🔄 Handling Dynamic Patterns

Adapting dynamic control flow, tensors, scalars, and shapes to static graph execution.

[Handling Dynamic Patterns](handling-dynamic-patterns.html)

✔️ Quick Checklist

Pre-capture checklist to verify your code is CUDA Graph compatible.

[Quick Checklist](quick-checklist.html)

## Quick Navigation

🎯 Your Situation | 📍 Start Here  
---|---  
New to CUDA graphs in PyTorch | [PyTorch Integration](torch-integration.html)  
Ready to capture | [Quick Checklist](quick-checklist.html)  
Capture fails with sync errors | [Writing Sync-Free Code](sync-free-code.html)  
Wrong results or model divergence | [Handling Dynamic Patterns](handling-dynamic-patterns.html)  
Need adoption strategy guidance | [Best Practices](best-practices.html)  
Debugging capture/runtime failures | [Troubleshooting](../troubleshooting/introduction.html)  
  
## What’s Next?

Begin with **[PyTorch Integration](torch-integration.html)** to learn PyTorch’s CUDA graph APIs and integration patterns.