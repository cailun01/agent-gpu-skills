---
url: https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/introduction.html
---

# Troubleshooting CUDA Graphs

Note

This section provides comprehensive guidance for debugging and resolving issues when working with CUDA Graphs in PyTorch.

## Overview

PyTorch with CUDA Graphs introduces a fundamentally different execution model—operations are captured once and replayed as a fixed unit. Issues that would be obvious in eager mode can manifest as silent failures, memory errors, or unexpected behavior in graphed code. This chapter provides systematic approaches to diagnose and fix these issues efficiently.

## What This Chapter Covers

🛠️ Debugging Strategies

Systematic approaches to debug CUDA graph issues (detection techniques, isolation strategies, debugging workflows).

[Debugging Strategies](debugging-strategies.html)

🚫 Capture Failures

Diagnosing and fixing errors during graph capture (constraint violations, common error messages, workarounds).

[Capture Failures](capture-failures.html)

🔢 Numerical Errors

Identifying wrong results, NaN/Inf issues, and numerical precision problems (tensor reassignment, uninitialized memory, gradient issues, RNG state).

[Numerical Errors](numerical-errors.html)

💾 Memory Issues

Debugging memory-related problems (OOM errors during capture/replay, memory leaks, pool management).

[Memory Issues](memory-issues.html)

⏳ Process Hang

Resolving hanging or stuck processes (deadlocks, NCCL conflicts, infinite loops, distributed training hangs).

[Process Hang](process-hang.html)

📉 Performance Issues

Diagnosing cases where graphs don’t deliver expected speedup (bottleneck identification, graph overhead, optimization opportunities).

[Performance Issues](performance-issues.html)

## Failure Modes

Failure Mode | Symptoms | When It Occurs | Debugging Difficulty  
---|---|---|---  
🛠️ **[Debugging Strategies](debugging-strategies.html)** | Any issue | All phases | N/A (methodology)  
🚫 **[Capture Failures](capture-failures.html)** | RuntimeError during capture | Capture phase | ⭐ Easy (immediate error)  
🔢 **[Numerical Errors](numerical-errors.html)** | Wrong results, NaN/Inf | Replay phase | ⭐⭐⭐ Hard (silent failures)  
💾 **[Memory Issues](memory-issues.html)** | OOM, allocation errors | Capture or replay | ⭐⭐ Medium (clear errors)  
⏳ **[Process Hang](process-hang.html)** | Freezes, no progress | Any phase | ⭐⭐⭐⭐ Very Hard (no error)  
📉 **[Performance Issues](performance-issues.html)** | Slower than expected | Replay phase | ⭐⭐ Medium (requires profiling)  
  
## What’s Next?

Start with **[Debugging Strategies](debugging-strategies.html)** for systematic approaches, or jump directly to a specific failure mode from the table above.

Tip

After Troubleshooting

  * ✅ **[Quick Checklist](../torch-cuda-graph/quick-checklist.html)** : Verify your code meets all requirements before capture

  * 📖 **[Best Practices](../torch-cuda-graph/best-practices.html)** : Proactive strategies to avoid issues

  * 💡 **[Examples](../examples/introduction.html)** : Real-world CUDA Graph implementations