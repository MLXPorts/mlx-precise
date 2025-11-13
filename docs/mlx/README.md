# MLX Metal Kernel Documentation

This directory contains curated documentation for writing high-performance Metal kernels in MLX.

## Getting Started

1. **[MetalKernel-Patterns.md](MetalKernel-Patterns.md)** - Start here for the correct MLX kernel pattern (header + body-only source)
2. **[DEVICES_STREAMS.md](DEVICES_STREAMS.md)** - Understanding devices and streams for overlap

## Performance Optimization

3. **[WWDC16-Optimization-Patterns.md](WWDC16-Optimization-Patterns.md)** ⚡ - Critical performance patterns from Apple WWDC16
   - **Key warning**: Dynamically-indexed stack arrays can lose 30% performance
   - FMA usage, data types, control flow, memory access patterns

4. **[Comprehensive-MLX-Metal-Guide.md](Comprehensive-MLX-Metal-Guide.md)** - Production-tested patterns for QR/SVD/GEMM kernels

## Streams and Concurrency

5. **[Streams-Guide.md](Streams-Guide.md)** - Practical stream usage for overlap and concurrency
6. **[Streams-and-Banding.md](Streams-and-Banding.md)** - When and how to use banded execution

## Common Issues

7. **[COMMON_PITFALLS.md](COMMON_PITFALLS.md)** - Sharp edges and how to avoid them
8. **[PYTORCH_DISSONANCE.md](PYTORCH_DISSONANCE.md)** - PyTorch to MLX translation guide

## Quick Reference

### Kernel Structure
```python
_HEADER = """#include <metal_stdlib>
using namespace metal;
// Helper functions here
"""

_BODY = r"""
    uint gid = thread_position_in_grid.x;
    // Body only - NO kernel signature
"""

kernel = mx.fast.metal_kernel(
    name="kernel_name",
    header=_HEADER,
    source=_BODY,
    input_names=["input", "length"],
    output_names=["output"],
    ensure_row_contiguous=True
)
```

### Critical Performance Rules
- ⚠️ **NO dynamically-indexed stack arrays** (can lose 30% performance)
- Use `fma(a, b, c)` everywhere
- Eliminate integer division (use shifts for powers of 2)
- Branchless ternary for uniform control flow
- Barriers **before** reads, not after
- Compile once, cache globally
- Pass params as buffers, not string templates

### Stream Usage
```python
s_gpu = mx.new_stream(mx.gpu)

with mx.stream(s_gpu):
    result = mx.matmul(a, b)

mx.synchronize(s_gpu)  # Only at boundaries
```

## Source

These documents are curated from production xLSTM and MetalFaiss implementations, incorporating insights from:
- Apple WWDC16 Session 606: "Advanced Metal Shader Optimization"
- MLX official documentation
- Production kernel development experience

---

*For implementation examples, see `ember_ml/backend/mlx/linearalg/`*
