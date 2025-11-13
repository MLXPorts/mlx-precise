# MLX Metal Kernel Documentation Index

Curated documentation for writing high-performance Metal kernels in MLX.

## Documentation Files

| File | Description | Priority |
|------|-------------|----------|
| [README.md](README.md) | Quick start and overview | ⭐ Start Here |
| [MetalKernel-Patterns.md](MetalKernel-Patterns.md) | Correct MLX kernel pattern (header + body) | ⭐ Essential |
| [WWDC16-Optimization-Patterns.md](WWDC16-Optimization-Patterns.md) | Critical performance patterns from Apple | 🔥 Performance |
| [Comprehensive-MLX-Metal-Guide.md](Comprehensive-MLX-Metal-Guide.md) | Production-tested kernel patterns | 🔥 Performance |
| [DEVICES_STREAMS.md](DEVICES_STREAMS.md) | Device and stream management | ⭐ Essential |
| [Streams-Guide.md](Streams-Guide.md) | Practical stream usage for overlap | 💡 Concurrency |
| [Streams-and-Banding.md](Streams-and-Banding.md) | Banded execution patterns | 💡 Concurrency |
| [COMMON_PITFALLS.md](COMMON_PITFALLS.md) | Common mistakes and fixes | ⚠️ Debugging |
| [PYTORCH_DISSONANCE.md](PYTORCH_DISSONANCE.md) | PyTorch → MLX translation | 💡 Reference |

## Learning Path

### Beginner
1. Read [README.md](README.md) for overview
2. Study [MetalKernel-Patterns.md](MetalKernel-Patterns.md) for correct kernel structure
3. Review [DEVICES_STREAMS.md](DEVICES_STREAMS.md) for basic stream usage

### Intermediate
4. Apply patterns from [Comprehensive-MLX-Metal-Guide.md](Comprehensive-MLX-Metal-Guide.md)
5. Learn stream overlap from [Streams-Guide.md](Streams-Guide.md)
6. Avoid mistakes in [COMMON_PITFALLS.md](COMMON_PITFALLS.md)

### Advanced
7. Master [WWDC16-Optimization-Patterns.md](WWDC16-Optimization-Patterns.md) for maximum performance
8. Implement banded execution from [Streams-and-Banding.md](Streams-and-Banding.md)

## Implementation Examples

See working implementations in:
- `ember_ml/backend/mlx/linearalg/math_ops.py` - Double-double arithmetic with error-free transformations
- `ember_ml/backend/mlx/linearalg/qr_ops.py` - Tiled QR decomposition
- `ember_ml/backend/mlx/linearalg/hpc16x8_ops.py` - HPC precision kernels

## Quick Reference Card

```python
# Kernel Structure
_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_BODY = r"""
    uint gid = thread_position_in_grid.x;
    // Body only
"""

kernel = mx.fast.metal_kernel(
    name="name",
    header=_HEADER,
    source=_BODY,
    input_names=[...],
    output_names=[...],
    ensure_row_contiguous=True
)

# Stream Usage
s_gpu = mx.new_stream(mx.gpu)
with mx.stream(s_gpu):
    result = mx.matmul(a, b)
mx.synchronize(s_gpu)

# Lazy Compilation
_KERNEL = None
def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(...)
    return _KERNEL
```

## Performance Checklist

- [ ] No dynamically-indexed stack arrays
- [ ] Using `fma()` in hot loops
- [ ] Integer division eliminated
- [ ] Branchless ternary control flow
- [ ] Barriers before reads
- [ ] Kernels compiled once, cached globally
- [ ] Params passed as buffers
- [ ] Memory access coalesced
- [ ] Threadgroup size multiple of 32
- [ ] Streams used for overlap

---

*Source: Production xLSTM and MetalFaiss implementations + Apple WWDC16 Session 606*
