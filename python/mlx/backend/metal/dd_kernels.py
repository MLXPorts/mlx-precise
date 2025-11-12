"""
Double-Double (Extended Precision) Metal Kernels for MLX

Implements numerically precise operations using double-double arithmetic
to eliminate intermediate rounding errors. Based on patterns from xLSTM-metal.

Key operations:
- Complex multiply in DD (critical for FFT-based convolutions)
- FFT scaling in DD (eliminates rounding on large N)
- Depthwise convolution in DD
- Dot product with DD accumulation

References:
- Dekker (1971): A floating-point technique for extending the available precision
- Knuth TAOCP Vol 2: Seminumerical Algorithms
- xLSTM-metal kernel_development patterns
"""

from __future__ import annotations

import math
import os
from typing import Tuple, Optional

import mlx.core as mx

# Metal header with DD arithmetic primitives
_HEADER = """
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Double-Double Representation and Error-Free Transformations
// ============================================================================

// Two-Sum: Exact sum with error term (Knuth)
inline float2 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return float2(s, e);  // (sum, error)
}

// Two-Product: Exact product with FMA
inline float2 two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);  // Error: a*b - round(a*b)
    return float2(p, e);
}

// Quick-Two-Sum: Optimized when |a| >= |b|
inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return float2(s, e);
}

// ============================================================================
// Double-Double Operations
// ============================================================================

// DD addition
inline float2 dd_add(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);  // High order sum
    float2 t = two_sum(a.y, b.y);  // Low order sum

    // Normalize: collect error terms
    s.y += t.x;
    s = quick_two_sum(s.x, s.y);
    s.y += t.y;
    s = quick_two_sum(s.x, s.y);

    return s;
}

// DD subtraction
inline float2 dd_sub(float2 a, float2 b) {
    return dd_add(a, float2(-b.x, -b.y));
}

// DD multiplication
inline float2 dd_mul(float2 a, float2 b) {
    float2 p = two_prod(a.x, b.x);  // High * High

    // Add cross terms
    p.y += a.x * b.y + a.y * b.x;
    p = quick_two_sum(p.x, p.y);

    return p;
}

// DD division by float
inline float2 dd_div_f(float2 a, float b) {
    float q = a.x / b;
    float2 p = two_prod(q, b);
    float e = (a.x - p.x - p.y + a.y) / b;
    return quick_two_sum(q, e);
}

// ============================================================================
// Complex Double-Double Operations (float4: re_hi, re_lo, im_hi, im_lo)
// ============================================================================

// Complex DD addition
inline float4 cdd_add(float4 a, float4 b) {
    float2 re = dd_add(a.xy, b.xy);
    float2 im = dd_add(a.zw, b.zw);
    return float4(re.x, re.y, im.x, im.y);
}

// Complex DD subtraction
inline float4 cdd_sub(float4 a, float4 b) {
    float2 re = dd_sub(a.xy, b.xy);
    float2 im = dd_sub(a.zw, b.zw);
    return float4(re.x, re.y, im.x, im.y);
}

// Complex DD multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
// THIS IS THE CRITICAL OPERATION FOR FFT PRECISION
inline float4 cdd_mul(float4 a, float4 b) {
    // ac = a.re * b.re
    float2 ac = dd_mul(a.xy, b.xy);

    // bd = a.im * b.im
    float2 bd = dd_mul(a.zw, b.zw);

    // Real part: ac - bd
    float2 re = dd_sub(ac, bd);

    // ad = a.re * b.im
    float2 ad = dd_mul(a.xy, b.zw);

    // bc = a.im * b.re
    float2 bc = dd_mul(a.zw, b.xy);

    // Imaginary part: ad + bc
    float2 im = dd_add(ad, bc);

    return float4(re.x, re.y, im.x, im.y);
}

// Complex DD multiply by DD scalar
inline float4 cdd_mul_scalar(float4 a, float2 s) {
    float2 re = dd_mul(a.xy, s);
    float2 im = dd_mul(a.zw, s);
    return float4(re.x, re.y, im.x, im.y);
}

// Round DD to single float (SINGLE ROUNDING POINT)
inline float dd_to_float(float2 a) {
    return a.x + a.y;
}

// Round complex DD to float2
inline float2 cdd_to_float2(float4 a) {
    return float2(dd_to_float(a.xy), dd_to_float(a.zw));
}

// Lift float to DD
inline float2 float_to_dd(float x) {
    return float2(x, 0.0f);
}

// Lift float2 (complex) to DD complex
inline float4 float2_to_cdd(float2 z) {
    return float4(z.x, 0.0f, z.y, 0.0f);
}
"""

# ============================================================================
# Kernel Builders
# ============================================================================

_KERNEL_COMPLEX_MUL = None
_KERNEL_FFT_SCALE = None
_KERNEL_DEPTHWISE3 = None
_KERNEL_LIFT_COMPLEX = None
_KERNEL_ROUND_COMPLEX = None


def _dd_length(array: mx.array) -> int:
    """Return number of complex elements processed by DD kernels."""
    if not array.shape:
        return 1

    if array.shape[-1] == 4:
        complex_shape = array.shape[:-1]
    else:
        complex_shape = array.shape

    # math.prod handles scalars (empty leading shape) by returning 1.
    return int(math.prod(complex_shape) if complex_shape else 1)


def _build_complex_multiply_kernel():
    """
    Build kernel for complex multiplication in extended precision.

    This is the MOST CRITICAL kernel for FFT-based operations.
    Standard float32: 6 rounding operations per multiply
    Extended DD: 1 rounding operation (or 0 if output left in DD)
    """
    source = """
    // Complex multiply in DD
    // Inputs: u_freq, k_freq (float4 = complex DD)
    // Output: result (float4 = complex DD) OR float2 if round_output=true

    uint idx = thread_position_in_grid.x;
    const uint total = length[0];
    if (idx >= total) return;

    const uint base_u = idx * 4;
    float4 u = float4(
        u_freq[base_u + 0],
        u_freq[base_u + 1],
        u_freq[base_u + 2],
        u_freq[base_u + 3]
    );

    const uint base_k = idx * 4;
    float4 k = float4(
        k_freq[base_k + 0],
        k_freq[base_k + 1],
        k_freq[base_k + 2],
        k_freq[base_k + 3]
    );

    // Multiply in extended precision (NO intermediate rounding)
    float4 result_dd = cdd_mul(u, k);

    // Write result (still in DD - rounding deferred)
    const uint base = idx * 4;
    result[base + 0] = result_dd.x;
    result[base + 1] = result_dd.y;
    result[base + 2] = result_dd.z;
    result[base + 3] = result_dd.w;
"""

    return mx.fast.metal_kernel(
        name="complex_multiply_dd",
        input_names=["u_freq", "k_freq", "length"],
        output_names=["result"],
        header=_HEADER,
        source=source
    )


def _build_fft_scale_kernel():
    """
    Build kernel for FFT scaling (1/N) in extended precision.

    Avoids float32 rounding on large N.
    """
    source = """
    uint idx = thread_position_in_grid.x;
    const uint total = length[0];
    if (idx >= total) return;

    const uint in_base = idx * 4;
    float4 z = float4(
        input[in_base + 0],
        input[in_base + 1],
        input[in_base + 2],
        input[in_base + 3]
    );

    // Compute 1/N in DD
    float n_val = n[0];
    float2 scale = dd_div_f(float2(1.0f, 0.0f), n_val);

    // Scale in DD (no intermediate rounding)
    float4 scaled = cdd_mul_scalar(z, scale);

    // Write result (still in DD)
    const uint out_base = idx * 4;
    output[out_base + 0] = scaled.x;
    output[out_base + 1] = scaled.y;
    output[out_base + 2] = scaled.z;
    output[out_base + 3] = scaled.w;
"""

    return mx.fast.metal_kernel(
        name="fft_scale_dd",
        input_names=["input", "n", "length"],
        output_names=["output"],
        header=_HEADER,
        source=source
    )


def _build_lift_complex_kernel():
    """Lift float2 (complex float32) to float4 (complex DD)."""
    source = """
    uint idx = thread_position_in_grid.x;
    const uint total = length[0];
    if (idx >= total) return;

    float2 z = input[idx];
    float4 dd_val = float2_to_cdd(z);
    const uint base = idx * 4;
    output[base + 0] = dd_val.x;
    output[base + 1] = dd_val.y;
    output[base + 2] = dd_val.z;
    output[base + 3] = dd_val.w;
"""

    return mx.fast.metal_kernel(
        name="lift_complex_to_dd",
        input_names=["input", "length"],
        output_names=["output"],
        header=_HEADER,
        source=source
    )


def _build_round_complex_kernel():
    """Round float4 (complex DD) to float2 (complex float32)."""
    source = """
    uint idx = thread_position_in_grid.x;
    const uint total = length[0];
    if (idx >= total) return;

    const uint base = idx * 4;
    float4 z = float4(
        input[base + 0],
        input[base + 1],
        input[base + 2],
        input[base + 3]
    );

    // ⚠️ SINGLE ROUNDING POINT
    output[idx] = cdd_to_float2(z);
"""

    return mx.fast.metal_kernel(
        name="round_complex_from_dd",
        input_names=["input", "length"],
        output_names=["output"],
        header=_HEADER,
        source=source
    )


def _build_depthwise3_kernel():
    """3-tap depthwise convolution with DD accumulation."""
    source = """
    uint idx = thread_position_in_grid.x;
    const uint total = length[0];
    if (idx >= total) return;

    // Lift weights to DD
    float2 w0 = float_to_dd(weights[0]);
    float2 w1 = float_to_dd(weights[1]);
    float2 w2 = float_to_dd(weights[2]);

    // Accumulate in DD (no intermediate rounding)
    float2 acc = float2(0.0f, 0.0f);

    // Left tap
    if (idx > 0) {
        float2 prod = dd_mul(w0, float_to_dd(input[idx - 1]));
        acc = dd_add(acc, prod);
    }

    // Center tap
    float2 prod = dd_mul(w1, float_to_dd(input[idx]));
    acc = dd_add(acc, prod);

    // Right tap
    if (idx < total - 1) {
        float2 prod = dd_mul(w2, float_to_dd(input[idx + 1]));
        acc = dd_add(acc, prod);
    }

    // ⚠️ SINGLE ROUNDING: DD → float32
    output[idx] = dd_to_float(acc);
"""

    return mx.fast.metal_kernel(
        name="depthwise3_dd",
        input_names=["input", "weights", "length"],
        output_names=["output"],
        header=_HEADER,
        source=source
    )


# ============================================================================
# Public API Functions
# ============================================================================

def complex_multiply_extended(
    u_freq: mx.array,
    k_freq: mx.array,
    round_output: bool = True
) -> mx.array:
    """
    Complex multiply in extended precision (frequency domain).

    This is the CRITICAL operation for Hyena long convolution.
    Standard float32: 6 rounding operations per multiply
    Extended DD: 1 rounding operation (or 0 if round_output=False)

    Args:
        u_freq: Input spectrum (complex64 or DD complex)
        k_freq: Kernel spectrum (complex64 or DD)
        round_output: If True, round to complex64; else return DD

    Returns:
        Product spectrum (complex64 if round_output, else DD)
    """
    global _KERNEL_COMPLEX_MUL

    # Handle dtype conversion if needed
    needs_lift = u_freq.dtype == mx.complex64
    if needs_lift:
        u_freq = lift_complex_to_dd(u_freq)
        k_freq = lift_complex_to_dd(k_freq)

    length = _dd_length(u_freq)
    length_array = mx.array([length], dtype=mx.uint32)

    # Build kernel on first use
    if _KERNEL_COMPLEX_MUL is None:
        _KERNEL_COMPLEX_MUL = _build_complex_multiply_kernel()

    output_shape = u_freq.shape
    output_dtype = mx.float32  # float4 storage

    # Launch kernel
    grid = (length, 1, 1)
    threadgroup = (min(256, length), 1, 1)

    (result_dd,) = _KERNEL_COMPLEX_MUL(
        inputs=[u_freq, k_freq, length_array],
        output_shapes=[output_shape],
        output_dtypes=[output_dtype],
        grid=grid,
        threadgroup=threadgroup
    )

    if round_output:
        return round_complex_from_dd(result_dd)
    return result_dd


def fft_scale_extended(
    x_freq: mx.array,
    n: int
) -> mx.array:
    """
    Scale FFT spectrum by 1/N in extended precision.

    Avoids float32 rounding on large N.

    Args:
        x_freq: Frequency spectrum (DD complex)
        n: FFT size for 1/N scaling

    Returns:
        Scaled spectrum (DD complex)
    """
    global _KERNEL_FFT_SCALE

    if _KERNEL_FFT_SCALE is None:
        _KERNEL_FFT_SCALE = _build_fft_scale_kernel()

    length = _dd_length(x_freq)
    length_array = mx.array([length], dtype=mx.uint32)
    n_array = mx.array([n], dtype=mx.float32)

    grid = (length, 1, 1)
    threadgroup = (min(256, length), 1, 1)

    (result,) = _KERNEL_FFT_SCALE(
        inputs=[x_freq, n_array, length_array],
        output_shapes=[x_freq.shape],
        output_dtypes=[x_freq.dtype],
        grid=grid,
        threadgroup=threadgroup
    )

    return result


def lift_complex_to_dd(x: mx.array) -> mx.array:
    """
    Lift complex64 to DD complex representation.

    Args:
        x: Complex array (complex64)

    Returns:
        DD complex array (stored as float32 with 4x size)
    """
    real = mx.real(x)
    imag = mx.imag(x)
    zeros = mx.zeros_like(real)
    return mx.stack([real, zeros, imag, zeros], axis=-1)


def round_complex_from_dd(x_dd: mx.array) -> mx.array:
    """
    Round DD complex to complex64 (SINGLE ROUNDING POINT).

    Args:
        x_dd: DD complex array

    Returns:
        complex64 array
    """
    real = x_dd[..., 0] + x_dd[..., 1]
    imag = x_dd[..., 2] + x_dd[..., 3]
    return real + 1j * imag


def depthwise3_extended(
    x: mx.array,
    weights: mx.array
) -> mx.array:
    """
    3-tap depthwise convolution with DD accumulation.

    Used for Hyena gated stream.

    Args:
        x: Input signal (..., L)
        weights: 3 convolution weights

    Returns:
        Convolved signal (..., L) in float32
    """
    global _KERNEL_DEPTHWISE3

    if _KERNEL_DEPTHWISE3 is None:
        _KERNEL_DEPTHWISE3 = _build_depthwise3_kernel()

    # Flatten to 1D for kernel, reshape after
    original_shape = x.shape
    x_flat = x.reshape(-1)
    length = x_flat.shape[0]
    length_array = mx.array([length], dtype=mx.uint32)

    grid = (length, 1, 1)
    threadgroup = (min(256, length), 1, 1)

    (result,) = _KERNEL_DEPTHWISE3(
        inputs=[x_flat, weights, length_array],
        output_shapes=[(length,)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup
    )

    return result.reshape(original_shape)


# ============================================================================
# Helper Functions
# ============================================================================

def get_kernel_info() -> dict:
    """Return information about compiled kernels."""
    return {
        "complex_multiply": _KERNEL_COMPLEX_MUL is not None,
        "fft_scale": _KERNEL_FFT_SCALE is not None,
        "depthwise3": _KERNEL_DEPTHWISE3 is not None,
        "lift_complex": _KERNEL_LIFT_COMPLEX is not None,
        "round_complex": _KERNEL_ROUND_COMPLEX is not None,
    }


def reset_kernels():
    """Reset all compiled kernels (useful for testing different configurations)."""
    global _KERNEL_COMPLEX_MUL, _KERNEL_FFT_SCALE, _KERNEL_DEPTHWISE3
    global _KERNEL_LIFT_COMPLEX, _KERNEL_ROUND_COMPLEX

    _KERNEL_COMPLEX_MUL = None
    _KERNEL_FFT_SCALE = None
    _KERNEL_DEPTHWISE3 = None
    _KERNEL_LIFT_COMPLEX = None
    _KERNEL_ROUND_COMPLEX = None
