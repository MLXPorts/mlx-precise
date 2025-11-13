// Copyright © 2025 The Solace Project
// Extended Precision FFT Kernels for MLX
//
// Implements FFT operations with double-double arithmetic to eliminate
// intermediate rounding errors. Single rounding occurs only at output.

#include <metal_stdlib>
#include "double_double.metal"

using namespace metal;

// ============================================================================
// Complex Multiply (Frequency Domain) - Extended Precision
// ============================================================================

// This is the MOST CRITICAL operation for Hyena long convolution
// Standard float32 path rounds 6 times per multiply (4 muls + 2 adds)
// DD path rounds once at the final output
kernel void complex_multiply_extended(
    constant float4* u_freq [[buffer(0)]],      // Input spectrum (packed complex_dd)
    constant float4* k_freq [[buffer(1)]],      // Kernel spectrum (packed complex_dd)
    device float4* output [[buffer(2)]],        // Output spectrum (packed complex_dd)
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    // Unpack to complex DD
    complex_dd u = unpack_cdd(u_freq[gid]);
    complex_dd k = unpack_cdd(k_freq[gid]);

    // Multiply in extended precision (NO intermediate rounding)
    complex_dd result = cdd_mul(u, k);

    // Pack back (still in DD, no rounding yet)
    output[gid] = pack_cdd(result);
}

// Variant: Multiply and round to float32 immediately (single rounding point)
kernel void complex_multiply_extended_round(
    constant float4* u_freq [[buffer(0)]],
    constant float4* k_freq [[buffer(1)]],
    device float2* output [[buffer(2)]],        // Output as float2 (rounded)
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    complex_dd u = unpack_cdd(u_freq[gid]);
    complex_dd k = unpack_cdd(k_freq[gid]);

    complex_dd result = cdd_mul(u, k);

    // ⚠️ SINGLE ROUNDING POINT: DD → float32
    output[gid] = cdd_to_float2(result);
}

// ============================================================================
// FFT Radix-2 Butterfly (Extended Precision)
// ============================================================================

// Cooley-Tukey radix-2 butterfly in DD arithmetic
// Input: x[0], x[1] (complex DD)
// Output: X[0] = x[0] + x[1], X[1] = x[0] - x[1]
inline void radix2_dd(thread complex_dd* x, thread complex_dd* y) {
    y[0] = cdd_add(x[0], x[1]);
    y[1] = cdd_sub(x[0], x[1]);
}

// Radix-2 butterfly with twiddle factor
// X[0] = x[0] + W*x[1]
// X[1] = x[0] - W*x[1]
inline void radix2_twiddle_dd(
    thread complex_dd* x,
    thread complex_dd* y,
    complex_dd twiddle
) {
    complex_dd t = cdd_mul(x[1], twiddle);
    y[0] = cdd_add(x[0], t);
    y[1] = cdd_sub(x[0], t);
}

// ============================================================================
// Small FFT in Extended Precision (for testing/reference)
// ============================================================================

// 8-point FFT entirely in DD arithmetic
// This is a reference implementation; full FFT uses multi-pass for large N
kernel void fft8_extended(
    constant float4* input [[buffer(0)]],   // 8 complex_dd values (packed as float4)
    device float4* output [[buffer(1)]],    // 8 complex_dd values
    constant bool& inverse [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;  // Single-threaded for this small FFT

    // Unpack input
    complex_dd x[8];
    for (int i = 0; i < 8; i++) {
        x[i] = unpack_cdd(input[i]);
    }

    // Stage 1: 4 radix-2 butterflies
    complex_dd temp[8];
    for (int i = 0; i < 4; i++) {
        radix2_dd(&x[i*2], &temp[i*2]);
    }

    // Stage 2: 2 radix-4 groups with twiddles
    complex_dd stage2[8];
    for (int group = 0; group < 2; group++) {
        for (int i = 0; i < 2; i++) {
            int k = group * 4 + i;
            complex_dd w = twiddle_dd(i * group, 4);
            radix2_twiddle_dd(&temp[k], &stage2[k], w);
        }
    }

    // Stage 3: Final radix-8 with twiddles
    complex_dd result[8];
    for (int i = 0; i < 4; i++) {
        complex_dd w = twiddle_dd(i, 8);
        radix2_twiddle_dd(&stage2[i], &result[i], w);
    }

    // Apply inverse scaling if needed (1/N in DD)
    if (inverse) {
        double_double scale = dd_div(dd_constants::ONE, 8.0f);
        for (int i = 0; i < 8; i++) {
            result[i] = cdd_mul(result[i], scale);
        }
    }

    // Pack output (still in DD, no rounding)
    for (int i = 0; i < 8; i++) {
        output[i] = pack_cdd(result[i]);
    }
}

// ============================================================================
// Lift float32 FFT to Extended Precision
// ============================================================================

// Convert float32 complex array to DD complex array
kernel void lift_to_dd(
    constant float2* input [[buffer(0)]],   // float32 complex
    device float4* output [[buffer(1)]],    // DD complex (packed)
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    float2 z = input[gid];
    complex_dd zdd = complex_dd(z);  // Lift: set hi=value, lo=0
    output[gid] = pack_cdd(zdd);
}

// Round DD complex array to float32 complex array
kernel void round_to_float32(
    constant float4* input [[buffer(0)]],   // DD complex (packed)
    device float2* output [[buffer(1)]],    // float32 complex
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    complex_dd z = unpack_cdd(input[gid]);
    output[gid] = cdd_to_float2(z);  // ⚠️ ROUNDING HAPPENS HERE
}

// ============================================================================
// Scaling with Extended Precision
// ============================================================================

// Apply normalization factor (1/N) in DD
kernel void scale_by_inverse_dd(
    constant float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& n [[buffer(3)]],        // N for normalization
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    // Compute 1/N in DD (avoids float32 rounding on large N)
    double_double scale = dd_div(dd_constants::ONE, n);

    complex_dd z = unpack_cdd(input[gid]);
    z = cdd_mul(z, scale);
    output[gid] = pack_cdd(z);
}

// Scale and round in one step
kernel void scale_and_round_dd(
    constant float4* input [[buffer(0)]],
    device float2* output [[buffer(1)]],    // float32 output
    constant uint& length [[buffer(2)]],
    constant float& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    double_double scale = dd_div(dd_constants::ONE, n);

    complex_dd z = unpack_cdd(input[gid]);
    z = cdd_mul(z, scale);

    output[gid] = cdd_to_float2(z);  // ⚠️ SINGLE ROUNDING
}

// ============================================================================
// Depthwise Convolution (Extended Precision)
// ============================================================================

// 3-tap depthwise convolution in DD
// This is used for Hyena's gated stream (v)
// y[i] = w[0]*x[i-1] + w[1]*x[i] + w[2]*x[i+1]
kernel void depthwise3_extended(
    constant float* input [[buffer(0)]],    // Input signal (float32)
    constant float* weights [[buffer(1)]],  // 3 weights (float32)
    device float* output [[buffer(2)]],     // Output (float32)
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    // Lift weights to DD once
    double_double w0 = double_double(weights[0]);
    double_double w1 = double_double(weights[1]);
    double_double w2 = double_double(weights[2]);

    // Accumulate in DD (no intermediate rounding)
    double_double acc = dd_constants::ZERO;

    // Left tap (with boundary handling)
    if (gid > 0) {
        acc = dd_add(acc, dd_mul(w0, input[gid - 1]));
    }

    // Center tap
    acc = dd_add(acc, dd_mul(w1, input[gid]));

    // Right tap (with boundary handling)
    if (gid < length - 1) {
        acc = dd_add(acc, dd_mul(w2, input[gid + 1]));
    }

    // ⚠️ SINGLE ROUNDING: DD → float32
    output[gid] = dd_to_float(acc);
}

// ============================================================================
// Dot Product (Extended Precision)
// ============================================================================

// Deterministic dot product with serial accumulation
// Used for Linear layers in extended precision mode
kernel void dot_product_extended_serial(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;  // Single thread for serial accumulation

    // Serial accumulation in DD (completely deterministic)
    double_double acc = dd_constants::ZERO;
    for (uint i = 0; i < length; i++) {
        double_double prod = two_prod(a[i], b[i]);
        acc = dd_add(acc, prod);
    }

    // ⚠️ SINGLE ROUNDING
    output[0] = dd_to_float(acc);
}

// Parallel dot product with pairwise reduction (faster, still deterministic)
kernel void dot_product_extended_pairwise(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    device float4* partial_sums [[buffer(2)]],  // DD partial sums
    constant uint& length [[buffer(3)]],
    constant uint& chunk_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float4 shared[256];  // Threadgroup storage for DD values

    // Each thread accumulates a chunk in DD
    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, length);

    double_double local_sum = dd_constants::ZERO;
    for (uint i = start; i < end; i++) {
        double_double prod = two_prod(a[i], b[i]);
        local_sum = dd_add(local_sum, prod);
    }

    // Store in threadgroup memory
    shared[tid] = pack_cdd(complex_dd(local_sum, dd_constants::ZERO));

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pairwise reduction (deterministic)
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            complex_dd a = unpack_cdd(shared[tid]);
            complex_dd b = unpack_cdd(shared[tid + stride]);
            shared[tid] = pack_cdd(complex_dd(dd_add(a.re, b.re), dd_constants::ZERO));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial sum (still in DD)
    if (tid == 0) {
        partial_sums[gid / threads_per_group] = shared[0];
    }
}

// ============================================================================
// Real-to-Complex FFT Lifting (for rfft entry point)
// ============================================================================

// Lift real float32 array to complex DD for extended-precision rfft
kernel void lift_real_to_complex_dd(
    constant float* input [[buffer(0)]],    // Real float32
    device float4* output [[buffer(1)]],    // Complex DD
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    // Lift real value to complex DD: (x, 0) with x in DD
    complex_dd z = complex_dd(double_double(input[gid]), dd_constants::ZERO);
    output[gid] = pack_cdd(z);
}

// Extract real part from complex DD and round to float32
kernel void extract_real_from_complex_dd(
    constant float4* input [[buffer(0)]],   // Complex DD
    device float* output [[buffer(1)]],     // Real float32
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    complex_dd z = unpack_cdd(input[gid]);
    output[gid] = dd_to_float(z.re);  // ⚠️ ROUNDING
}

// ============================================================================
// Testing and Validation Utilities
// ============================================================================

// Compute error between DD and float32 reference
kernel void compute_precision_error(
    constant float4* dd_values [[buffer(0)]],
    constant float2* f32_values [[buffer(1)]],
    device float* errors [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;

    complex_dd z_dd = unpack_cdd(dd_values[gid]);
    float2 z_f32 = f32_values[gid];

    float2 z_dd_rounded = cdd_to_float2(z_dd);

    float err_re = abs(z_dd_rounded.x - z_f32.x);
    float err_im = abs(z_dd_rounded.y - z_f32.y);

    errors[gid] = max(err_re, err_im);
}
