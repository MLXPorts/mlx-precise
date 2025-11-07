// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.
//
// Float64 copy kernels with double-double conversion

#pragma once

#include "mlx/backend/metal/kernels/double_double.h"

// Convert native double to float2 (double-double representation)
inline float2 double_to_dd(double x) {
  // Handle special values
  if (isinf(x) || isnan(x)) {
    float val = static_cast<float>(x);
    return float2(val, 0.0f);
  }

  // Split double into hi and lo components
  // hi = round(x) to nearest float32
  float hi = static_cast<float>(x);

  // lo = error term (what was lost in the rounding)
  // Computed in double precision for accuracy
  double x_hi = static_cast<double>(hi);
  double error = x - x_hi;
  float lo = static_cast<float>(error);

  return float2(hi, lo);
}

// Convert float2 (double-double) back to native double
inline double dd_to_double(float2 x) {
  // Handle special values
  if (isinf(x.x) || isnan(x.x)) {
    return static_cast<double>(x.x);
  }

  // Reconstruct double from hi and lo components
  // Using double precision for the addition
  return static_cast<double>(x.x) + static_cast<double>(x.y);
}

// Copy double (CPU) to float2 (GPU double-double)
template <int N = 1>
[[kernel]] void copy_v_double_float64(
    device const double* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      dst[index + i] = double_to_dd(src[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      dst[index + i] = double_to_dd(src[index + i]);
    }
  }
}

// Copy float2 (GPU double-double) to double (CPU)
template <int N = 1>
[[kernel]] void copy_v_float64_double(
    device const float2* src [[buffer(0)]],
    device double* dst [[buffer(1)]],
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      dst[index + i] = dd_to_double(src[index + i]);
    }
  } else {
    for (int i = 0; i < N; ++i) {
      dst[index + i] = dd_to_double(src[index + i]);
    }
  }
}

// Copy float2 to float2 (GPU to GPU) - no conversion needed
template <int N = 1>
[[kernel]] void copy_v_float64_float64(
    device const float2* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      dst[index + i] = src[index + i];
    }
  } else {
    for (int i = 0; i < N; ++i) {
      dst[index + i] = src[index + i];
    }
  }
}

// Scalar copy variants
template <int N = 1>
[[kernel]] void copy_s_double_float64(
    device const double* src [[buffer(0)]],
    device float2* dst [[buffer(1)]],
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  auto val = double_to_dd(src[0]);
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      dst[index + i] = val;
    }
  } else {
    for (int i = 0; i < N; ++i) {
      dst[index + i] = val;
    }
  }
}

template <int N = 1>
[[kernel]] void copy_s_float64_double(
    device const float2* src [[buffer(0)]],
    device double* dst [[buffer(1)]],
    constant uint& size,
    uint index [[thread_position_in_grid]]) {
  auto val = dd_to_double(src[0]);
  index *= N;
  if (N > 1 && index + N > size) {
    for (int i = 0; index + i < size; ++i) {
      dst[index + i] = val;
    }
  } else {
    for (int i = 0; i < N; ++i) {
      dst[index + i] = val;
    }
  }
}
