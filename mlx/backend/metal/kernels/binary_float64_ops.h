// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.
//
// Float64 binary operations using double-double arithmetic

#pragma once

#include "mlx/backend/metal/kernels/double_double.h"

using namespace mlx::core::metal;

// Float64 is stored as float2 (hi, lo) in Metal
// This typedef matches the type name used in kernel instantiation
using float64_t = metal::float2;
using double_precision = metal::float2;  // Alias for kernel instantiation

// Convert float2 to double_double for computation
inline double_double to_dd(float64_t x) {
  return unpack_dd(x);
}

// Convert double_double back to float2 for storage
inline float64_t from_dd(double_double x) {
  return pack_dd(x);
}

// ============================================================================
// Float64 Binary Operations
// ============================================================================

struct AddFloat64 {
  float64_t operator()(float64_t x, float64_t y) {
    return from_dd(dd_add(to_dd(x), to_dd(y)));
  }
};

struct SubtractFloat64 {
  float64_t operator()(float64_t x, float64_t y) {
    return from_dd(dd_sub(to_dd(x), to_dd(y)));
  }
};

struct MultiplyFloat64 {
  float64_t operator()(float64_t x, float64_t y) {
    return from_dd(dd_mul(to_dd(x), to_dd(y)));
  }
};

struct DivideFloat64 {
  float64_t operator()(float64_t x, float64_t y) {
    // Full DD division: reciprocal + multiply
    // For now, use simple approach (can optimize later)
    auto dx = to_dd(x);
    auto dy = to_dd(y);

    // q = x.hi / y.hi (approximate quotient)
    float q = dx.hi / dy.hi;

    // r = x - q * y (remainder in DD)
    auto qy = dd_mul(double_double(q), dy);
    auto r = dd_sub(dx, qy);

    // Refine: q += r / y.hi
    float correction = dd_to_float(r) / dy.hi;
    return from_dd(double_double(q + correction));
  }
};

struct MaximumFloat64 {
  float64_t operator()(float64_t x, float64_t y) {
    auto dx = to_dd(x);
    auto dy = to_dd(y);
    // Compare hi parts first, then lo if needed
    bool x_greater = (dx.hi > dy.hi) || (dx.hi == dy.hi && dx.lo > dy.lo);
    return x_greater ? x : y;
  }
};

struct MinimumFloat64 {
  float64_t operator()(float64_t x, float64_t y) {
    auto dx = to_dd(x);
    auto dy = to_dd(y);
    bool x_less = (dx.hi < dy.hi) || (dx.hi == dy.hi && dx.lo < dy.lo);
    return x_less ? x : y;
  }
};

// Comparison operations (return bool)
struct EqualFloat64 {
  bool operator()(float64_t x, float64_t y) {
    return x.x == y.x && x.y == y.y;
  }
};

struct NotEqualFloat64 {
  bool operator()(float64_t x, float64_t y) {
    return x.x != y.x || x.y != y.y;
  }
};

struct LessFloat64 {
  bool operator()(float64_t x, float64_t y) {
    return (x.x < y.x) || (x.x == y.x && x.y < y.y);
  }
};

struct LessEqualFloat64 {
  bool operator()(float64_t x, float64_t y) {
    return (x.x < y.x) || (x.x == y.x && x.y <= y.y);
  }
};

struct GreaterFloat64 {
  bool operator()(float64_t x, float64_t y) {
    return (x.x > y.x) || (x.x == y.x && x.y > y.y);
  }
};

struct GreaterEqualFloat64 {
  bool operator()(float64_t x, float64_t y) {
    return (x.x > y.x) || (x.x == y.x && x.y >= y.y);
  }
};
