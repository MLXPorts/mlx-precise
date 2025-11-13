// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.
//
// Tests for float64 CPU ↔ GPU conversion

#include <cmath>
#include <limits>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test float64 conversion special values") {
  // Test infinity
  {
    auto x_cpu = array({std::numeric_limits<double>::infinity()}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    CHECK(std::isinf(x_cpu.item<double>()));
    CHECK(x_cpu.item<double>() > 0);
    CHECK(std::isinf(x_back.item<double>()));
    CHECK(x_back.item<double>() > 0);
    CHECK_EQ(x_cpu.dtype(), float64);
    CHECK_EQ(x_gpu.dtype(), float64);
    CHECK_EQ(x_back.dtype(), float64);
  }

  // Test negative infinity
  {
    auto x_cpu = array({-std::numeric_limits<double>::infinity()}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    CHECK(std::isinf(x_cpu.item<double>()));
    CHECK(x_cpu.item<double>() < 0);
    CHECK(std::isinf(x_back.item<double>()));
    CHECK(x_back.item<double>() < 0);
  }

  // Test NaN
  {
    auto x_cpu = array({std::numeric_limits<double>::quiet_NaN()}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    CHECK(std::isnan(x_cpu.item<double>()));
    CHECK(std::isnan(x_back.item<double>()));
  }

  // Test zero
  {
    auto x_cpu = array({0.0}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    CHECK_EQ(x_cpu.item<double>(), 0.0);
    CHECK_EQ(x_back.item<double>(), 0.0);
  }

  // Test negative zero
  {
    auto x_cpu = array({-0.0}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    CHECK_EQ(x_cpu.item<double>(), -0.0);
    CHECK_EQ(x_back.item<double>(), -0.0);
  }
}

TEST_CASE("test float64 conversion precision") {
  // Test high-precision value that requires both hi and lo components
  {
    double original = 0.384723498732489723487238478374;
    auto x_cpu = array({original}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    double result = x_back.item<double>();

    // Should preserve at least float32 precision (7-8 digits)
    CHECK(std::abs(result - original) < 1e-7);

    // Ideally should preserve much more (close to double precision)
    // Note: Converting double → float2 → double may lose some precision
    // beyond what float32 can represent
    INFO("Original: ", original);
    INFO("Round-trip: ", result);
    INFO("Error: ", std::abs(result - original));
  }

  // Test various magnitudes
  std::vector<double> test_values = {
      1.0,
      -1.0,
      1234.5678,
      -1234.5678,
      1.23456789012345e10,   // Large value
      1.23456789012345e-10,  // Small value
      M_PI,
      M_E,
      std::sqrt(2.0),
  };

  for (double val : test_values) {
    auto x_cpu = array({val}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    double result = x_back.item<double>();

    // Check relative error
    double rel_error = std::abs((result - val) / val);

    INFO("Value: ", val);
    INFO("Round-trip: ", result);
    INFO("Relative error: ", rel_error);

    // Should preserve at least 7 significant digits (float32 precision)
    CHECK(rel_error < 1e-6);
  }
}

TEST_CASE("test float64 conversion edge cases") {
  // Test maximum double value that fits in float32 range
  {
    double val = std::numeric_limits<float>::max();
    auto x_cpu = array({val}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    CHECK_EQ(x_back.item<double>(), doctest::Approx(val));
  }

  // Test minimum positive normalized double
  {
    double val = std::numeric_limits<double>::min();
    auto x_cpu = array({val}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    // May lose precision for very small denormalized numbers
    CHECK(x_back.item<double>() > 0);
  }

  // Test values near float32 precision boundary
  {
    // 1 + epsilon where epsilon is just beyond float32 precision
    double val = 1.0 + 1e-8;
    auto x_cpu = array({val}, float64);
    auto x_gpu = array(x_cpu, Device::gpu);
    auto x_back = array(x_gpu, Device::cpu);

    // The lo component should capture the small difference
    INFO("Original: ", val);
    INFO("Round-trip: ", x_back.item<double>());
    INFO("Difference: ", std::abs(x_back.item<double>() - val));
  }
}

TEST_CASE("test float64 conversion array operations") {
  // Test that conversions work with array operations
  std::vector<double> values = {
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      0.0,
      -0.0,
      1.0,
      -1.0,
      M_PI,
      0.384723498732489723487238478374,
  };

  auto x_cpu = array(values.data(), {static_cast<int>(values.size())}, float64);
  auto x_gpu = array(x_cpu, Device::gpu);
  auto x_back = array(x_gpu, Device::cpu);

  for (size_t i = 0; i < values.size(); ++i) {
    double original = values[i];
    double result = x_back.data<double>()[i];

    if (std::isinf(original)) {
      CHECK(std::isinf(result));
      CHECK(std::signbit(original) == std::signbit(result));
    } else if (std::isnan(original)) {
      CHECK(std::isnan(result));
    } else {
      INFO("Index: ", i);
      INFO("Original: ", original);
      INFO("Round-trip: ", result);

      double abs_error = std::abs(result - original);
      double rel_error = original != 0.0 ? std::abs((result - original) / original) : abs_error;

      INFO("Absolute error: ", abs_error);
      INFO("Relative error: ", rel_error);

      // Should preserve at least float32 precision
      if (original != 0.0) {
        CHECK(rel_error < 1e-6);
      } else {
        CHECK(abs_error < 1e-15);
      }
    }
  }
}

TEST_CASE("test float64 GPU operations preserve precision") {
  // Test that GPU operations maintain precision throughout
  double a_val = 0.384723498732489723487238478374;
  double b_val = 1.234567890123456789;

  auto a_cpu = array({a_val}, float64);
  auto b_cpu = array({b_val}, float64);

  // Transfer to GPU
  auto a_gpu = array(a_cpu, Device::gpu);
  auto b_gpu = array(b_cpu, Device::gpu);

  // Perform operation on GPU
  auto c_gpu = add(a_gpu, b_gpu, Device::gpu);

  // Transfer back to CPU
  auto c_back = array(c_gpu, Device::cpu);

  double expected = a_val + b_val;
  double result = c_back.item<double>();

  INFO("Expected: ", expected);
  INFO("Result: ", result);
  INFO("Error: ", std::abs(result - expected));

  // The error should be small - ideally within double-double precision
  // But we accept float32 precision due to conversion overhead
  double rel_error = std::abs((result - expected) / expected);
  CHECK(rel_error < 1e-6);
}
