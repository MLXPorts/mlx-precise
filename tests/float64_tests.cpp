// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test float64 basic operations") {
  // Test CPU reference
  auto a_cpu = array({1.0, 2.0, 3.0}, float64);
  auto b_cpu = array({4.0, 5.0, 6.0}, float64);

  // Test GPU
  auto a_gpu = array({1.0, 2.0, 3.0}, float64);
  auto b_gpu = array({4.0, 5.0, 6.0}, float64);

  // Addition
  auto c_cpu = add(a_cpu, b_cpu, Device::cpu);
  auto c_gpu = add(a_gpu, b_gpu, Device::gpu);
  CHECK_EQ(c_gpu.dtype(), float64);
  CHECK(array_equal(c_gpu, c_cpu, Device::cpu).item<bool>());

  // Subtraction
  c_cpu = subtract(a_cpu, b_cpu, Device::cpu);
  c_gpu = subtract(a_gpu, b_gpu, Device::gpu);
  CHECK_EQ(c_gpu.dtype(), float64);
  CHECK(array_equal(c_gpu, c_cpu, Device::cpu).item<bool>());

  // Multiplication
  c_cpu = multiply(a_cpu, b_cpu, Device::cpu);
  c_gpu = multiply(a_gpu, b_gpu, Device::gpu);
  CHECK_EQ(c_gpu.dtype(), float64);
  CHECK(array_equal(c_gpu, c_cpu, Device::cpu).item<bool>());

  // Division
  c_cpu = divide(a_cpu, b_cpu, Device::cpu);
  c_gpu = divide(a_gpu, b_gpu, Device::gpu);
  CHECK_EQ(c_gpu.dtype(), float64);
  CHECK(array_equal(c_gpu, c_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test float64 precision") {
  // Test that float64 maintains precision better than float32
  // Use a value that loses precision in float32
  double precise_value = 1.0 + 1e-8;  // This rounds to 1.0 in float32

  auto a_f32 = array({precise_value}, float32);
  auto b_f32 = array({1.0}, float32);
  auto diff_f32 = subtract(a_f32, b_f32, Device::gpu);

  auto a_f64 = array({precise_value}, float64);
  auto b_f64 = array({1.0}, float64);
  auto diff_f64 = subtract(a_f64, b_f64, Device::gpu);

  // float32 loses precision
  CHECK_LT(std::abs(diff_f32.item<float>()), 1e-7);

  // float64 maintains precision (using double-double arithmetic)
  CHECK_GT(std::abs(diff_f64.item<double>()), 1e-9);
}

TEST_CASE("test float64 scalar operations") {
  // Scalar-vector
  auto x = array({1.0, 2.0, 3.0}, float64);
  auto y = array(2.0, float64);

  auto z_cpu = add(x, y, Device::cpu);
  auto z_gpu = add(x, y, Device::gpu);
  CHECK(array_equal(z_gpu, z_cpu, Device::cpu).item<bool>());

  // Vector-scalar
  z_cpu = add(y, x, Device::cpu);
  z_gpu = add(y, x, Device::gpu);
  CHECK(array_equal(z_gpu, z_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test float64 broadcasting") {
  auto x = array({1.0, 2.0}, {2, 1}, float64);
  auto y = array({3.0, 4.0, 5.0}, {1, 3}, float64);

  auto z_cpu = add(x, y, Device::cpu);
  auto z_gpu = add(x, y, Device::gpu);

  CHECK_EQ(z_gpu.shape(), Shape{2, 3});
  CHECK_EQ(z_gpu.dtype(), float64);
  CHECK(array_equal(z_gpu, z_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test float64 comparison") {
  auto a = array({1.0, 2.0, 3.0}, float64);
  auto b = array({2.0, 2.0, 2.0}, float64);

  // Less than
  auto lt_cpu = less(a, b, Device::cpu);
  auto lt_gpu = less(a, b, Device::gpu);
  CHECK(array_equal(lt_gpu, lt_cpu, Device::cpu).item<bool>());

  // Equal
  auto eq_cpu = equal(a, b, Device::cpu);
  auto eq_gpu = equal(a, b, Device::gpu);
  CHECK(array_equal(eq_gpu, eq_cpu, Device::cpu).item<bool>());

  // Greater
  auto gt_cpu = greater(a, b, Device::cpu);
  auto gt_gpu = greater(a, b, Device::gpu);
  CHECK(array_equal(gt_gpu, gt_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test float64 min_max") {
  auto a = array({1.0, 5.0, 3.0}, float64);
  auto b = array({2.0, 4.0, 6.0}, float64);

  // Maximum
  auto max_cpu = maximum(a, b, Device::cpu);
  auto max_gpu = maximum(a, b, Device::gpu);
  CHECK(array_equal(max_gpu, max_cpu, Device::cpu).item<bool>());
  CHECK(array_equal(max_gpu, array({2.0, 5.0, 6.0}, float64)).item<bool>());

  // Minimum
  auto min_cpu = minimum(a, b, Device::cpu);
  auto min_gpu = minimum(a, b, Device::gpu);
  CHECK(array_equal(min_gpu, min_cpu, Device::cpu).item<bool>());
  CHECK(array_equal(min_gpu, array({1.0, 4.0, 3.0}, float64)).item<bool>());
}
