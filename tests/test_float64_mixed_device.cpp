// Copyright © 2025 The Solace Project
// Copyright © 2023-2024 Apple Inc.
//
// Tests for float64 mixed CPU/GPU device operations

#include <cmath>
#include <limits>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test float64 basic CPU to GPU to CPU") {
  // Start on CPU
  auto x_cpu = array({1.0, 2.0, 3.0}, float64);

  // Transfer to GPU
  auto x_gpu = array(x_cpu.shape(), float64, nullptr, {});
  x_gpu = copy(x_cpu, Device::gpu);
  eval(x_gpu);

  // Operate on GPU
  auto y_gpu = add(x_gpu, x_gpu, Device::gpu);
  eval(y_gpu);

  // Transfer back to CPU
  auto y_cpu = copy(y_gpu, Device::cpu);
  eval(y_cpu);

  // Verify dtype preserved
  CHECK_EQ(y_cpu.dtype(), float64);

  // Verify values: x + x = 2x
  auto expected = array({2.0, 4.0, 6.0}, float64);
  CHECK(allclose(y_cpu, expected, Device::cpu).item<bool>());
}

TEST_CASE("test float64 multiple boundary crossings") {
  double precise_val = 0.123456789012345678;
  auto x = array({precise_val}, float64);

  // CPU → GPU
  auto y1 = add(x, array({0.1}, float64), Device::gpu);
  eval(y1);

  // GPU → CPU
  auto y2 = multiply(y1, array({2.0}, float64), Device::cpu);
  eval(y2);

  // CPU → GPU
  auto y3 = add(y2, array({0.2}, float64), Device::gpu);
  eval(y3);

  // GPU → CPU
  auto y4 = multiply(y3, array({0.5}, float64), Device::cpu);
  eval(y4);

  // Expected: ((x + 0.1) * 2 + 0.2) * 0.5
  double expected = ((precise_val + 0.1) * 2.0 + 0.2) * 0.5;
  double result = y4.item<double>();

  double abs_error = std::abs(result - expected);
  double rel_error = abs_error / std::abs(expected);

  INFO("Input: ", precise_val);
  INFO("Expected: ", expected);
  INFO("Result: ", result);
  INFO("Relative error: ", rel_error);

  // Should preserve at least float32 precision
  CHECK(rel_error < 1e-6);
}

TEST_CASE("test float64 RNN state pattern") {
  // Simulate RNN hidden state crossing boundaries
  int batch_size = 4;
  int hidden_dim = 8;

  std::vector<double> h_data(batch_size * hidden_dim);
  std::vector<double> x_data(batch_size * hidden_dim);

  for (int i = 0; i < batch_size * hidden_dim; ++i) {
    h_data[i] = static_cast<double>(i) / 100.0;
    x_data[i] = static_cast<double>(i % 10) / 10.0;
  }

  auto h = array(h_data.data(), {batch_size, hidden_dim}, float64);
  auto x = array(x_data.data(), {batch_size, hidden_dim}, float64);

  // GPU: Combine input and hidden state
  auto combined = add(x, h, Device::gpu);
  eval(combined);

  // CPU: Apply threshold (control logic)
  auto clipped = minimum(
    maximum(combined, array({-5.0}, float64), Device::cpu),
    array({5.0}, float64),
    Device::cpu
  );
  eval(clipped);

  // GPU: Activation
  auto activated = tanh(clipped, Device::gpu);
  eval(activated);

  // Verify shape and dtype
  CHECK_EQ(activated.shape()[0], batch_size);
  CHECK_EQ(activated.shape()[1], hidden_dim);
  CHECK_EQ(activated.dtype(), float64);

  // Verify tanh bounds
  auto data_ptr = activated.data<double>();
  for (int i = 0; i < batch_size * hidden_dim; ++i) {
    CHECK(std::abs(data_ptr[i]) <= 1.0);
  }
}

TEST_CASE("test float64 special values across boundaries") {
  // Test infinity
  {
    auto x_cpu = array({std::numeric_limits<double>::infinity()}, float64);
    auto x_gpu = copy(x_cpu, Device::gpu);
    eval(x_gpu);
    auto x_back = copy(x_gpu, Device::cpu);
    eval(x_back);

    CHECK(std::isinf(x_back.item<double>()));
    CHECK(x_back.item<double>() > 0);
  }

  // Test negative infinity
  {
    auto x_cpu = array({-std::numeric_limits<double>::infinity()}, float64);
    auto x_gpu = copy(x_cpu, Device::gpu);
    eval(x_gpu);
    auto x_back = copy(x_gpu, Device::cpu);
    eval(x_back);

    CHECK(std::isinf(x_back.item<double>()));
    CHECK(x_back.item<double>() < 0);
  }

  // Test NaN
  {
    auto x_cpu = array({std::numeric_limits<double>::quiet_NaN()}, float64);
    auto x_gpu = copy(x_cpu, Device::gpu);
    eval(x_gpu);
    auto x_back = copy(x_gpu, Device::cpu);
    eval(x_back);

    CHECK(std::isnan(x_back.item<double>()));
  }

  // Test zero
  {
    auto x_cpu = array({0.0}, float64);
    auto x_gpu = copy(x_cpu, Device::gpu);
    eval(x_gpu);
    auto x_back = copy(x_gpu, Device::cpu);
    eval(x_back);

    CHECK_EQ(x_back.item<double>(), 0.0);
  }
}

TEST_CASE("test float64 array operations across boundaries") {
  std::vector<double> values = {
    1.0,
    -1.0,
    M_PI,
    M_E,
    0.123456789012345,
    1234.5678,
    1.23456789e-10,
  };

  auto x_cpu = array(values.data(), {static_cast<int>(values.size())}, float64);

  // CPU → GPU
  auto x_gpu = copy(x_cpu, Device::gpu);
  eval(x_gpu);

  // Operate on GPU
  auto y_gpu = multiply(x_gpu, array({2.0}, float64), Device::gpu);
  eval(y_gpu);

  // GPU → CPU
  auto y_cpu = copy(y_gpu, Device::cpu);
  eval(y_cpu);

  // Verify
  auto y_data = y_cpu.data<double>();
  for (size_t i = 0; i < values.size(); ++i) {
    double expected = values[i] * 2.0;
    double result = y_data[i];
    double rel_error = std::abs((result - expected) / expected);

    INFO("Index: ", i);
    INFO("Expected: ", expected);
    INFO("Result: ", result);
    INFO("Relative error: ", rel_error);

    CHECK(rel_error < 1e-6);
  }
}

TEST_CASE("test float64 sequential operations") {
  // Simulate 10 timesteps of state updates
  int timesteps = 10;
  auto h = array({0.5}, float64);

  for (int t = 0; t < timesteps; ++t) {
    auto x = array({static_cast<double>(t) * 0.1}, float64);

    // GPU: Update
    h = add(h, x, Device::gpu);
    eval(h);

    // CPU: Clip
    h = minimum(maximum(h, array({-10.0}, float64), Device::cpu),
                array({10.0}, float64), Device::cpu);
    eval(h);

    // GPU: Scale
    h = multiply(h, array({0.9}, float64), Device::gpu);
    eval(h);

    // Verify dtype maintained
    CHECK_EQ(h.dtype(), float64);
  }

  // Final value should be reasonable
  double final_val = h.item<double>();
  CHECK(std::abs(final_val) < 10.0);
  CHECK(std::isfinite(final_val));
}

TEST_CASE("test float64 large array boundary crossing") {
  // Test with realistic sizes
  int batch_size = 32;
  int hidden_dim = 512;
  int total_size = batch_size * hidden_dim;

  std::vector<double> data(total_size);
  for (int i = 0; i < total_size; ++i) {
    data[i] = static_cast<double>(i) / static_cast<double>(total_size);
  }

  auto x_cpu = array(data.data(), {batch_size, hidden_dim}, float64);

  // Transfer to GPU
  auto x_gpu = copy(x_cpu, Device::gpu);
  eval(x_gpu);

  // Operate
  auto y_gpu = multiply(x_gpu, array({0.5}, float64), Device::gpu);
  eval(y_gpu);

  // Transfer back
  auto y_cpu = copy(y_gpu, Device::cpu);
  eval(y_cpu);

  // Verify shape
  CHECK_EQ(y_cpu.shape()[0], batch_size);
  CHECK_EQ(y_cpu.shape()[1], hidden_dim);
  CHECK_EQ(y_cpu.dtype(), float64);

  // Spot check values
  auto y_data = y_cpu.data<double>();
  CHECK(std::abs(y_data[0] - data[0] * 0.5) < 1e-10);
  CHECK(std::abs(y_data[total_size - 1] - data[total_size - 1] * 0.5) < 1e-10);
}
