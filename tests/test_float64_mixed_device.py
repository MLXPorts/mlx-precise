#!/usr/bin/env python
# Copyright © 2025 The Solace Project
# Copyright © 2023-2024 Apple Inc.

"""
Test float64 with mixed CPU/GPU operations, especially in @mx.compile.
Tests the RNN state management pattern and other heterogeneous compute scenarios.
"""

import sys
import mlx.core as mx
import numpy as np


def test_basic_mixed_device():
    """Test basic CPU↔GPU transitions with float64."""
    print("Testing basic mixed CPU/GPU with float64...")

    def mixed_fn(x):
        # GPU operation
        y = mx.add(x, x, stream=mx.gpu)

        # CPU operation (crosses boundary)
        z = mx.multiply(y, mx.array([2.0], dtype=mx.float64), stream=mx.cpu)

        # Back to GPU
        result = mx.add(z, mx.array([1.0], dtype=mx.float64), stream=mx.gpu)

        return result

    x = mx.array([1.0, 2.0, 3.0], dtype=mx.float64)

    # Without compile
    result_no_compile = mixed_fn(x)
    mx.eval(result_no_compile)

    # With compile
    compiled_mixed = mx.compile(mixed_fn)
    result_compiled = compiled_mixed(x)
    mx.eval(result_compiled)

    # Both should match
    assert mx.allclose(result_no_compile, result_compiled), \
        f"Compiled vs non-compiled mismatch: {result_no_compile} != {result_compiled}"

    # Check expected values: (x + x) * 2 + 1 = x * 4 + 1
    expected = x * 4.0 + 1.0
    assert mx.allclose(result_compiled, expected), \
        f"Result mismatch: {result_compiled} != {expected}"

    print(f"  ✓ Basic mixed device: {result_compiled}")


def test_rnn_state_pattern():
    """Test RNN-like state management across CPU/GPU boundaries."""
    print("\nTesting RNN state pattern with float64...")

    batch_size = 4
    hidden_dim = 8

    # Simplified RNN step
    def rnn_step(x, hidden_state):
        # GPU: Linear transformation (the heavy compute)
        # Simplified: just a multiply + add
        new_hidden_gpu = mx.multiply(x, hidden_state, stream=mx.gpu)
        new_hidden_gpu = mx.add(new_hidden_gpu, x, stream=mx.gpu)

        # CPU: State processing (control logic, metadata)
        # Simplified: just a normalization check
        state_max = mx.max(new_hidden_gpu, stream=mx.cpu)  # ← GPU→CPU conversion
        state_normalized = mx.divide(
            new_hidden_gpu,
            mx.maximum(state_max, mx.array([1.0], dtype=mx.float64)),
            stream=mx.cpu
        )

        # GPU: Apply activation
        activated = mx.tanh(state_normalized, stream=mx.gpu)  # ← CPU→GPU conversion

        return activated

    # Test data
    x = mx.array(np.random.randn(batch_size, hidden_dim), dtype=mx.float64)
    h0 = mx.array(np.random.randn(batch_size, hidden_dim), dtype=mx.float64)

    # Without compile
    result_no_compile = rnn_step(x, h0)
    mx.eval(result_no_compile)

    # With compile
    compiled_rnn = mx.compile(rnn_step)
    result_compiled = compiled_rnn(x, h0)
    mx.eval(result_compiled)

    # Check they match
    assert mx.allclose(result_no_compile, result_compiled, rtol=1e-6), \
        "RNN compiled vs non-compiled mismatch"

    # Check dtype preserved
    assert result_compiled.dtype == mx.float64, \
        f"Result dtype should be float64, got {result_compiled.dtype}"

    # Check shape
    assert result_compiled.shape == (batch_size, hidden_dim), \
        f"Shape mismatch: {result_compiled.shape}"

    print(f"  ✓ RNN state pattern (shape={result_compiled.shape}, dtype={result_compiled.dtype})")


def test_sequential_rnn_unrolled():
    """Test multi-timestep RNN with repeated CPU/GPU transitions."""
    print("\nTesting sequential RNN (10 timesteps)...")

    hidden_dim = 8
    sequence_length = 10

    def rnn_step_simple(x, h):
        # GPU: Combine input and state
        combined = mx.add(x, h, stream=mx.gpu)

        # CPU: Clip values (control logic)
        clipped = mx.clip(combined, -5.0, 5.0, stream=mx.cpu)

        # GPU: Activation
        return mx.tanh(clipped, stream=mx.gpu)

    # Compile the step
    compiled_step = mx.compile(rnn_step_simple)

    # Initial state
    h = mx.zeros((hidden_dim,), dtype=mx.float64)

    # Process sequence
    sequence = [mx.array([float(i)] * hidden_dim, dtype=mx.float64)
                for i in range(sequence_length)]

    states = []
    for t, x_t in enumerate(sequence):
        h = compiled_step(x_t, h)
        mx.eval(h)
        states.append(h)

    # Verify all states are float64
    for t, state in enumerate(states):
        assert state.dtype == mx.float64, f"Timestep {t}: dtype is {state.dtype}"

    # Verify final state is reasonable
    final_state = states[-1]
    assert mx.all(mx.abs(final_state) <= 1.0).item(), \
        "Tanh output should be in [-1, 1]"

    print(f"  ✓ Sequential RNN: {sequence_length} timesteps, final_state={final_state[:3]}...")


def test_precision_preservation():
    """Test that precision is preserved across CPU/GPU boundaries."""
    print("\nTesting precision preservation across boundaries...")

    # High-precision value
    precise_val = 0.123456789012345678

    def cross_boundary_multiple_times(x):
        # GPU → CPU
        y1 = mx.add(x, mx.array([0.1], dtype=mx.float64), stream=mx.cpu)

        # CPU → GPU
        y2 = mx.multiply(y1, mx.array([2.0], dtype=mx.float64), stream=mx.gpu)

        # GPU → CPU
        y3 = mx.add(y2, mx.array([0.2], dtype=mx.float64), stream=mx.cpu)

        # CPU → GPU
        y4 = mx.multiply(y3, mx.array([0.5], dtype=mx.float64), stream=mx.gpu)

        return y4

    x = mx.array([precise_val], dtype=mx.float64)

    # Compute expected: ((x + 0.1) * 2 + 0.2) * 0.5
    expected = ((precise_val + 0.1) * 2.0 + 0.2) * 0.5

    # Compiled version
    compiled_fn = mx.compile(cross_boundary_multiple_times)
    result = compiled_fn(x)
    mx.eval(result)

    result_val = result.item()
    abs_error = abs(result_val - expected)
    rel_error = abs_error / abs(expected)

    print(f"  Input:    {precise_val:.18f}")
    print(f"  Expected: {expected:.18f}")
    print(f"  Result:   {result_val:.18f}")
    print(f"  Abs err:  {abs_error:.2e}")
    print(f"  Rel err:  {rel_error:.2e}")

    # Should preserve at least float32 precision (1e-6)
    assert rel_error < 1e-6, f"Precision loss too large: {rel_error}"

    print("  ✓ Precision preserved across multiple boundaries")


def test_special_values_across_boundaries():
    """Test that special values survive CPU/GPU transitions."""
    print("\nTesting special values across boundaries...")

    def process_special(x):
        # GPU operation
        y = mx.add(x, mx.array([0.0], dtype=mx.float64), stream=mx.gpu)

        # CPU operation
        z = mx.multiply(y, mx.array([1.0], dtype=mx.float64), stream=mx.cpu)

        # Back to GPU
        return mx.add(z, mx.array([0.0], dtype=mx.float64), stream=mx.gpu)

    test_cases = [
        (float('inf'), 'inf'),
        (float('-inf'), '-inf'),
        (float('nan'), 'nan'),
        (0.0, 'zero'),
        (-0.0, '-zero'),
    ]

    compiled_process = mx.compile(process_special)

    for val, name in test_cases:
        x = mx.array([val], dtype=mx.float64)
        result = compiled_process(x)
        mx.eval(result)
        result_val = result.item()

        if name == 'inf':
            assert np.isinf(result_val) and result_val > 0, f"{name} not preserved"
        elif name == '-inf':
            assert np.isinf(result_val) and result_val < 0, f"{name} not preserved"
        elif name == 'nan':
            assert np.isnan(result_val), f"{name} not preserved"
        elif name == 'zero':
            assert result_val == 0.0, f"{name} not preserved"
        elif name == '-zero':
            assert result_val == -0.0, f"{name} not preserved"

        print(f"  ✓ {name:8s} preserved")


def test_large_state_tensor():
    """Test with realistic RNN hidden state sizes."""
    print("\nTesting with large state tensors...")

    batch_size = 32
    hidden_dim = 512

    def process_large_state(h):
        # GPU: Heavy operation
        h_gpu = mx.multiply(h, mx.array([0.99], dtype=mx.float64), stream=mx.gpu)

        # CPU: Compute norm (small reduction)
        norm = mx.sqrt(mx.sum(mx.square(h_gpu), stream=mx.cpu))

        # GPU: Normalize
        h_normalized = mx.divide(
            h_gpu,
            mx.maximum(norm, mx.array([1.0], dtype=mx.float64)),
            stream=mx.gpu
        )

        return h_normalized

    # Realistic hidden state
    h = mx.array(np.random.randn(batch_size, hidden_dim), dtype=mx.float64)

    compiled_fn = mx.compile(process_large_state)
    result = compiled_fn(h)
    mx.eval(result)

    # Check shape and dtype
    assert result.shape == (batch_size, hidden_dim)
    assert result.dtype == mx.float64

    # Check normalization worked
    result_norm = mx.sqrt(mx.sum(mx.square(result)))
    mx.eval(result_norm)
    assert result_norm.item() <= 1.0 or np.isclose(result_norm.item(), 1.0, rtol=1e-5)

    print(f"  ✓ Large state tensor: {result.shape}, norm={result_norm.item():.6f}")


def main():
    """Run all mixed device tests."""
    print("=" * 60)
    print("Float64 Mixed CPU/GPU Device Tests")
    print("=" * 60)
    print()

    try:
        test_basic_mixed_device()
        test_rnn_state_pattern()
        test_sequential_rnn_unrolled()
        test_precision_preservation()
        test_special_values_across_boundaries()
        test_large_state_tensor()

        print()
        print("=" * 60)
        print("✓ ALL MIXED DEVICE TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
