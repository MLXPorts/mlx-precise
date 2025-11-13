#!/usr/bin/env python
# Copyright © 2025 The Solace Project
# Copyright © 2023-2024 Apple Inc.

"""
Test float64 CPU ↔ GPU conversion for precision and special values.
"""

import math
import sys

import mlx.core as mx
import numpy as np


def test_special_values():
    """Test infinity, -infinity, NaN, and zeros."""
    print("Testing special values...")

    # Test infinity
    x_cpu = mx.array([float('inf')], dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    assert math.isinf(x_cpu.item()), "CPU: inf not preserved"
    assert math.isinf(x_back.item()), "Round-trip: inf not preserved"
    assert x_cpu.item() > 0, "CPU: inf should be positive"
    assert x_back.item() > 0, "Round-trip: inf should be positive"
    print("  ✓ +inf preserved")

    # Test negative infinity
    x_cpu = mx.array([float('-inf')], dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    assert math.isinf(x_cpu.item()), "CPU: -inf not preserved"
    assert math.isinf(x_back.item()), "Round-trip: -inf not preserved"
    assert x_cpu.item() < 0, "CPU: -inf should be negative"
    assert x_back.item() < 0, "Round-trip: -inf should be negative"
    print("  ✓ -inf preserved")

    # Test NaN
    x_cpu = mx.array([float('nan')], dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    assert math.isnan(x_cpu.item()), "CPU: NaN not preserved"
    assert math.isnan(x_back.item()), "Round-trip: NaN not preserved"
    print("  ✓ NaN preserved")

    # Test zero
    x_cpu = mx.array([0.0], dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    assert x_cpu.item() == 0.0, "CPU: zero not preserved"
    assert x_back.item() == 0.0, "Round-trip: zero not preserved"
    print("  ✓ Zero preserved")

    # Test negative zero
    x_cpu = mx.array([-0.0], dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    assert x_cpu.item() == -0.0, "CPU: -zero not preserved"
    assert x_back.item() == -0.0, "Round-trip: -zero not preserved"
    print("  ✓ -Zero preserved")

    print("✓ All special values passed!\n")


def test_precision():
    """Test precision preservation for various values."""
    print("Testing precision preservation...")

    # The key test case from the user
    original = 0.384723498732489723487238478374
    x_cpu = mx.array([original], dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    result = x_back.item()
    abs_error = abs(result - original)
    rel_error = abs_error / abs(original)

    print(f"  High-precision value:")
    print(f"    Original:    {original:.30f}")
    print(f"    Round-trip:  {result:.30f}")
    print(f"    Abs error:   {abs_error:.2e}")
    print(f"    Rel error:   {rel_error:.2e}")

    # Should preserve at least float32 precision (7-8 significant digits)
    assert rel_error < 1e-6, f"Precision loss too large: {rel_error}"
    print("    ✓ Precision acceptable (< 1e-6 relative error)")

    # Test various magnitudes
    test_values = [
        1.0,
        -1.0,
        1234.5678,
        -1234.5678,
        1.23456789012345e10,   # Large value
        1.23456789012345e-10,  # Small value
        math.pi,
        math.e,
        math.sqrt(2.0),
    ]

    max_rel_error = 0.0
    for val in test_values:
        x_cpu = mx.array([val], dtype=mx.float64)
        x_gpu = mx.array(x_cpu, stream=mx.gpu)
        x_back = mx.array(x_gpu, stream=mx.cpu)

        result = x_back.item()
        abs_error = abs(result - val)
        rel_error = abs_error / abs(val) if val != 0 else abs_error

        max_rel_error = max(max_rel_error, rel_error)

        if rel_error > 1e-6:
            print(f"  ✗ FAILED: {val} -> {result} (rel_error={rel_error:.2e})")
            return False

    print(f"  ✓ All test values passed (max rel_error={max_rel_error:.2e})\n")
    return True


def test_array_conversion():
    """Test conversion works for entire arrays."""
    print("Testing array conversion...")

    values = [
        float('inf'),
        float('-inf'),
        0.0,
        -0.0,
        1.0,
        -1.0,
        math.pi,
        0.384723498732489723487238478374,
    ]

    x_cpu = mx.array(values, dtype=mx.float64)
    x_gpu = mx.array(x_cpu, stream=mx.gpu)
    x_back = mx.array(x_gpu, stream=mx.cpu)

    for i, original in enumerate(values):
        result = float(x_back[i].item())

        if math.isinf(original):
            assert math.isinf(result), f"Index {i}: inf not preserved"
            assert (original > 0) == (result > 0), f"Index {i}: inf sign changed"
        elif math.isnan(original):
            assert math.isnan(result), f"Index {i}: NaN not preserved"
        else:
            abs_error = abs(result - original)
            rel_error = abs_error / abs(original) if original != 0 else abs_error

            if rel_error > 1e-6:
                print(f"  ✗ FAILED at index {i}: {original} -> {result} (rel_error={rel_error:.2e})")
                return False

    print("  ✓ All array elements preserved\n")
    return True


def test_gpu_operations():
    """Test that GPU operations maintain precision."""
    print("Testing GPU operations with conversion...")

    a_val = 0.384723498732489723487238478374
    b_val = 1.234567890123456789

    a_cpu = mx.array([a_val], dtype=mx.float64)
    b_cpu = mx.array([b_val], dtype=mx.float64)

    # Transfer to GPU
    a_gpu = mx.array(a_cpu, stream=mx.gpu)
    b_gpu = mx.array(b_cpu, stream=mx.gpu)

    # Perform operation on GPU
    c_gpu = mx.add(a_gpu, b_gpu, stream=mx.gpu)

    # Transfer back to CPU
    c_back = mx.array(c_gpu, stream=mx.cpu)

    expected = a_val + b_val
    result = c_back.item()

    abs_error = abs(result - expected)
    rel_error = abs_error / abs(expected)

    print(f"  Operation: {a_val} + {b_val}")
    print(f"    Expected:    {expected:.30f}")
    print(f"    Result:      {result:.30f}")
    print(f"    Abs error:   {abs_error:.2e}")
    print(f"    Rel error:   {rel_error:.2e}")

    assert rel_error < 1e-6, f"GPU operation precision loss: {rel_error}"
    print("    ✓ Precision acceptable\n")


def main():
    """Run all conversion tests."""
    print("=" * 60)
    print("Float64 CPU ↔ GPU Conversion Tests")
    print("=" * 60)
    print()

    try:
        test_special_values()
        if not test_precision():
            sys.exit(1)
        if not test_array_conversion():
            sys.exit(1)
        test_gpu_operations()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
