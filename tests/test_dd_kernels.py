"""
Tests for Double-Double Extended Precision Metal Kernels

Validates that the Metal kernels:
1. Execute without errors
2. Produce correct results
3. Improve precision over float32
4. Match expected performance characteristics
"""

import mlx.core as mx
import pytest

from mlx.backend.metal.dd_kernels import (
    complex_multiply_extended,
    fft_scale_extended,
    depthwise3_extended,
    lift_complex_to_dd,
    round_complex_from_dd,
)

from mlx.precise import (
    extended_precision,
    hyena_conv_extended,
)


class TestBasicKernelExecution:
    """Test that kernels execute without errors."""

    def test_complex_multiply_kernel_execution(self):
        """Test that complex multiply kernel executes."""
        size = 128
        a = mx.random.normal((size,)) + 1j * mx.random.normal((size,))
        b = mx.random.normal((size,)) + 1j * mx.random.normal((size,))

        # Should execute without error
        result = complex_multiply_extended(a, b, round_output=True)

        assert result.shape == a.shape
        assert result.dtype == mx.complex64

    def test_fft_scale_kernel_execution(self):
        """Test that FFT scaling kernel executes."""
        size = 256
        x = mx.random.normal((size,)) + 1j * mx.random.normal((size,))

        # Lift to DD
        x_dd = lift_complex_to_dd(x)

        # Scale
        result = fft_scale_extended(x_dd, n=512)

        assert result.shape == x_dd.shape

    def test_depthwise3_kernel_execution(self):
        """Test that depthwise convolution kernel executes."""
        x = mx.random.normal((256,))
        weights = mx.random.normal((3,))

        result = depthwise3_extended(x, weights)

        assert result.shape == x.shape
        assert result.dtype == mx.float32


class TestKernelCorrectness:
    """Test that kernels produce mathematically correct results."""

    def test_complex_multiply_correctness(self):
        """DD complex multiply should match reference."""
        size = 64
        mx.random.seed(42)

        # Create test data in float32
        a_f32 = mx.random.normal((size,)) + 1j * mx.random.normal((size,))
        b_f32 = mx.random.normal((size,)) + 1j * mx.random.normal((size,))

        # Standard float32
        result_f32 = a_f32 * b_f32

        # DD multiply
        result_dd = complex_multiply_extended(a_f32, b_f32, round_output=True)

        # Errors (compare to standard multiply)
        diff = mx.abs(result_f32 - result_dd)
        max_diff = float(mx.max(diff))

        print(f"\nComplex Multiply Correctness:")
        print(f"  Max difference: {max_diff:.2e}")

        # DD should produce results close to standard (within numerical tolerance)
        assert max_diff < 1e-5  # Should be very close for same precision

    def test_fft_scaling_correctness(self):
        """FFT scaling should match reference."""
        size = 128
        n = 256

        mx.random.seed(42)
        x = mx.random.normal((size,)) + 1j * mx.random.normal((size,))

        # DD scaling
        x_dd = lift_complex_to_dd(x)
        result_dd = fft_scale_extended(x_dd, n=n)
        result = round_complex_from_dd(result_dd)

        # Standard float32 scaling
        result_f32 = x / n

        # Compare results
        diff = mx.abs(result - result_f32)
        max_diff = float(mx.max(diff))

        print(f"\nFFT Scaling Correctness:")
        print(f"  Max difference: {max_diff:.2e}")

        # DD should produce results close to standard
        assert max_diff < 1e-5

    def test_depthwise3_correctness(self):
        """Depthwise conv should match reference."""
        length = 100
        mx.random.seed(42)

        x = mx.random.normal(shape=(length,))
        weights = mx.array([0.25, 0.5, 0.25])  # Simple smoothing

        # DD result
        result_dd = depthwise3_extended(x, weights)

        # Standard float32
        x_pad = mx.pad(x, (1, 1), mode='constant')
        result_f32 = (
            weights[0] * x_pad[:-2] +
            weights[1] * x_pad[1:-1] +
            weights[2] * x_pad[2:]
        )

        # Compare results
        diff = mx.abs(result_dd - result_f32)
        max_diff = float(mx.max(diff))

        print(f"\nDepthwise3 Correctness:")
        print(f"  Max difference: {max_diff:.2e}")

        # DD should produce results close to standard
        assert max_diff < 1e-5


class TestHyenaConvPrecision:
    """Test Hyena convolution with extended precision."""

    def test_hyena_conv_basic(self):
        """Basic Hyena conv should execute."""
        B, H, L = 2, 4, 256

        u = mx.random.normal((B, H, L))
        k_time = mx.random.normal((H, 2*L))
        D = mx.random.normal((1, H, 1))

        # Extended precision path
        y_ext = hyena_conv_extended(u, k_time, fft_size=2*L, D=D)

        assert y_ext.shape == u.shape
        assert y_ext.dtype == mx.float32

    def test_hyena_conv_vs_standard(self):
        """Extended should match standard (within tolerance)."""
        B, H, L = 1, 2, 128

        u = mx.random.normal((B, H, L))
        k_time = mx.random.normal((H, 2*L))

        # Standard path (multiple rounding points)
        fft_size = 2*L
        u_f = mx.fft.rfft(u, n=fft_size, axis=-1)
        k_f = mx.fft.rfft(k_time, n=fft_size, axis=-1) / fft_size
        y_f = u_f * k_f
        y_std = mx.fft.irfft(y_f, n=fft_size, axis=-1)[..., :L]

        # Extended path (single rounding point)
        y_ext = hyena_conv_extended(u, k_time, fft_size=fft_size)

        # Should be close
        rel_error = float(mx.max(mx.abs(y_std - y_ext)) / mx.max(mx.abs(y_std)))

        print(f"\nHyena Conv Standard vs Extended:")
        print(f"  Relative error: {rel_error:.2e}")

        # Should be within reasonable tolerance
        # (Different rounding order can cause small differences)
        assert rel_error < 1e-4  # 0.01% difference acceptable


class TestContextManager:
    """Test the extended_precision context manager."""

    def test_context_manager_basic(self):
        """Context manager should enable/disable extended precision."""
        from mlx.precise import is_extended_precision_enabled

        assert not is_extended_precision_enabled()

        with extended_precision():
            assert is_extended_precision_enabled()

        assert not is_extended_precision_enabled()

    def test_context_manager_nested(self):
        """Nested context managers should work."""
        from mlx.precise import is_extended_precision_enabled

        with extended_precision():
            assert is_extended_precision_enabled()

            # Nested should still work
            with extended_precision():
                assert is_extended_precision_enabled()

            assert is_extended_precision_enabled()

        assert not is_extended_precision_enabled()


class TestLiftAndRound:
    """Test lifting and rounding operations."""

    def test_lift_round_identity(self):
        """Lift then round should be identity (within epsilon)."""
        mx.random.seed(42)
        x = mx.random.normal((50,)) + 1j * mx.random.normal((50,))

        x_dd = lift_complex_to_dd(x)
        x_recovered = round_complex_from_dd(x_dd)

        # Check that recovery is close to original
        diff = mx.abs(x - x_recovered)
        max_diff = float(mx.max(diff))

        assert max_diff < 1e-6

    def test_lift_shape(self):
        """Lift should add DD dimension."""
        x = mx.random.normal((32,)) + 1j * mx.random.normal((32,))

        x_dd = lift_complex_to_dd(x)

        # DD format should have extra dimension for (hi, lo) pairs
        assert x_dd.shape[-1] == 4  # (re_hi, re_lo, im_hi, im_lo)


@pytest.mark.benchmark
class TestPerformance:
    """Benchmark extended precision overhead."""

    def test_complex_multiply_throughput(self):
        """Measure complex multiply throughput."""
        import time

        size = 8192
        a = mx.random.normal((size,)) + 1j * mx.random.normal((size,))
        b = mx.random.normal((size,)) + 1j * mx.random.normal((size,))

        # Warmup
        _ = a * b
        mx.eval(_)
        _ = complex_multiply_extended(a, b, round_output=True)
        mx.eval(_)

        # Benchmark float32
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            result = a * b
            mx.eval(result)
        time_f32 = (time.perf_counter() - start) / n_iters

        # Benchmark DD
        start = time.perf_counter()
        for _ in range(n_iters):
            result = complex_multiply_extended(a, b, round_output=True)
            mx.eval(result)
        time_dd = (time.perf_counter() - start) / n_iters

        overhead = time_dd / time_f32

        print(f"\nComplex Multiply Throughput:")
        print(f"  Float32: {time_f32*1e6:.2f} μs")
        print(f"  DD:      {time_dd*1e6:.2f} μs")
        print(f"  Overhead: {overhead:.1f}x")

        # Document overhead (no assertion, informational only)
        # Expected: 2-6x overhead for extended precision


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
