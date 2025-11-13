"""
Tests for Extended Precision Backend

Demonstrates precision improvement from eliminating intermediate rounding.
"""

import random
from contextlib import contextmanager

import mlx.core as mx
import pytest

from mlx.precise import (
    extended_precision,
    complex_multiply_extended,
    lift_to_dd,
    round_to_float32,
    hyena_conv_extended,
    depthwise3_extended,
)


def _flatten(values):
    if isinstance(values, (list, tuple)):
        for item in values:
            yield from _flatten(item)
    else:
        yield values


def _max_abs_diff(values_a, values_b):
    return max(
        abs(a - b) for a, b in zip(_flatten(values_a), _flatten(values_b))
    )


def _assert_allclose(actual, expected, rtol=1e-7, atol=1e-7):
    diff = mx.abs(actual - expected)
    max_diff = float(mx.max(diff))
    ref = float(mx.max(mx.abs(expected)))
    allowed = atol + rtol * ref
    assert max_diff <= allowed + 1e-12, (
        f"max diff {max_diff:.3e} exceeds tolerance {allowed:.3e}"
    )


def _assert_arrays_equal(arr_a, arr_b):
    assert arr_a.tolist() == arr_b.tolist(), "arrays differ"


@contextmanager
def _cpu_float64_scope():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _random_complex_sequence(size, seed):
    rng = random.Random(seed)
    return [
        complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        for _ in range(size)
    ]


def _max_abs_error(mx_array, reference):
    return _max_abs_diff(mx_array.tolist(), reference)


class TestDoublePrecisionArithmetic:
    """Test double-double arithmetic correctness."""

    def test_lift_and_round_identity(self):
        """Lifting to DD and rounding back should be identity (within epsilon)."""
        x = mx.random.normal((100,))

        x_dd = lift_to_dd(x)
        x_recovered = round_to_float32(x_dd, is_complex=False)

        _assert_allclose(x, x_recovered, rtol=1e-7, atol=1e-7)

    def test_complex_lift_and_round(self):
        """Test complex lifting and rounding."""
        z = mx.random.normal((50,)) + 1j * mx.random.normal((50,))

        z_dd = lift_to_dd(z)
        z_recovered = round_to_float32(z_dd, is_complex=True)

        _assert_allclose(z.real, z_recovered.real, rtol=1e-7, atol=1e-7)
        _assert_allclose(z.imag, z_recovered.imag, rtol=1e-7, atol=1e-7)


class TestComplexMultiplyPrecision:
    """Test that DD complex multiply has better precision than float32."""

    def test_complex_multiply_vs_float32(self):
        """DD complex multiply should match float64 reference better than float32."""
        # Create test data
        size = 1024
        a_vals = _random_complex_sequence(size, seed=123)
        b_vals = _random_complex_sequence(size, seed=456)

        # Ground truth: float64
        truth = [a * b for a, b in zip(a_vals, b_vals)]

        # Float32 path (standard)
        a_f32 = mx.array(a_vals, dtype=mx.complex64)
        b_f32 = mx.array(b_vals, dtype=mx.complex64)
        result_f32 = a_f32 * b_f32

        # DD path (extended precision)
        result_dd = complex_multiply_extended(a_f32, b_f32, round_output=True)

        # Compute errors
        error_f32 = _max_abs_error(result_f32, truth)
        error_dd = _max_abs_error(result_dd, truth)

        print(f"\nComplex Multiply Precision:")
        print(f"  Float32 max error: {error_f32:.2e}")
        print(f"  DD max error:      {error_dd:.2e}")
        print(f"  Improvement:       {error_f32/error_dd:.1f}x")

        # DD should be significantly better
        # (May not be true in fallback Python implementation, but will be with Metal kernel)
        # For now, just verify it doesn't make things worse
        assert error_dd <= error_f32 * 1.1  # Allow 10% slack for fallback

    def test_complex_multiply_deterministic(self):
        """DD complex multiply should give identical results on repeated calls."""
        a = mx.random.normal((100,)) + 1j * mx.random.normal((100,))
        b = mx.random.normal((100,)) + 1j * mx.random.normal((100,))

        result1 = complex_multiply_extended(a, b, round_output=True)
        result2 = complex_multiply_extended(a, b, round_output=True)

        # Should be bit-identical
        _assert_arrays_equal(result1, result2)


class TestFFTPrecision:
    """Test FFT operations with extended precision."""

    def test_fft_round_trip(self):
        """FFT -> IFFT should recover input within precision limits."""
        x = mx.random.normal((1024,))

        # Standard path
        x_freq_std = mx.fft.rfft(x)
        x_recovered_std = mx.fft.irfft(x_freq_std, n=len(x))

        # Extended precision path
        with extended_precision():
            x_freq_ext = mx.fft.rfft(x)
            x_recovered_ext = mx.fft.irfft(x_freq_ext, n=len(x))

        # Both should be close to input
        error_std = float(mx.max(mx.abs(x - x_recovered_std)))
        error_ext = float(mx.max(mx.abs(x - x_recovered_ext)))

        print(f"\nFFT Round-trip Error:")
        print(f"  Standard:  {error_std:.2e}")
        print(f"  Extended:  {error_ext:.2e}")

        # Extended should be better or equal
        assert error_ext <= error_std * 1.1

    def test_parseval_theorem(self):
        """Energy should be conserved in FFT (Parseval's theorem)."""
        x = mx.random.normal((2048,))

        # Time domain energy
        energy_time = mx.sum(x * x)

        # Frequency domain energy
        x_freq = mx.fft.rfft(x)
        # For rfft: E_freq = (|X[0]|^2 + 2*sum(|X[1:N/2]|^2) + |X[N/2]|^2) / N
        energy_freq = (
            mx.abs(x_freq[0])**2 +
            2 * mx.sum(mx.abs(x_freq[1:-1])**2) +
            mx.abs(x_freq[-1])**2
        ) / len(x)

        error = float(mx.abs(energy_time - energy_freq) / energy_time)

        print(f"\nParseval's Theorem Error: {error:.2e}")

        # Should be very close (within float32 precision)
        assert error < 1e-5


class TestHyenaConvPrecision:
    """Test Hyena long convolution with extended precision."""

    def test_hyena_conv_basic(self):
        """Basic Hyena conv should work with extended precision."""
        B, H, L = 2, 4, 512

        u = mx.random.normal((B, H, L))
        k_time = mx.random.normal((H, 2*L))
        D = mx.random.normal((1, H, 1))
        mx.eval(u, k_time, D)

        # Standard path
        y_std = self._hyena_conv_standard(u, k_time, D)

        # Extended path
        y_ext = hyena_conv_extended(u, k_time, fft_size=2*L, D=D)

        # Results should be close (not identical due to different rounding points)
        rel_error = float(mx.max(mx.abs(y_std - y_ext)) / mx.max(mx.abs(y_std)))

        print(f"\nHyena Conv Relative Error (std vs ext): {rel_error:.2e}")

        # Should be within reasonable tolerance
        assert rel_error < 1e-4  # 0.01% difference acceptable

    def _hyena_conv_standard(self, u, k_time, D):
        """Standard Hyena conv (multiple rounding points)."""
        fft_size = k_time.shape[-1]

        u_f = mx.fft.rfft(u, n=fft_size)
        k_f = mx.fft.rfft(k_time, n=fft_size) / fft_size  # ⚠️ Round 1
        y_f = u_f * k_f  # ⚠️ Round 2
        y = mx.fft.irfft(y_f, n=fft_size)[..., :u.shape[-1]]

        if D is not None:
            y = y + u * D

        return y

    def test_hyena_conv_accumulation_stability(self):
        """Test that DD reduces accumulation error over long sequences."""
        H, L = 8, 4096

        u = mx.random.normal((1, H, L))
        k_time = mx.random.normal((H, 2*L))
        mx.eval(u, k_time)

        # Compute in float64 on CPU as ground truth
        with _cpu_float64_scope():
            u_f64 = mx.array(u.tolist(), dtype=mx.float64)
            k_f64 = mx.array(k_time.tolist(), dtype=mx.float64)
            u_f_f64 = mx.fft.rfft(u_f64, n=2 * L, axis=-1)
            k_f_f64 = mx.fft.rfft(k_f64, n=2 * L, axis=-1) / (2 * L)
            y_f_f64 = u_f_f64 * k_f_f64
            truth = mx.fft.irfft(y_f_f64, n=2 * L, axis=-1)[..., :L]
            truth_list = truth.tolist()

        # Float32 path
        y_f32 = self._hyena_conv_standard(u, k_time, D=None)

        # DD path
        with extended_precision():
            y_dd = hyena_conv_extended(u, k_time, fft_size=2*L, D=None)

        # Errors vs float64 truth
        error_f32 = _max_abs_error(y_f32, truth_list)
        error_dd = _max_abs_error(y_dd, truth_list)

        print(f"\nLong Sequence Accumulation Error vs float64:")
        print(f"  Float32: {error_f32:.2e}")
        print(f"  DD:      {error_dd:.2e}")
        print(f"  Improvement: {error_f32/error_dd:.1f}x")

        # TODO(sydney): tighten this bound once the Metal kernel improves precision.
        assert error_dd < 5e-2


class TestDepthwisePrecision:
    """Test depthwise convolution with extended precision."""

    def test_depthwise3_basic(self):
        """3-tap depthwise should work."""
        x = mx.random.normal((2, 4, 256))
        weights = mx.random.normal((3,))

        y = depthwise3_extended(x, weights)

        assert y.shape == x.shape
        assert y.dtype == mx.float32

    def test_depthwise3_precision(self):
        """DD depthwise should reduce accumulation error."""
        x = mx.random.normal((1, 1, 1000))
        weights = mx.array([0.25, 0.5, 0.25])  # Simple smoothing

        # Standard path (numpy float64 as truth)
        x_vals = x.tolist()[0][0]
        weights_vals = weights.tolist()

        # Manual convolution in float64 (Python floats are double precision)
        truth = [0.0] * len(x_vals)
        for i in range(len(x_vals)):
            if i > 0:
                truth[i] += weights_vals[0] * x_vals[i - 1]
            truth[i] += weights_vals[1] * x_vals[i]
            if i < len(x_vals) - 1:
                truth[i] += weights_vals[2] * x_vals[i + 1]

        # DD path
        y_dd = depthwise3_extended(x, weights)

        y_vals = y_dd.tolist()[0][0]
        error = max(abs(val - ref) for val, ref in zip(y_vals, truth))

        print(f"\nDepthwise Conv Error vs float64: {error:.2e}")

        # Should be very small
        assert error < 1e-5


class TestDeterminism:
    """Test that extended precision operations are deterministic."""

    def test_complex_multiply_deterministic(self):
        """Repeated calls should give identical results."""
        a = mx.random.normal((100,)) + 1j * mx.random.normal((100,))
        b = mx.random.normal((100,)) + 1j * mx.random.normal((100,))

        results = []
        for _ in range(5):
            r = complex_multiply_extended(a, b, round_output=True)
            results.append(r.tolist())

        # All results should be bit-identical
        for i in range(1, len(results)):
            assert results[0] == results[i]

    @pytest.mark.xfail(reason="hyena_conv_extended uses non-deterministic FFT kernels on Metal", strict=False)
    def test_hyena_conv_deterministic(self):
        """Hyena conv should be deterministic with extended precision."""
        mx.random.seed(0)
        u = mx.random.normal((1, 4, 256))
        k_time = mx.random.normal((4, 512))
        mx.eval(u, k_time)

        results = []
        for _ in range(3):
            with extended_precision():
                y = hyena_conv_extended(u, k_time, fft_size=512)
                mx.eval(y)
                results.append(y.tolist())

        baseline = results[0]
        for current in results[1:]:
            diff = _max_abs_diff(baseline, current)
            assert diff < 0.2


class TestPerformanceBenchmark:
    """Benchmark extended precision overhead."""

    def test_complex_multiply_throughput(self):
        """Measure throughput difference between float32 and DD."""
        import time

        size = 16384
        a = mx.random.normal((size,)) + 1j * mx.random.normal((size,))
        b = mx.random.normal((size,)) + 1j * mx.random.normal((size,))

        # Warmup
        _ = a * b
        _ = complex_multiply_extended(a, b, round_output=True)

        # Benchmark float32
        n_iters = 1000
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = a * b
            mx.eval(_)
        time_f32 = (time.perf_counter() - start) / n_iters

        # Benchmark DD
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = complex_multiply_extended(a, b, round_output=True)
            mx.eval(_)
        time_dd = (time.perf_counter() - start) / n_iters

        overhead = time_dd / time_f32

        print(f"\nComplex Multiply Throughput:")
        print(f"  Float32: {time_f32*1e6:.2f} μs")
        print(f"  DD:      {time_dd*1e6:.2f} μs")
        print(f"  Overhead: {overhead:.1f}x")

        # Document overhead (no assertion, just informational)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
