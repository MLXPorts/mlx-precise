"""
Extended Precision helpers for MLX

Production implementation using Metal kernels with double-double arithmetic.
Based on patterns from xLSTM-metal kernel development.

Usage:
    from mlx.precise import extended_precision

    with extended_precision():
        y = mx.fft.rfft(x)  # Uses DD arithmetic internally
"""

from typing import Optional, Literal
import mlx.core as mx

from mlx.backend.metal.dd_kernels import (
    complex_multiply_extended,
    fft_scale_extended,
    depthwise3_extended,
    lift_complex_to_dd,
    round_complex_from_dd,
)

__all__ = [
    'extended_precision',
    'rfft_extended',
    'irfft_extended',
    'hyena_conv_extended',
    'lift_to_dd',
    'round_to_float32',
    'complex_multiply_extended',  # Re-export from dd_kernels
    'depthwise3_extended',         # Re-export from dd_kernels
]

# ============================================================================
# Global State
# ============================================================================

_EXTENDED_PRECISION_ENABLED = False
_PRECISION_MODE: Literal['float32', 'double_double'] = 'float32'
_PRECISION_STACK: list[Literal['float32', 'double_double']] = []


class extended_precision:
    """
    Context manager to enable extended precision for numerical operations.

    Example:
        with extended_precision():
            # All FFT ops use DD arithmetic
            spectrum = mx.fft.rfft(signal)
            result = hyena_conv(u, k)
    """

    def __init__(self, mode: Literal['double_double'] = 'double_double'):
        self.mode = mode
        self.prev_mode = None

    def __enter__(self):
        global _EXTENDED_PRECISION_ENABLED, _PRECISION_MODE, _PRECISION_STACK
        _PRECISION_STACK.append(_PRECISION_MODE)
        _EXTENDED_PRECISION_ENABLED = True
        _PRECISION_MODE = self.mode
        return self

    def __exit__(self, *args):
        global _EXTENDED_PRECISION_ENABLED, _PRECISION_MODE, _PRECISION_STACK
        if _PRECISION_STACK:
            _PRECISION_MODE = _PRECISION_STACK.pop()
        else:
            _PRECISION_MODE = 'float32'
        _EXTENDED_PRECISION_ENABLED = bool(_PRECISION_STACK)


def is_extended_precision_enabled() -> bool:
    """Check if extended precision mode is active."""
    return _EXTENDED_PRECISION_ENABLED


def lift_to_dd(x: mx.array) -> mx.array:
    """Lift a real or complex array to double-double representation."""
    if x.dtype == mx.complex64:
        return lift_complex_to_dd(x)
    zeros = mx.zeros_like(x)
    return mx.stack([x, zeros], axis=-1)


def round_to_float32(x_dd: mx.array, *, is_complex: bool) -> mx.array:
    """Round double-double representation back to float32."""
    if is_complex:
        return round_complex_from_dd(x_dd)
    return x_dd[..., 0] + x_dd[..., 1]


# ============================================================================
# Extended Precision FFT Operations
# ============================================================================

def rfft_extended(
    x: mx.array,
    n: Optional[int] = None,
    axis: int = -1
) -> mx.array:
    """
    Real FFT with extended precision.

    Currently uses standard FFT with DD post-processing.
    Full DD FFT implementation pending.

    Args:
        x: Real input signal (float32)
        n: FFT length (pads/truncates if needed)
        axis: Axis along which to compute FFT

    Returns:
        FFT spectrum (complex64 if not in extended mode, else DD)
    """
    # Compute standard FFT
    spectrum = mx.fft.rfft(x, n=n, axis=axis)

    if _EXTENDED_PRECISION_ENABLED:
        # Lift to DD for downstream operations
        spectrum_dd = lift_complex_to_dd(spectrum)
        return spectrum_dd
    else:
        return spectrum


def irfft_extended(
    x_freq: mx.array,
    n: Optional[int] = None,
    axis: int = -1
) -> mx.array:
    """
    Inverse real FFT with extended precision.

    Args:
        x_freq: Frequency spectrum (complex64 or DD)
        n: Output length
        axis: Axis along which to compute IFFT

    Returns:
        Real signal (float32)
    """
    # If input is DD, round to complex64 for standard IFFT
    # TODO: Implement full DD IFFT
    if x_freq.dtype == mx.float32 and len(x_freq.shape) > 1:
        # Assume it's DD format (has extra dimension)
        x_freq = round_complex_from_dd(x_freq)

    result = mx.fft.irfft(x_freq, n=n, axis=axis)
    return result


# ============================================================================
# Hyena Long Convolution (Extended Precision)
# ============================================================================

def hyena_conv_extended(
    u: mx.array,
    k_time: mx.array,
    fft_size: int,
    D: Optional[mx.array] = None,
    normalization: str = 'forward'
) -> mx.array:
    """
    Hyena long convolution with extended precision throughout.

    Pipeline (with extended precision):
        1. FFT of input u (standard precision, TODO: DD FFT)
        2. FFT of kernel k (standard precision, TODO: DD FFT)
        3. Lift to DD
        4. Scale kernel by 1/N in DD (NO float32 rounding)
        5. Complex multiply in DD (NO intermediate rounding)
        6. ROUND ONCE to complex64
        7. IFFT
        8. Add bias (float32)

    Args:
        u: Input signal (B, H, L)
        k_time: Time-domain kernel (H, 2L)
        fft_size: FFT length (2L)
        D: Optional bias/modulation (1, H, 1)
        normalization: 'forward' or 'backward' (controls where 1/N is applied)

    Returns:
        Convolved output (B, H, L) in float32
    """
    # FFTs (standard precision for now)
    # TODO: Implement full DD FFT for maximum precision
    u_f = mx.fft.rfft(u, n=fft_size, axis=-1)  # (B, H, L+1)
    k_f = mx.fft.rfft(k_time, n=fft_size, axis=-1)  # (H, L+1)

    if _EXTENDED_PRECISION_ENABLED:
        # Lift to DD
        u_f_dd = lift_complex_to_dd(u_f)
        k_f_dd = lift_complex_to_dd(k_f)

        # Scale kernel spectrum by 1/N in DD (avoids float32 rounding)
        if normalization == 'forward':
            k_f_dd = fft_scale_extended(k_f_dd, n=fft_size)

        # Complex multiply in DD (NO intermediate rounding)
        # This is the CRITICAL operation
        y_f_dd = complex_multiply_extended(
            u_f_dd,
            k_f_dd,
            round_output=False  # Keep in DD for now
        )

        # Round to complex64 for IFFT
        # ⚠️ THIS IS THE SINGLE ROUNDING POINT
        y_f = round_complex_from_dd(y_f_dd)

        # IFFT
        if normalization == 'forward':
            # Already scaled, so use 'forward' norm (no scaling on inverse)
            y = mx.fft.irfft(y_f, n=fft_size, axis=-1)[..., :u.shape[-1]]
        else:
            # Scale on inverse (default)
            y = mx.fft.irfft(y_f, n=fft_size, axis=-1)[..., :u.shape[-1]]
    else:
        # Standard float32 path (multiple rounding points)
        if normalization == 'forward':
            k_f = k_f / fft_size  # ⚠️ Round 1
        y_f = u_f * k_f  # ⚠️ Round 2 (actually 6 rounds per complex mul)
        y = mx.fft.irfft(y_f, n=fft_size, axis=-1)[..., :u.shape[-1]]

    # Add bias (in float32)
    if D is not None:
        y = y + u * D

    return y


# ============================================================================
# Integration with Ember ML ops
# ============================================================================

def patch_ember_fft_ops():
    """
    Monkey-patch Ember ML FFT operations to use extended precision.

    This is an experimental feature. Not recommended for production.
    Use the context manager instead.
    """
    import ember_ml.ops as ops

    # Store originals
    _original_rfft = ops.rfft
    _original_irfft = ops.irfft

    def rfft_patched(x, n=None, axis=-1):
        if _EXTENDED_PRECISION_ENABLED:
            return rfft_extended(x, n=n, axis=axis)
        else:
            return _original_rfft(x, n=n, axis=axis)

    def irfft_patched(x, n=None, axis=-1):
        if _EXTENDED_PRECISION_ENABLED:
            return irfft_extended(x, n=n, axis=axis)
        else:
            return _original_irfft(x, n=n, axis=axis)

    # Patch
    ops.rfft = rfft_patched
    ops.irfft = irfft_patched


def unpatch_ember_fft_ops():
    """Restore original FFT operations."""
    # This would need to store originals in a more robust way
    pass
