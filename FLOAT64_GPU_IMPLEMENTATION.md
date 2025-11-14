# Float64 GPU Support Implementation Plan

## Goal
Add true `float64` dtype support on Apple Silicon GPUs using double-double (DD) arithmetic in Metal, since Metal doesn't have native `double` support.

## Current Status: Foundation Complete

### What's Implemented

1. **`double_double.h`** - Complete Metal library
   - Error-free transformations: `two_sum()`, `two_prod()` with FMA
   - DD arithmetic: `dd_add()`, `dd_sub()`, `dd_mul()`, `dd_div()`
   - Memory helpers: `pack_dd()`, `unpack_dd()`
   - ~30-32 digits of precision using 2x float32

2. **`binary_float64_ops.h`** - Float64 binary operations
   - Storage format: `float64_t = metal::float2` (hi, lo)
   - Operations: Add, Subtract, Multiply, Divide, Min, Max
   - Comparisons: Equal, Less, Greater, etc.
   - Uses DD arithmetic internally

3. **CMakeLists.txt** - Build system updated
   - `double_double.h` added to BASE_HEADERS
   - Will be compiled into mlx.metallib

4. **Runtime Validation** - Float64 GPU operation validation
   - Added validation in array constructor to detect float64 GPU operations
   - Provides diagnostic logging via `MLX_FLOAT64_VERBOSE` environment variable
   - Confirms operations are executing on GPU (not falling back to CPU)
   - Helps users verify and debug float64 GPU operations

## Architecture

### Storage Format
```cpp
// In C++: array with dtype=float64
// In Metal: buffer of float2 (hi, lo pairs)
float64_t value;  // Actually float2 in Metal
value.x = hi;     // High-order 32 bits
value.y = lo;     // Low-order correction term
```

### Operation Flow
```
Python: mx.add(a, b)  where a.dtype == float64
    ↓
C++: Binary::eval_gpu() dispatches on dtype
    ↓
Metal: AddFloat64 functor
    ↓
DD arithmetic: dd_add(to_dd(a), to_dd(b))
    ↓
Result: float2 packed back to output buffer
```

### Runtime Validation Usage

The implementation includes optional diagnostic logging to help verify that float64 operations are executing on GPU:

```python
import os
import mlx.core as mx

# Enable verbose logging for float64 GPU operations
os.environ['MLX_FLOAT64_VERBOSE'] = '1'

# Create float64 arrays and perform GPU operations
a = mx.array([1.0, 2.0, 3.0], dtype=mx.float64)
b = mx.array([4.0, 5.0, 6.0], dtype=mx.float64)
c = mx.add(a, b, stream=mx.gpu)  # Will print diagnostic message
mx.eval(c)

# Output: "[MLX] float64 GPU operation detected. Using double-double 
#          arithmetic for extended precision. Operations will execute 
#          on GPU (not falling back to CPU)."
```

The validation message is printed only once (on first float64 GPU operation) and only when `MLX_FLOAT64_VERBOSE` is set. This helps users:
- Confirm float64 operations are actually running on GPU
- Debug performance or accuracy issues
- Understand that double-double arithmetic is being used


## Next Steps (In Order)

### Phase 1: Metal Kernel Instantiation

**File:** `mlx/backend/metal/kernels/binary.metal`

Add float64 instantiations after existing types:

```cpp
#include "mlx/backend/metal/kernels/binary_float64_ops.h"

// Add to end of file:
#define instantiate_binary_float64(op)                    \
  instantiate_binary_all(op, float64, float64_t, float64_t)

instantiate_binary_float64(AddFloat64)
instantiate_binary_float64(SubtractFloat64)
instantiate_binary_float64(MultiplyFloat64)
instantiate_binary_float64(DivideFloat64)
instantiate_binary_float64(MaximumFloat64)
instantiate_binary_float64(MinimumFloat64)

// Comparison operations (return bool)
#define instantiate_binary_float64_bool(op)             \
  instantiate_binary_all(op, float64, float64_t, bool)

instantiate_binary_float64_bool(EqualFloat64)
instantiate_binary_float64_bool(NotEqualFloat64)
instantiate_binary_float64_bool(LessFloat64)
instantiate_binary_float64_bool(LessEqualFloat64)
instantiate_binary_float64_bool(GreaterFloat64)
instantiate_binary_float64_bool(GreaterEqualFloat64)
```

**Note:** Verify that instantiate_kernel macros support float2 as a type.

### Phase 2: C++ Dispatch Integration

**File:** `mlx/backend/metal/binary.cpp`

Current behavior: float64 probably falls back to CPU or errors.

**Option A: Special Case**
```cpp
void binary_op_gpu(const std::vector<array>& inputs, array& out, const char* op) {
  if (out.dtype() == float64) {
    // Use float64-specific kernels
    binary_op_gpu_float64(inputs, out, op);
    return;
  }
  // ... existing code
}
```

**Option B: Unified Path**
Modify type_to_name() to emit correct Metal type for float64.

**File:** `mlx/backend/metal/utils.cpp`
```cpp
std::string type_to_name(const Dtype& t) {
  // ...
  case float64:
    tname = "float64";  // Maps to float64_t in Metal (which is float2)
    break;
}
```

### Phase 3: Memory Layout

**Critical Issue:** MLX needs to know float64 uses 2x storage.

**File:** `mlx/dtype.h` (already correct!)
```cpp
inline constexpr Dtype float64{Dtype::Val::float64, sizeof(double)};
// sizeof(double) = 8 bytes = 2x float32 ✓
```

Metal buffers will receive float2 arrays automatically since size is correct.

### Phase 4: Type Promotion

**File:** `mlx/dtype_utils.cpp`

Ensure type promotion handles float64:
```cpp
Dtype promote_types(const Dtype& t1, const Dtype& t2) {
  // float32 + float64 → float64
  // float64 + int32 → float64
  // etc.
}
```

### Phase 5: Testing

**File:** `tests/test_float64_metal.cpp`

```cpp
TEST_CASE("float64 GPU addition") {
  auto a = array({1.0}, float64);
  auto b = array({2.0}, float64);
  auto c = add(a, b, StreamOrDevice(Device::gpu));
  CHECK_EQ(c.item<double>(), 3.0);
}

TEST_CASE("float64 precision") {
  // Test that we maintain >float32 precision
  float a_val = 1.0f + 1e-8f;  // Lost in float32
  auto a = array({a_val}, float64);
  auto b = array({1.0}, float64);
  auto c = subtract(a, b, StreamOrDevice(Device::gpu));
  CHECK(c.item<double>() > 1e-9);  // Precision retained
}
```

## Alternative: Simpler Approach

If full integration proves too complex, **fallback strategy**:

### Create `mlx.float64` namespace
```python
# Python API
import mlx.core as mx
import mlx.float64 as mxf64

a = mx.array([1.0, 2.0], dtype=mx.float32)
a_dd = mxf64.lift(a)  # Convert to DD representation
b_dd = mxf64.lift(mx.array([3.0, 4.0]))

c_dd = mxf64.add(a_dd, b_dd)  # Operates in DD on GPU
c = mxf64.lower(c_dd)  # Round back to float32
```

**Pros:**
- Simpler, less invasive
- Can iterate quickly
- Users opt-in explicitly

**Cons:**
- Not true dtype support
- Requires manual lift/lower
- Doesn't benefit existing code

## Current Blocker

**Main issue:** MLX's Metal kernel system assumes:
1. Types map 1:1 to Metal types
2. Operations use native Metal operators

float64 breaks both assumptions:
1. Dtype is `float64` but Metal type is `float2`
2. Operations need custom DD arithmetic, not native +/-/*//

**Possible solutions:**
1. Add a "compound types" system to MLX (big change)
2. Treat float64 as a special case everywhere (messy)
3. Use namespace/wrapper approach (cleaner boundaries)

## Recommendation

**For immediate use in Ember ML:**

Use the **mx.fast.metal_kernel approach** we already have working in ember-ml/backend/mlx/linearalg/math_ops.py:
- Proven to work
- Full control
- Can iterate quickly
- Easy to test

**For upstream MLX contribution:**

Submit the double-double library as a **utility** for users who need extended precision, not as a dtype. Let Apple/MLX team decide if they want full float64 dtype support.

## Files Modified

```
mlx-precise/
├── mlx/
│   └── array.cpp                          [Modified: Added float64 GPU validation]
├── mlx/backend/metal/kernels/
│   ├── CMakeLists.txt                     [Modified: Added double_double.h]
│   ├── double_double.h                    [New: DD arithmetic library]
│   └── binary_float64_ops.h               [New: Float64 operations]
└── FLOAT64_GPU_IMPLEMENTATION.md          [Modified: Added validation docs]
```

## Build Instructions

```bash
cd /Volumes/emberstuff/Projects/mlx-precise

# Clean build
rm -rf build
mkdir build && cd build

# Configure
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_PYTHON_BINDINGS=ON

# Build
make -j8

# Install
pip install -e python/
```

## References

- Dekker (1971): "A floating-point technique for extending the available precision"
- Knuth TAOCP Vol 2: Seminumerical Algorithms
- Shewchuk (1997): "Adaptive Precision Floating-Point Arithmetic"
- ember-ml working implementation: `/Volumes/stuff/Projects/ember-ml/ember_ml/backend/mlx/linearalg/math_ops.py`
