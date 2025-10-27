# MLX-Precise: True Double-Precision + Python 3.14 Free-Threading

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://ml-explore.github.io/mlx/build/html/index.html) |
[**Examples**](#examples)

[![CircleCI](https://circleci.com/gh/ml-explore/mlx.svg?style=svg)](https://circleci.com/gh/ml-explore/mlx)

> **This fork adds true double-precision math and Python 3.14 free-threading support to MLX.**

**Key Enhancements:**
- **True Double-Precision:** Correct exp underflows on CPU paths (no silent float32 downcasts)
- **Python 3.14 Free-Threading:** Full GIL-free operation support (cp314t builds)
- **NumPy-Compatible API:** Drop-in replacement for NumPy in MLX workflows
- **Part of MLX Ecosystem:** Used by [exo-mlx](https://github.com/SolaceHarmony/exo-mlx) and [opencv-mlx](https://github.com/SolaceHarmony/opencv-mlx)

**Requirements:**
- **Python 3.14 or later** (recommended for free-threading)
- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS 11.0+**

**Installation:**
```bash
# Requires Python 3.14 free-threading build
python --version  # Should show Python 3.14.0 or later

# Install from git
pip install git+https://github.com/SolaceHarmony/mlx-precise@main

# Or as a dependency in your project
# mlx @ git+https://github.com/SolaceHarmony/mlx-precise@main
```

**Branch Strategy:**
- `main`: Tracks upstream MLX 0.29.x with precision patches + Python 3.14t support
- Version: `0.29.99.dev` (upstream-based with custom enhancements)

**Upstream:** Based on [MLX v0.29.x](https://github.com/ml-explore/mlx/tree/v0.29.0)

Some key features of MLX include:

- **Familiar APIs**: MLX has a Python API that closely follows NumPy. MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API. MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

- **Composable function transformations**: MLX supports composable function
  transformations for automatic differentiation, automatic vectorization,
  and computation graph optimization.

- **Lazy computation**: Computations in MLX are lazy. Arrays are only
  materialized when needed.

- **Dynamic graph construction**: Computation graphs in MLX are constructed
  dynamically. Changing the shapes of function arguments does not trigger
  slow compilations, and debugging is simple and intuitive.

- **Multi-device**: Operations can run on any of the supported devices
  (currently the CPU and the GPU).

- **Unified memory**: A notable difference from MLX and other frameworks
  is the *unified memory model*. Arrays in MLX live in shared memory.
  Operations on MLX arrays can be performed on any of the supported
  device types without transferring data.

MLX is designed by machine learning researchers for machine learning
researchers. The framework is intended to be user-friendly, but still efficient
to train and deploy models. The design of the framework itself is also
conceptually simple. We intend to make it easy for researchers to extend and
improve MLX with the goal of quickly exploring new ideas.

The design of MLX is inspired by frameworks like
[NumPy](https://numpy.org/doc/stable/index.html),
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/).

## Examples

The [MLX examples repo](https://github.com/ml-explore/mlx-examples) has a
variety of examples, including:

- [Transformer language model](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) training.
- Large-scale text generation with
  [LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) and
  finetuning with [LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora).
- Generating images with [Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
- Speech recognition with [OpenAI's Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

## Quickstart

See the [quick start
guide](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)
in the documentation.

## Installation

Requirements
------------
- Python 3.14 free‑threaded ONLY (cp314t). Other Python versions/builds are not supported.
- Non‑GIL runtime: this build declares free‑threaded safety and is intended to run with the GIL disabled.
  Use the 3.14 interpreter in non‑GIL mode (e.g., `python -X gil=0 ...`).

Install this fork from Git (recommended for precision fixes):

```bash
python -m pip install -U --no-build-isolation \
  "git+https://github.com/SolaceHarmony/mlx-precise@main"
```

To install the CUDA backend on Linux, run:

```bash
pip install mlx[cuda]
```

To install a CPU-only Linux package, run:

```bash
pip install mlx[cpu]
```

Conda packaging (optional):

```bash
conda build -c conda-forge conda/recipe
anaconda upload -u <your-channel> <built .conda>
```

Notes
-----
- This fork targets macOS/Metal primarily. The Python 3.14 free‑threaded build is required; earlier Python versions or GIL builds are out of scope.
- Wheels bundle the native Metal payload and libmlx.dylib inside the Python package; no environment variables are required for import.

## Contributing

Check out the [contribution guidelines](https://github.com/ml-explore/mlx/tree/main/CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

For original acknowledgments and citation, see upstream MLX.

## Citing MLX

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find
MLX useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```
