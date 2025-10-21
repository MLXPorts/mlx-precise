MLX‑precise Packaging
=====================

Goal
----
Mirror upstream MLX packaging semantics while keeping dev installs simple:

- Dev installs (pip from Git): single monolithic wheel that includes the native
  backend payload for the current platform (Metal on macOS). No extra steps.
- Release packaging (optional): two‑stage split wheels (Python + backend payload)
  for PyPI/conda distribution.

How it works (already in setup.py)
----------------------------------
- `MLX_BUILD_STAGE=0` (default): monolithic wheel named `mlx` for the current
  platform. This is used by `pip install git+…` in development and CI.
- `MLX_BUILD_STAGE=1`: Python‑only `mlx` wheel with a dependency on backend
  payload wheels built in stage 2.
- `MLX_BUILD_STAGE=2`: backend payload wheel:
  - macOS: `mlx-metal`
  - Linux CUDA: `mlx-cuda`
  - Linux CPU: `mlx-cpu`

Dev install (monolithic, recommended)
-------------------------------------
```bash
MLX_BUILD_STAGE=0 \
  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64" \
  python -m pip install -U --no-build-isolation \
    "git+https://github.com/SolaceHarmony/mlx-precise@main"
```

Two‑stage release (optional)
----------------------------
1) Stage 2 (payload):
```bash
export MLX_BUILD_STAGE=2
python -m pip wheel . -w dist --no-deps --no-build-isolation
# produces mlx-metal (macOS), or mlx-{cuda,cpu} on Linux
```

2) Stage 1 (Python):
```bash
export MLX_BUILD_STAGE=1
python -m pip wheel . -w dist --no-deps --no-build-isolation
# produces mlx, with install_requires referencing the stage‑2 wheel for the
# current platform
```

Conda packaging (optional)
--------------------------
- A simple recipe is provided at `conda/recipe/meta.yaml`. Build with:

```bash
conda build -c conda-forge conda/recipe
```

Versioning policy
-----------------
- This fork uses `0.29.99` (and tags `v0.29.99-precise`) to ensure resolver
  precedence over upstream `0.29.x`. Update as needed when upstream bumps.

