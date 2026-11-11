# CentraleSupélec projects

Coursework, assignments, challenges, and projects from CentraleSupélec.

The owner approved publishing this consolidated repository, including coursework
from previously private sources. The nine original GitHub repositories were deleted
after verifying their final source files were preserved here.

## Repository map

| Directory | Contents |
| --- | --- |
| `AI-2A/` | Second-year artificial intelligence coursework |
| `RL/` | Reinforcement learning labs and reports |
| `RL/project/` | Reinforcement learning project |
| `VIC/` | Visual recognition coursework |
| `VIC/project/` | Face recognition project |
| `VIC/challenge/` | Visual recognition challenge |
| `EI-ST5/` | Wave equation with fractal boundary conditions |
| `InterCentrales-2023/` | Leaf Nothing Behind competition |
| `Machine-Learning-Challenge/` | Beacon Biosignals sleep staging challenge |

## Running projects

There is no shared environment. Enter a project directory and follow its local
README and dependency files. Projects with a `pyproject.toml` and `uv.lock` can
usually be set up with `uv sync`. Dataset preparation and training may download
external data, require credentials, or use substantial compute.

## Consolidation

`CONSOLIDATION.json` records source commits and file checksums. These are file
snapshots, not imported Git histories, issues, or releases. Python caches and old
submodule configuration are omitted; other source files are preserved.

Nested projects are ordinary directories, not submodules:

- The original `RL/RL-project` submodule is now `RL/project/`, using the latest
  standalone project snapshot recorded in the manifest.
- `VIC/project/` contains the latest standalone VIC project snapshot.
- `RL/Week6/tp05_value_function_approximation/` was recovered from the original
  CentraleSupélec GitLab source at the exact referenced commit
  `3e9af6fd3aab7a1bedbcffddcdd123ef9e428b86`.

References to old standalone repository names may remain in source documents.
Runtime environments and training have not been tested during consolidation.
