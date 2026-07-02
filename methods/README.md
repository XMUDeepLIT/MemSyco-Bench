# Vendored Method Sources

This directory holds **third-party source code** vendored for a subset of memory baselines.
It is **not** the benchmark integration layer.

All nine evaluation settings are integrated equally through [`baseline_adapters/`](../baseline_adapters/):

| Setting | Integration | Vendored code here? |
|---------|-------------|---------------------|
| `NoMemory` | evaluation runner | no |
| `RawDialogue` | evaluation runner | no |
| `MemZero` | `baseline_adapters/memzero.py` | yes (`LightMem/.../memory_toolkits`) |
| `A-MEM` | `baseline_adapters/amem.py` | yes (`LightMem/.../memory_toolkits`) |
| `NaiveRAG` | `baseline_adapters/naive_rag.py` | yes (`LightMem/.../memory_toolkits`) |
| `LightMem` | `baseline_adapters/lightmem.py` | yes (`LightMem/` native package) |
| `MemoryBank` | `baseline_adapters/memorybank.py` | no |
| `MemGPT` | `baseline_adapters/memgpt.py` | no |
| `Supermemory` | `baseline_adapters/supermemory.py` | no |

## Why `methods/LightMem/` exists

Historically, MemZero, A-MEM, and NaiveRAG were bundled inside the upstream LightMem
repository's `memory_toolkits`. MemSyco-Bench vendors a trimmed copy of that tree so these
four baselines can share one install path (`pip install -r requirements-memory-baselines.txt`).

The directory name reflects upstream packaging, not benchmark priority. A future refactor may
relocate this vendor tree (for example to `vendor/memory_toolkits/`), but the public integration
surface should remain `baseline_adapters/`.
