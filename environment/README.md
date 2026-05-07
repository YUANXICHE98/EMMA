# Environment Layout

This project now treats each benchmark as its own environment package while sharing one EMMA backend implementation.

## Structure

- `memrl_core/`: shared EMMA backend surface used by every benchmark adapter
- `scienceworld/`: ScienceWorld benchmark adapter, config, and runner
- `intercode_sql/`: InterCode-SQL benchmark adapter, config, and runner
- `swebench/`: SWE-bench benchmark adapter, config, and runner
- `alfworld/`, `webarena/`, `osworld/`: benchmark-specific adapters or scaffolds

## Design Rule

- Interfaces are shared.
- Memory banks are benchmark-private.
- Cross-benchmark memory transfer is a separate experiment, not the default runtime path.

In practice that means each benchmark writes its own memory files under its own `results/` directory, even though all benchmarks call the same EMMA backend loop.
