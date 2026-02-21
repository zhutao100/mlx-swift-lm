# Documentation index

This folder is repo-level documentation. Module-level docs live next to the code under `Libraries/*`.

## Start here

- [Architecture](architecture.md)
- [Testing and benchmarks](testing.md)

## Module docs

- `Libraries/MLXLMCommon/README.md` — shared runtime
- `Libraries/MLXLLM/README.md` — text-only models
- `Libraries/MLXVLM/README.md` — vision-language models
- `Libraries/MLXEmbedders/README.md` — embedding models

## DocC articles

These are authored under `Libraries/*/Documentation.docc/`.

- `Libraries/MLXLMCommon/Documentation.docc/porting.md` — porting models
- `Libraries/MLXLMCommon/Documentation.docc/wired-memory.md` — wired-memory budgeting
- `Libraries/MLXLLM/Documentation.docc/evaluation.md` — high-level evaluation (chat + streaming)
- `Libraries/MLXLLM/Documentation.docc/using-model.md` — lower-level generation

## Agent skill docs

If your workflow uses “skill bundle” tooling, see `skills/`.
