# Agent guide (mlx-swift-lm)

This repository is a SwiftPM package that ships **libraries** (no executable targets).
Treat the Swift source as the source of truth; keep Markdown docs aligned with actual code.

## Repo map

- `Package.swift` — SwiftPM products/targets and platform minimums
- `Libraries/MLXLMCommon/` — shared runtime (loading, evaluation, `ModelContainer`, `ChatSession`, tool calling, wired memory)
- `Libraries/MLXLLM/` — text-only model implementations + `LLMModelFactory`/registries
- `Libraries/MLXVLM/` — VLM implementations + `VLMModelFactory`/registries
- `Libraries/MLXEmbedders/` — embedding models
- `Tests/MLXLMTests/` — unit tests (no network)
- `Tests/MLXLMIntegrationTests/` — integration tests (download real models)
- `Tests/Benchmarks/` — benchmarks (gated by `RUN_BENCHMARKS`)
- `docs/` — repo-level docs (architecture/testing)

## Read this first (by task)

### Triage / debug

1. Reproduce with unit tests first:
   - `swift test --filter MLXLMTests`
2. If the issue only reproduces with real models:
   - `swift test --filter MLXLMIntegrationTests`
   - Expect large downloads + long runtimes.
3. When debugging model loading:
   - `Libraries/MLXLMCommon/Load.swift` (Hub download + local fallback)
   - `Libraries/MLXLMCommon/ModelFactory.swift` (factory registry + convenience loaders)
4. When debugging generation:
   - `Libraries/MLXLMCommon/Evaluate.swift` (token iteration, detokenization, stop reasons)
   - `Libraries/MLXLMCommon/ModelContainer.swift` (`prepare(...)`, `generate(...)` entry points)

### Feature work

- Add an LLM model type:
  - Implement under `Libraries/MLXLLM/Models/`
  - Register `model_type` in `Libraries/MLXLLM/LLMModelFactory.swift` (`LLMTypeRegistry.shared`)
  - Add optional overrides in `LLMRegistry` if needed
- Add a VLM model type/processor:
  - Implement under `Libraries/MLXVLM/Models/`
  - Register model type in `VLMTypeRegistry.shared`
  - Register processor type in `VLMProcessorTypeRegistry.shared`
- Extend tool calling:
  - Parsing/format detection: `Libraries/MLXLMCommon/Tool/*`
  - End-to-end emission: `ChatSession.streamDetails(...)` and `ModelContainer.generate(...)`
- Extend adapters (LoRA/DoRA):
  - Core types: `Libraries/MLXLMCommon/Adapters/*`
  - LoRA/DoRA layers: `Adapters/LoRA/*`

### Bugfix

- Prefer minimal repros in `Tests/MLXLMTests`.
- If the bug is model-specific, add a small regression test with the smallest practical model ID.
- When touching concurrency boundaries, prefer:
  - `ModelContainer` actor isolation
  - `SerialAccessContainer` for internal state machines

### Release

- This repo is versioned as a Swift package. Ensure:
  - `Package.swift` dependency constraints are intentional.
  - Any doc updates match the released API.

## Conventions and constraints

- **Strict concurrency** is enabled for targets (`StrictConcurrency`).
- **`MLXArray` is not `Sendable`**. Don’t try to “fix” concurrency errors by marking it as sendable.
  Use existing patterns:
  - `ModelContainer` actor isolation
  - `SendableBox` / `SerialAccessContainer`
  - `sending`/`consuming` parameters when transferring ownership across isolation boundaries
- `ChatSession` is **not** thread-safe; use one session per task.

## Common commands

Build:

```sh
swift build
```

Unit tests:

```sh
swift test --filter MLXLMTests
```

Integration tests (downloads models):

```sh
swift test --filter MLXLMIntegrationTests
```

Benchmarks:

```sh
RUN_BENCHMARKS=1 swift test --filter Benchmarks
```

## Where the “source of truth” lives

- Public API surface: Swift sources under `Libraries/**`.
- Loading behavior: `Libraries/MLXLMCommon/Load.swift`, `Libraries/MLXLMCommon/ModelFactory.swift`.
- Supported model types:
  - LLM: `Libraries/MLXLLM/LLMModelFactory.swift` (`LLMTypeRegistry.shared`)
  - VLM: `Libraries/MLXVLM/VLMModelFactory.swift` (`VLMTypeRegistry.shared`, processor registry)
- Tool calling:
  - Schemas: `Libraries/MLXLMCommon/Tool/Tool.swift`
  - Parsing/formats: `Libraries/MLXLMCommon/Tool/Parsers/*`
- Wired memory:
  - Policies: `Libraries/MLXLMCommon/WiredMemoryPolicies.swift`
  - Measurement: `Libraries/MLXLMCommon/WiredMemoryUtils.swift`
