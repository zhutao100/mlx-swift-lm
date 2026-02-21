# MLXLLM

Text-only model implementations for `mlx-swift-lm`.

This module provides:

- `LLMModelFactory` — downloads config/weights, instantiates a model, loads weights, and loads the tokenizer.
- `LLMTypeRegistry.shared` — mapping from `config.json` `model_type` → Swift model constructor.
- `LLMRegistry` — optional per-model overrides (prompt, tokenizer quirks, etc.).

The concrete model implementations live under `Models/`.

## Quickstart

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let session = ChatSession(container)
for try await chunk in session.streamResponse(to: "Write a haiku about Swift.") {
    print(chunk, terminator: "")
}
print()
```

## Supported model types

The authoritative list is the `creators:` dictionary passed to `LLMTypeRegistry.shared`:

- `LLMTypeRegistry.shared` — `LLMModelFactory.swift`

If you need to check whether a given `model_type` is supported, look up the key there.

## Adding a new model

1. Add the model implementation in `Models/`.
2. Register its `model_type` in `LLMTypeRegistry.shared` (`LLMModelFactory.swift`).
3. (Optional) Add a convenience entry in `LLMRegistry` if you need per-model overrides.

See also the DocC porting guide in `MLXLMCommon/Documentation.docc/porting.md`.

## Adapters (LoRA / DoRA)

`MLXLLM` models conform to `LoRAModel` via `LLMModel`.
Adapter application (load/unload/fuse) lives in `MLXLMCommon`:

- `Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift`
- `Libraries/MLXLMCommon/Adapters/ModelAdapterFactory.swift`

This repo includes a basic LoRA training loop (`LoraTrain.swift`), but does not currently ship
an end-to-end training driver.

## Examples

This repository contains libraries only (no CLI targets).
For runnable examples, see `ml-explore/mlx-swift-examples`.
