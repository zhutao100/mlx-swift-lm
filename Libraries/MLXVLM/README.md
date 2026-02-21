# MLXVLM

Vision-language model (VLM) implementations for `mlx-swift-lm`.

This module provides:

- `VLMModelFactory` — downloads config/weights, instantiates the model, loads weights/tokenizer, and
  constructs a model-specific `UserInputProcessor` from `processor_config.json` / `preprocessor_config.json`.
- `VLMTypeRegistry.shared` — mapping from `config.json` `model_type` → Swift model constructor.
- `VLMProcessorTypeRegistry.shared` — mapping from processor type → processor constructor.
- `VLMRegistry` — optional per-model overrides.

Concrete model implementations live under `Models/`.

## Quickstart (image)

```swift
import MLXVLM
import MLXLMCommon

let container = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-VL-4B-Instruct-4bit")
)

let session = ChatSession(container)
let answer = try await session.respond(
    to: "What is in this image?",
    image: .url(URL(fileURLWithPath: "./test.jpg"))
)

print(answer)
```

## Video inputs

`UserInput.Video` supports either a URL (decoded via AVFoundation) or an in-memory list of frames.
Media preprocessing lives in `MLXVLM/MediaProcessing.swift`.

## Supported model/processor types

The authoritative lists are the registries constructed in:

- `VLMTypeRegistry.shared` — `VLMModelFactory.swift`
- `VLMProcessorTypeRegistry.shared` — `VLMModelFactory.swift`

If you need to check whether a given `model_type` / processor class is supported, look up the keys there.

## Examples

This repository contains libraries only (no CLI targets).
For runnable examples, see `ml-explore/mlx-swift-examples`.
