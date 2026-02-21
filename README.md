# mlx-swift-lm

SwiftPM libraries for running **LLMs**, **VLMs** (image/video), and **embedding models** on Apple Silicon using
[MLX Swift](https://github.com/ml-explore/mlx-swift).

This repository ships four SwiftPM products:

- **MLXLMCommon**: shared runtime (model loading, evaluation/generation, KV cache, `ModelContainer`, `ChatSession`,
  tool calling helpers, wired-memory budgeting utilities)
- **MLXLLM**: text-only model implementations + `LLMModelFactory`/registries
- **MLXVLM**: vision-language model implementations + `VLMModelFactory`/registries (image + video inputs)
- **MLXEmbedders**: encoder / embedding model implementations + pooling utilities

## What works today

- Load models from a local directory or the Hugging Face Hub (downloads `*.safetensors` and JSON config files).
- Text generation with streaming output.
- Multimodal chat for supported VLMs (images and videos).
- Tool-calling event emission during generation (via `ChatSession.streamDetails(...)` and `ModelContainer.generate(...)`).
- LoRA/DoRA **adapter application** (load/unload/fuse) for models that expose LoRA layers.
- Wired-memory coordination helpers (policies + measurement utilities) for budgeting and admission control.

## Not in this repo

- No CLI or sample app targets live here.
  For runnable examples, see [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples).
- “Full fine-tuning” helpers are not provided as a turnkey workflow.
  This repo includes adapter application utilities and a basic LoRA training loop, but does not currently ship
  an end-to-end training driver (dataset ingestion, checkpoint packaging, etc.).

## Requirements

- Swift toolchain that supports `swift-tools-version: 5.12`.
- Platforms: macOS 14+, iOS 17+, tvOS 17+, visionOS 1+.

## Add as a dependency

In `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
```

Or use the latest release:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.29.1")),
```

Then depend on one or more products:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
    ]
)
```

## Quickstart

### Text-only LLM

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let session = ChatSession(container)
print(try await session.respond(to: "What are two things to see in San Francisco?"))

for try await chunk in session.streamResponse(to: "Name one great place to eat there.") {
    print(chunk, terminator: "")
}
print()
```

### VLM (image)

```swift
import MLXVLM
import MLXLMCommon

let container = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-VL-4B-Instruct-4bit")
)

let session = ChatSession(container)
let answer = try await session.respond(
    to: "Describe this image.",
    image: .url(URL(fileURLWithPath: "./test.jpg"))
)
print(answer)
```

### Embeddings

```swift
import MLX
import MLXEmbedders

let container = try await loadModelContainer(configuration: .nomic_text_v1_5)

let embeddings: [[Float]] = await container.perform { model, tokenizer, pooling in
    let inputs = ["search_query: Tropical climates.", "search_document: Elephants"].map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
    }

    let maxLength = inputs.map(\.count).max() ?? 0
    let eos = tokenizer.eosTokenId ?? 0

    let padded = stacked(inputs.map { MLXArray($0 + Array(repeating: eos, count: maxLength - $0.count)) })
    let mask = (padded .!= eos)
    let tokenTypes = MLXArray.zeros(like: padded)

    let pooled = pooling(
        model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true,
        applyLayerNorm: true
    )

    eval(pooled)
    return pooled.map { $0.asArray(Float.self) }
}

print("Embeddings:", embeddings.count)
```

## Testing

Unit tests (fast, no network):

```sh
swift test --filter MLXLMTests
```

Integration tests download real models from the Hub (slow, large downloads):

```sh
swift test --filter MLXLMIntegrationTests
```

Benchmarks are gated behind an env var:

```sh
RUN_BENCHMARKS=1 swift test --filter Benchmarks
```

## Documentation map

- `docs/` — repo-level docs (architecture + testing)
- `Libraries/MLXLMCommon/` — core runtime
- `Libraries/MLXLLM/` — text-only models
- `Libraries/MLXVLM/` — vision-language models
- `Libraries/MLXEmbedders/` — embedding models
- `skills/` — agent/assistant “skill” docs for tooling that supports local skill bundles
