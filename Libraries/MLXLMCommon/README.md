# MLXLMCommon

Shared runtime and utilities for **text (LLM)** and **multimodal (VLM)** inference.

This module contains the code you typically build against in an application:

- `ModelConfiguration`, `BaseConfiguration`, registries
- `ModelContext` + `ModelContainer` (actor isolation)
- `UserInput` → `LMInput` preparation
- `generate(...)` / streaming APIs (text chunks + tool calls)
- `ChatSession` (multi-turn convenience API)
- Wired-memory helpers (`Wired*Policy`, `WiredMemoryUtils`)

## Quickstart

For a concrete model implementation, link `MLXLLM` or `MLXVLM` and load through the
corresponding factory:

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let session = ChatSession(container)
print(try await session.respond(to: "Why is the sky blue?"))
```

## Lower-level generation

`ModelContainer` exposes a lower-level interface that surfaces tool calls:

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let input = try await container.prepare(input: UserInput(prompt: "Hello"))
let stream = try await container.generate(input: input, parameters: .init())

for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")
    case .toolCall(let call):
        print("\nTool call: \(call.function.name)")
    case .info:
        break
    }
}
print()
```

## DocC articles

- Porting models: `Documentation.docc/porting.md`
- Wired-memory budgeting: `Documentation.docc/wired-memory.md`
