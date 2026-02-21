# Using a model

`ChatSession` (see <doc:evaluation>) is the recommended entry point for most apps.
This article describes the lower-level APIs for callers that want tighter control over generation.

## Loading a model

LLMs are typically loaded via `LLMModelFactory`:

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)
```

`ModelContainer` is an actor that provides isolation for the underlying `ModelContext`.

## Preparing inputs

You typically start from `UserInput` and convert it to an `LMInput` using the model’s processor:

```swift
import MLXLMCommon

let userInput = UserInput(prompt: "Hello")
let lmInput = try await container.prepare(input: userInput)
```

For VLMs, include images/videos in `UserInput` (or use `ChatSession.respond(...)` convenience overloads).

## Generating output (stream)

`ModelContainer.generate(...)` returns an `AsyncStream<Generation>` that includes text chunks and tool calls:

```swift
import MLXLMCommon

let stream = try await container.generate(
    input: lmInput,
    parameters: GenerateParameters(maxTokens: 256)
)

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

## Wired memory (optional)

You can pass a `WiredMemoryTicket` to coordinate a single global wired limit across concurrent inference tasks:

```swift
import MLX
import MLXLMCommon

let policy = WiredSumPolicy()
let ticket = policy.ticket(size: estimatedBytes)

let stream = try await container.generate(
    input: lmInput,
    parameters: .init(),
    wiredMemoryTicket: ticket
)
```

For runtime sizing, see `MLXLMCommon/Documentation.docc/wired-memory.md`.
