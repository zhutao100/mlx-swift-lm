# ``MLXLLM``

Example implementations of various Large Language Models (LLMs).

## Other MLX libraries in this repo

- [MLXEmbedders](MLXEmbedders)
- [MLXLLM](MLXLLM)
- [MLXLMCommon](MLXLMCommon)
- [MLXVLM](MLXVLM)

## Quick Start

See <doc:evaluation>.

Using an LLM is as easy as:

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let session = ChatSession(container)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

More advanced APIs are available for those that need them, see <doc:using-model>.

## Topics

- <doc:evaluation>
- <doc:adding-model>
- <doc:using-model>
