# Evaluation

The simplified chat API (`ChatSession`) allows you to load a model and evaluate prompts with only a few lines of code.

## Multi-turn chat (text)

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

The second question can refer to information introduced earlier; the session keeps a cached history/KV state.

If you need a one-shot prompt/response, create a session, evaluate once, then discard it.

## Streaming output

```swift
import MLXLLM
import MLXLMCommon

let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-4B-4bit")
)

let session = ChatSession(container)
for try await chunk in session.streamResponse(to: "Why is the sky blue?") {
    print(chunk, terminator: "")
}
print()
```

## VLMs (image/video)

The same `ChatSession` API works for VLMs when you load a VLM model:

```swift
import MLXVLM
import MLXLMCommon

let container = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Qwen3-VL-4B-Instruct-4bit")
)

let session = ChatSession(container)

let answer1 = try await session.respond(
    to: "What kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "./test.jpg"))
)
print(answer1)

// Follow-up question can refer back to the previous image.
let answer2 = try await session.respond(to: "What is behind it?")
print(answer2)
```

## Advanced usage

`ChatSession` has parameters you can supply when creating it:

- `instructions`: optional system instructions
- `generateParameters`: token limits, temperature, etc. (see ``GenerateParameters``)
- `processing`: image/video preprocessing settings (resize/normalization)
- `tools`: tool schemas for models that emit tool calls
