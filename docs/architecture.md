# Architecture

This repository is structured as a set of SwiftPM libraries that layer model *implementations* (LLM/VLM/embedder)
on top of a shared runtime (`MLXLMCommon`).

## High-level components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Your App                                                                    │
│  - selects factory (LLMModelFactory / VLMModelFactory)                       │
│  - uses ModelContainer and/or ChatSession                                    │
└──────────────┬──────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MLXLMCommon                                                                  │
│  Loading: ModelConfiguration + downloadModel/loadWeights                      │
│  Runtime: ModelContext + ModelContainer (actor isolation)                     │
│  UX: ChatSession (multi-turn, history/KV caching)                             │
│  Gen: generate(...) streaming (chunks + tool calls)                           │
│  Tooling: tool schema + format detection + parsers                            │
│  Memory: wired-memory policies + measurement helpers                          │
└──────────────┬──────────────────────────────────────────────────────────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
┌───────────────┐   ┌────────────────┐
│ MLXLLM        │   │ MLXVLM         │
│ - LLM models  │   │ - VLM models   │
│ - registries  │   │ - registries   │
│ - factory     │   │ - factory      │
└───────────────┘   └────────────────┘

(and MLXEmbedders for embedding models)
```

## Model loading

There are two common approaches:

1. **Call a concrete factory** (recommended for clarity):
   - `LLMModelFactory.shared.loadContainer(configuration: ...)`
   - `VLMModelFactory.shared.loadContainer(configuration: ...)`

2. **Call the convenience `loadModel*` functions** in `MLXLMCommon`:
   - `loadModel(id:)`, `loadModelContainer(id:)`, etc.

The convenience functions iterate over factories registered with `ModelFactoryRegistry`.
`ModelFactoryRegistry` uses Objective-C runtime lookup (`NSClassFromString(...)`) to discover
trampoline classes in `MLXLLM` and `MLXVLM`. That means your binary must *link* the corresponding
product(s) for those factories to be available.

Source of truth:

- Factory registry: `Libraries/MLXLMCommon/ModelFactory.swift` (`ModelFactoryRegistry`)
- Download path: `Libraries/MLXLMCommon/Load.swift` (`downloadModel(...)`)

## Evaluation and generation

The runtime data flow is:

1. **`UserInput`** (text + optional images/videos + optional tool specs)
2. Processor prepares an **`LMInput`**
3. **`generate(...)`** runs token iteration and emits streaming events

Key entry points:

- `ModelContainer.prepare(input:)` → `LMInput`
- `ModelContainer.generate(input:parameters:wiredMemoryTicket:)` → `AsyncStream<Generation>`
- `ChatSession.respond(...)` / `ChatSession.streamResponse(...)` (convenience wrapper)

`Generation` can carry:

- `.chunk(String)` — decoded text
- `.toolCall(ToolCall)` — tool call emitted by the model
- `.info(GenerationInfo)` — stop reason, perf counters, etc.

Source of truth:

- Generation loop: `Libraries/MLXLMCommon/Evaluate.swift`
- Container wrappers: `Libraries/MLXLMCommon/ModelContainer.swift`
- Session wrapper + caching: `Libraries/MLXLMCommon/ChatSession.swift`

## Tool calling

Tool calling is modeled as:

- Tool schemas (input/output shapes): `Libraries/MLXLMCommon/Tool/Tool.swift`
- Tool call events (`ToolCall`, `ToolCallFormat`): `Libraries/MLXLMCommon/Tool/*`
- Parser implementations for common formats (JSON/XML/“pythonic”, model-specific):
  `Libraries/MLXLMCommon/Tool/Parsers/*`

If you want tool calls, use `ChatSession.streamDetails(...)` or `ModelContainer.generate(...)`.
`ChatSession.respond(...)` and `ChatSession.streamResponse(...)` intentionally drop tool calls
and return only text.

## Wired memory coordination

On Apple Silicon, MLX can coordinate a single process-wide wired memory limit.
`MLXLMCommon` adds:

- Policies for sizing/admission: `WiredMemoryPolicies.swift`
- Measurement helpers for real budgets: `WiredMemoryUtils.swift`

DocC article:

- `Libraries/MLXLMCommon/Documentation.docc/wired-memory.md`
