# MLXEmbedders

Ports of popular encoder / embedding models.

This module provides:

- `ModelConfiguration` (embedding-specific)
- `loadModelContainer(configuration: ...)` → `ModelContainer`
- `EmbeddingModel` implementations + pooling helpers

## Usage example

```swift
import MLX
import MLXEmbedders

let container = try await loadModelContainer(configuration: .nomic_text_v1_5)

let inputs = [
    "search_query: Animals in tropical climates.",
    "search_document: Elephants",
    "search_document: Polar bears",
]

let embeddings: [[Float]] = await container.perform { model, tokenizer, pooling in
    let tokens = inputs.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
    let maxLength = tokens.map(\.count).max() ?? 0
    let eos = tokenizer.eosTokenId ?? 0

    let padded = stacked(tokens.map { MLXArray($0 + Array(repeating: eos, count: maxLength - $0.count)) })
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

## Notes

This codebase started as a Swift port of models from `taylorai/mlx_embedding_models`.
