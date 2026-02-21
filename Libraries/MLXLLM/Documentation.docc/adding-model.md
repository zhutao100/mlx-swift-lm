# Adding an LLM

If a model follows the typical LLM layout (Hugging Face / MLX):

- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- one or more `*.safetensors` shards

…you can usually add it by following the patterns in `Models/`.

## 1) Create a configuration type

Create a configuration struct that matches the fields you need from `config.json`:

```swift
public struct YourModelConfiguration: Codable, Sendable {
    public let hiddenSize: Int

    // use this pattern for values that need defaults
    public let _layerNormEps: Float?
    public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case _layerNormEps = "layer_norm_eps"
    }
}
```

## 2) Implement the model

The public model type must conform to `LanguageModel` (and typically `LLMModel` + `KVCacheDimensionProvider`).

```swift
import MLX
import MLXLMCommon
import MLXNN

public final class YourModel: Module, LLMModel, KVCacheDimensionProvider {
    public let kvHeads: [Int]

    @ModuleInfo var inner: YourModelInner

    public init(_ config: YourModelConfiguration) {
        self.kvHeads = Array(repeating: /* kvHeads */, count: /* num layers */)
        self.inner = YourModelInner(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> LMOutput {
        // return LMOutput(logits: ...)
        fatalError("implement")
    }

    public var loraLayers: [Module] {
        // return the transformer blocks you want adapters applied to
        []
    }
}
```

## 3) Register the `model_type`

Add the mapping from `config.json` `model_type` → Swift constructor by editing the
`creators:` dictionary passed to `LLMTypeRegistry.shared` in `LLMModelFactory.swift`.

## 4) (Optional) Add an `LLMRegistry` entry

If you need per-model overrides (prompt defaults, tokenizer overrides, etc.), add a
`ModelConfiguration` to `LLMRegistry` and include it in its `all()` list.

## See also

- `MLXLMCommon/Documentation.docc/porting.md` (porting workflow + debugging tips)
