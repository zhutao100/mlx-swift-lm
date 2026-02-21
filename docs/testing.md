# Testing and benchmarks

This package uses a mix of XCTest-based tests and Swift Testing (`import Testing`) suites.

## Unit tests

Unit tests live under `Tests/MLXLMTests/` and do **not** download models.

```sh
swift test --filter MLXLMTests
```

Notable coverage:

- `ChatSession` behavior
- Tool parsing/format detection
- Media preprocessing utilities (image/video)
- Wired memory policy logic

## Integration tests

Integration tests live under `Tests/MLXLMIntegrationTests/` and **download real models** from the Hub.
Expect large downloads and long runtimes.

```sh
swift test --filter MLXLMIntegrationTests
```

These tests are currently not gated behind an env var; if you run the full test suite (`swift test`),
they will execute.

## Benchmarks

Benchmarks live under `Tests/Benchmarks/` and are **gated** behind `RUN_BENCHMARKS`.

```sh
RUN_BENCHMARKS=1 swift test --filter Benchmarks
```

Benchmarks include model loading timing (LLM/VLM).
