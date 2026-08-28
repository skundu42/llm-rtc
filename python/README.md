# llm-rtc

**Low-latency WebRTC for voice LLM applications.**

A Rust-first realtime audio transport library purpose-built for voice AI:
bidirectional (full-duplex) streaming between a user and a streaming LLM,
with the lowest achievable end-to-end latency.

## Why llm-rtc

Generic WebRTC stacks optimize for robustness and scale. Voice LLM apps need
the opposite: **minimal latency** from microphone to model and back. llm-rtc
tunes every layer for that goal:

- **Opus** with voice-optimized bitrate/DTX settings
- **Aggressive jitter buffer** (low-latency over resilience)
- **WebRTC audio processing** (AEC, NS, AGC) for clean model input
- **Full-duplex** audio sessions with a first-class LLM streaming API
- Rust core with **Rust and Python** SDKs

## Crates

- `llm-rtc-core` — WebRTC peer + audio pipeline engine
- `llm-rtc-sdk` — high-level Rust SDK (voice LLM session API)

## Status

Early development. See `crates/*/src` and `examples/`.

## License

MIT OR Apache-2.0