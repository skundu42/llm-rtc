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

- `llm-rtc-core` - WebRTC peer + audio pipeline engine
- `llm-rtc-sdk` - high-level Rust SDK (voice LLM session API)

## Quick Start

Requirements: Rust (stable), and on Debian/Ubuntu the build deps:

```sh
sudo apt-get install -y cmake pkg-config libwebrtc-audio-processing-dev
```

Build and test the whole workspace:

```sh
cargo build --workspace
cargo test --workspace
```

## Python SDK

The Python bindings live in `python/` and are built with
[PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs). Install maturin
(`pip install maturin`) and build the wheel:

```sh
cd python && python3 -m maturin build --release
pip install target/wheels/llm_rtc-*.whl
```

The wheel exposes Opus codecs, the audio processor (AEC/NS/AGC/VAD), the
jitter buffer, and the full audio pipeline:

```python
from llm_rtc import (
    OpusEncoder, OpusDecoder, AudioProcessor, JitterBuffer,
    CodecConfig, ProcessorConfig, JitterBufferConfig, AudioPacket,
)

# Encode PCM (16-bit mono, 48 kHz) with voice-optimized Opus
encoder = OpusEncoder(CodecConfig())
frames = encoder.encode_frames(pcm_i16)  # one RTP packet per frame

# Clean up the mic signal before sending it to the model
processor = AudioProcessor(ProcessorConfig())
cleaned = processor.process(mic_i16)

# Buffer incoming packets and playout at a fixed low latency
jitter = JitterBuffer(JitterBufferConfig())
jitter.push(AudioPacket(sequence_number=seq, timestamp=ts, payload=payload))
pkt = jitter.pop()  # None until the target latency is reached
```

## Architecture

The workspace is split into two crates plus the Python bindings:

- `crates/llm-rtc-core` - the engine: WebRTC peer (ICE/DTLS/SRTP), Opus
  codec, WebRTC audio processing (AEC/NS/AGC/VAD), jitter buffer, and the
  audio pipeline that wires them together
- `crates/llm-rtc-sdk` - the high-level `VoiceLlmSession` API used by
  applications (connect, stream audio in and out, observe state)
- `python/` - PyO3 bindings over the core building blocks

Audio flows through each side of a session in one direction of this pipeline:

```
mic -> process (AEC/NS/AGC) -> encode (Opus) -> network (SRTP)
network (SRTP) -> jitter buffer -> decode (Opus) -> playout
```

The send path captures PCM, cleans it up, compresses it, and ships it over
the secure transport. The receive path does the reverse, with the jitter
buffer absorbing network timing variance before decode and playout. See
[docs/architecture.md](docs/architecture.md) for the full picture.

## Latency Tuning

The main knobs, from codec to capture:

- **`CodecConfig`** (`llm_rtc_core::audio::codec`)
  - `bitrate`: 24 kbps default, clean speech with small packets
  - `frame_size_ms`: 10 ms default for low algorithmic latency; 20 ms reduces
    packet overhead when latency is less important
  - `complexity`: CPU vs. quality trade-off
  - DTX: stop transmitting during silence
  - FEC: in-band forward error correction for lossy links
- **`JitterBufferConfig`** (`llm_rtc_core::audio::jitter`)
  - `target_latency_ms`: minimum startup depth (5 ms by default); lower is snappier
  - `max_latency_ms`: hard ceiling for adaptive startup depth; valid packets
    are not discarded merely because they arrived early
- **`ProcessorConfig`** (`llm_rtc_core::audio::processor`)
  - `enable_aec`, `enable_ns`, `enable_agc`, `enable_vad`: WebRTC audio
    processing modules; AEC needs a far-end reference

All three default to voice-first, low-latency values.

## Roadmap

- STUN/TURN relay support for traversal through symmetric NATs
- WebRTC data channels for control/metadata beside the audio
- Adaptive bitrate driven by RTCP receiver reports
- Echo cancellation tuning (delay estimation, far-end reference handling)
- More SDKs (Node.js, WASM)

## Status

Early development. See `crates/*/src` and `examples/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT OR Apache-2.0
