# llm-rtc

Low-latency, full-duplex WebRTC audio for real-time voice AI.

`llm-rtc` is a Rust-first audio transport library for streaming microphone
audio to a voice model and returning synthesized audio with minimal buffering.
It combines WebRTC transport, voice-tuned Opus, audio processing, and
deadline-driven playout behind a high-level session API.

> **Project status:** early development. APIs and performance characteristics
> may change before the first stable release.

## Highlights

- 48 kHz mono Opus with 10 ms frames, DTX, and in-band FEC
- Adaptive jitter buffering with a 5 ms startup target
- Packet-loss concealment and FEC recovery at fixed RTP playout deadlines
- Acoustic echo cancellation, noise suppression, automatic gain control, and
  voice activity detection
- Full-duplex WebRTC sessions over ICE, DTLS, and SRTP
- Reusable hot-path buffers to reduce allocations and scheduling overhead
- Rust core and high-level SDK, with Python bindings for the audio components

## Performance

The repository includes deterministic transport and voice-application
benchmarks against Chromium NetEq. The current 10 ms-frame application run
uses a 10-second severe network trace with 50 ms base delay, ±40 ms jitter,
10% packet loss, a 5 ms startup target, and a 115 ms latency ceiling.

| Metric | llm-rtc | Chromium NetEq |
| --- | ---: | ---: |
| Audio continuity | 100% | 100% |
| Median playout delay | 36.47 ms | 76.67 ms |
| p95 playout delay | 71.75 ms | 105.00 ms |
| p99 playout delay | 74.29 ms | 113.33 ms |
| Median barge-in detection | 85.00 ms | 122.42 ms |
| Speech end to completed batch transcript | 510.14 ms | 576.58 ms |
| Whisper `base.en` word error rate | 0% | 0% |
| PESQ wideband MOS-LQO | 1.779 | 1.658 |

These results measure the components present in this repository. Transcript
latency ends after batch ASR and does not include LLM inference or text-to-
speech. PESQ is an objective estimate, and the load comparison includes
different runtime boundaries, so it should not be read as an isolated
jitter-buffer CPU comparison.

Reproduce the transport comparison and application benchmark with Docker:

```sh
./benchmarks/neteq-comparison/run.sh
./benchmarks/neteq-comparison/run-application.sh
```

See the [raw application results](benchmarks/neteq-comparison/results/application-frame10/application-summary.json)
and [full methodology](benchmarks/neteq-comparison/README.md). For a local core
benchmark, run:

```sh
cargo run --release -p llm-rtc-sdk --example benchmark
```

## Architecture

Audio travels through two independent paths:

```text
microphone -> AEC/NS/AGC -> Opus encode -> SRTP network
SRTP network -> jitter/FEC/PLC -> Opus decode -> playout
```

Incoming packets and playout scheduling run independently. The receiver adds
packets to the jitter buffer while a deadline-driven task emits one frame per
RTP slot, including recovered or concealed frames when packets are missing.
This prevents packet arrival timing from directly controlling audio delivery.

The workspace contains:

- [`llm-rtc-core`](crates/llm-rtc-core): WebRTC peer, codec, audio processor,
  jitter buffer, and audio pipeline
- [`llm-rtc-sdk`](crates/llm-rtc-sdk): high-level `VoiceLlmSession` API
- [`python`](python): PyO3 bindings for the codec, processor, jitter buffer,
  and pipeline

See [`docs/architecture.md`](docs/architecture.md) for implementation details.

## Getting started

### Requirements

- Stable Rust toolchain
- CMake and `pkg-config`
- WebRTC audio-processing development headers

On Debian or Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y cmake pkg-config libwebrtc-audio-processing-dev
```

Build and test the workspace:

```sh
cargo build --workspace
cargo test --workspace
```

Run the in-process voice pipeline example:

```sh
cargo run -p llm-rtc-sdk --example voice_pipeline
```

## Python bindings

The Python package is built locally with
[`maturin`](https://www.maturin.rs):

```sh
python3 -m pip install maturin
cd python
python3 -m maturin build --release
python3 -m pip install target/wheels/llm_rtc-*.whl
```

The bindings expose the codec, audio processor, jitter buffer, and complete
audio pipeline:

```python
from llm_rtc import CodecConfig, OpusEncoder

encoder = OpusEncoder(CodecConfig())
packets = encoder.encode_frames(pcm_i16)
```

`pcm_i16` must contain signed 16-bit mono PCM sampled at 48 kHz when using the
default configuration.

## Latency and quality tuning

The defaults favor interactive voice latency. Tune them only after measuring
with network conditions representative of production.

| Setting | Default | Trade-off |
| --- | ---: | --- |
| Opus bitrate | 24 kbps | Higher values improve quality and increase bandwidth |
| Opus frame size | 10 ms | 20 ms reduces packet overhead but adds packetization latency |
| Opus complexity | 0 | Higher values use more CPU for encoding quality |
| Jitter startup target | 5 ms | Higher values absorb more jitter before playout starts |
| Jitter latency ceiling | 120 ms | Lower values reduce delay but increase late drops |
| DTX / FEC | Enabled | DTX saves silence bandwidth; FEC improves lossy-link recovery |

AEC requires far-end playout audio as a reference. Applications without that
reference should disable AEC rather than feeding unrelated audio.

## Roadmap

- STUN/TURN relay support for symmetric NAT traversal
- WebRTC data channels for control and metadata
- RTCP-driven adaptive bitrate
- Improved echo-delay estimation and reference handling
- Node.js and WebAssembly SDKs

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Licensed under either the MIT License or Apache License 2.0.
