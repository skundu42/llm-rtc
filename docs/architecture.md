# Architecture

This document describes how llm-rtc is structured and why each layer is
tuned the way it is. The guiding constraint throughout is end-to-end
latency: a voice conversation with an LLM feels broken once round-trip
latency climbs past a few hundred milliseconds, so every component is
tuned for speed first and resilience second.

## Crate layout

The workspace has two crates plus the Python bindings:

```
crates/
  llm-rtc-core/        # engine, no opinion about LLMs
    src/
      lib.rs
      audio/
        codec.rs       # Opus encoder/decoder wrapper (voice-tuned)
        jitter.rs      # low-latency jitter buffer (packet reordering + playout)
        processor.rs   # WebRTC audio processing: AEC, NS, AGC, VAD
        pipeline.rs    # wires capture/process/encode/playout together
      peer.rs          # WebRTC peer connection (offer/answer, tracks)
      transport/
        dtls.rs        # DTLS handshake for key exchange
        srtp.rs        # SRTP packet protection for media
  llm-rtc-sdk/         # application-facing API
    src/
      lib.rs
      session.rs       # VoiceLlmSession: connect, on_remote_audio, send, stats
python/                # PyO3 bindings over the core building blocks
```

The layering is strict: `llm-rtc-sdk` depends on `llm-rtc-core`, never the
other way around. Core contains no policy about who is on the other end of
the call; the SDK frames it as a voice LLM session and exposes an async,
callback-oriented API. Python bindings are thin PyO3 wrappers over the
core types (Opus codecs, audio processor, jitter buffer, pipeline) so
Python apps can reuse the exact same tuned primitives.

## Audio pipeline data flow

Each session runs one full-duplex stream: user audio flows up to the
model, model audio flows back down to the user. Conceptually the signal
passes through this chain in both directions:

```
capture -> process -> encode -> network -> jitter -> decode -> playout
```

### Send path (user to model)

1. **Capture**. The platform delivers 16-bit mono PCM at 48 kHz.
2. **Process** (`audio/processor.rs`). The WebRTC audio processing module
   runs AEC (using the far-end reference, i.e. what the user is hearing),
   noise suppression, and AGC. The goal is a clean, consistent signal for
   the model's ASR front end; garbage in means transcription errors out.
3. **Encode** (`audio/codec.rs`). Opus at a voice-optimized bitrate. The
   encoder emits one packet per frame.
4. **Network** (`transport/`). Each frame becomes an RTP packet, protected
   with SRTP, and leaves through the peer connection. Frame sizes are kept
   small so packets are small and retransmission losses are cheap.

### Receive path (model to user)

1. **Network**. SRTP-authenticated RTP packets arrive and are decrypted.
2. **Jitter buffer** (`audio/jitter.rs`). Packets are resequenced by
   sequence number and held until playout depth is reached. Missing
   packets are skipped after a bounded wait rather than stalling the
   stream.
3. **Decode** (`audio/codec.rs`). Opus packets are decoded back to PCM,
   optionally using the decoder's FEC mode to conceal earlier losses.
4. **Playout**. Decoded PCM is handed to the output device, or in the SDK
   to the `on_remote_audio` callback, which is where an application would
   feed model audio (or its own TTS, or analyzer) downstream.

The two paths are independent, which is what makes the session truly
full-duplex: the user can keep talking (barge-in) while model audio is
playing, and AEC keeps the microphone from picking up that playback.

## Jitter buffer: the low-latency policy

Conventional jitter buffers maximize the chance that every packet arrives
before it is needed. They grow under jitter and hold tens or hundreds of
milliseconds of audio. For a voice conversation that growth is fatal to
interactivity, so llm-rtc's buffer implements the opposite policy:

- **Adaptive startup target.** `JitterBufferConfig::target_latency_ms` is the
  minimum playout depth. Before playout begins, the target grows toward
  `max_latency_ms` using four times the RFC 3550 inter-arrival jitter estimate.
  Once playout starts, RTP deadlines stay fixed on the 20 ms clock.
- **Bounded wait for loss.** If the next expected packet has not arrived
  within the target window, `pop()` stops waiting, declares it lost, and
  advances the sequence. The caller plays a gap (the decoder's FEC/PLC can
  fill in) instead of accumulating delay.
- **Hard ceiling.** `max_latency_ms` caps actual/projected packet residence,
  measured from arrival to its RTP-derived deadline; `max_packets` is a memory
  safety valve. Discarded packets remain queued as missing playout slots, so
  the decoder produces PLC instead of silently shortening the stream.

The trade-off is explicit: under sustained packet loss or severe jitter
this policy produces audible gaps where a classical buffer would produce a
growing delay. For turn-taking voice conversation, a short gap is far less
damaging than a growing one, because delay breaks the rhythm of
conversation (talking over each other, awkward pauses) in ways users
notice immediately.

## Opus voice tuning rationale

The codec defaults in `CodecConfig` are chosen for speech, not music:

- **24 kbps mono.** Opus is perceptually tuned for speech at this rate.
  Higher bitrates add packet size (and therefore loss cost and serialize
  latency) with little audible benefit for voice.
- **20 ms frames.** This is the classic WebRTC trade-off between
  algorithmic latency and compression efficiency. Shorter frames reduce
  the codec's own latency contribution but waste bits on per-frame
  overhead; 20 ms keeps packets small without a big bitrate penalty.
- **DTX (discontinuous transmission).** During silence no audio frames are
  sent. In a conversation one side is usually quiet, so this can cut
  average bandwidth roughly in half and, more importantly for latency,
  keeps the network path unloaded.
- **FEC (in-band forward error correction).** Opus can carry a lower-fidelity
  copy of the previous frame inside each packet. When a packet is lost the
  decoder can reconstruct it from the next packet's FEC data, which is a
  loss recovery that costs zero round trips.
- **Complexity** is exposed as a knob: lower complexity trades a little
  quality for CPU, which matters on embedded or heavily shared machines.

## WebRTC peer flow

Establishing a session follows the standard WebRTC handshake, kept minimal:

1. **Signaling.** The application exchanges an SDP offer/answer out of band
   (WebSocket, HTTP, however it likes). Core builds the offer, sets the
   remote answer, and negotiates a single Opus audio track in each
   direction.
2. **ICE.** Candidates are gathered and checked to find a direct path
   between the peers. (Traversal via STUN/TURN relays is on the roadmap.)
3. **DTLS** (`transport/dtls.rs`). The peers run a DTLS handshake over the
   ICE channel to negotiate session keys.
4. **SRTP** (`transport/srtp.rs`). Those keys are exported to protect the
   RTP media: every packet is authenticated and encrypted.
5. **Media.** Once the transport is up, the send pipeline feeds frames in
   and the receive pipeline drains them out, as described above.

The SDK's `VoiceLlmSession` wraps all of this: `new` constructs the peer
and pipeline, connection-state callbacks surface the handshake progress,
`on_remote_audio` delivers decoded model audio, and `jitter_stats` /
`processor_stats` expose the live health of the buffer and the DSP so
applications can observe (and tune) latency in production.
