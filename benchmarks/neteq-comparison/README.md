# llm-rtc vs Chromium NetEq

This benchmark sends identical Opus RTP payloads, sequence numbers, RTP
timestamps, packet losses, arrival delays, and reorderings through llm-rtc and
Chromium NetEq. The NetEq receiver is provided by the ARM64 libwebrtc binary in
`@roamhq/wrtc`.

Run it with Docker Desktop only:

```sh
./benchmarks/neteq-comparison/run.sh
```

Run the voice-application benchmark separately:

```sh
./benchmarks/neteq-comparison/run-application.sh
```

That benchmark uses the severe trace, a 5 ms startup target, and the
latency-normalized 115 ms llm-rtc ceiling. It records:

- Whisper base.en word-error rate on a deterministic macOS system-voice
  utterance padded to ten seconds.
- WebRTC VAD detection and miss rates for five isolated barge-in phrases.
- Source-speech-end to completed batch-ASR transcript latency.
- Wideband PESQ MOS-LQO as an objective speech-quality estimate.
- CPU and RSS at 1, 10, 25, and 50 simultaneous calls.

Application results are written to `results/application/application-summary.json`
and `results/application/application-load.csv`. The script uses `uv` for the
Python analysis environment and a cached or downloaded
`Systran/faster-whisper-base.en` model.

The turn measurement intentionally stops at the transcript. This repository
does not contain the production LLM inference or TTS service, so reporting a
full user-speech-to-agent-audio number would require choosing and measuring
those application-specific components. PESQ MOS-LQO is an objective estimator,
not a substitute for a blinded human listening panel.

Three deterministic 10-second traces of 10 ms Opus packets are exercised
three times each:

- `clean`: 20 ms base network delay, +/-2 ms jitter, no loss.
- `moderate`: 30 ms base delay, +/-15 ms jitter, 5% loss.
- `severe`: 50 ms base delay, +/-40 ms jitter, 10% loss.

llm-rtc starts at a 5 ms target and may adapt up to its 120 ms RTP-deadline
ceiling. NetEq manages its delay through the standard libwebrtc receiver.

Results are written to `benchmarks/neteq-comparison/results/summary.json`.
The severe trace is also replayed with 120, 115, 110, 100, 90, and 80 ms
llm-rtc latency ceilings. Its latency-normalized data is written to
`results/latency-sweep.json` and `results/latency-sweep.csv`.

Metrics are computed over the 1,000 content frames (10 seconds):

- Continuity is the share of expected playout slots that produced an audio
  frame. For NetEq, missed 10 ms audio-sink callbacks are treated as gaps.
- Concealment is explicit llm-rtc PLC frames or NetEq's non-silent concealed
  samples. FEC output is not counted as concealment.
- Playout delay is packet residence time for llm-rtc and the per-sample
  `jitterBufferDelay / jitterBufferEmittedCount` delta for NetEq.
- Audio quality uses time-aligned STOI, SI-SDR, and segmental SNR against the
  common pre-encode reference signal.
- NetEq figures are medians of three runs. llm-rtc CPU is process user+system
  time, reported as the median of 21 deterministic offline replays.

CPU and memory describe the receive implementation used by the harness:
llm-rtc's jitter/decode loop versus the Node + libwebrtc receiver process.
They are useful operational footprints, but not isolated microbenchmarks of
the jitter-buffer classes alone.

NetEq concealment excludes `silentConcealedSamples` so startup and deliberate
post-stream silence do not count as audible packet-loss concealment.
