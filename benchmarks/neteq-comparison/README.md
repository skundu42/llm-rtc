# llm-rtc vs Chromium NetEq

This benchmark sends identical Opus RTP payloads, sequence numbers, RTP
timestamps, packet losses, arrival delays, and reorderings through llm-rtc and
Chromium NetEq. The NetEq receiver is provided by the ARM64 libwebrtc binary in
`@roamhq/wrtc`.

Run it with Docker Desktop only:

```sh
./benchmarks/neteq-comparison/run.sh
```

Three deterministic 10-second traces are exercised three times each:

- `clean`: 20 ms base network delay, +/-2 ms jitter, no loss.
- `moderate`: 30 ms base delay, +/-15 ms jitter, 5% loss.
- `severe`: 50 ms base delay, +/-40 ms jitter, 10% loss.

llm-rtc starts at a 40 ms target and may adapt up to its 120 ms RTP-deadline
ceiling. NetEq manages its delay through the standard libwebrtc receiver.

Results are written to `benchmarks/neteq-comparison/results/summary.json`.

Metrics are computed over the 500 content frames (10 seconds):

- Continuity is the share of expected playout slots that produced an audio
  frame. For NetEq, missed 10 ms audio-sink callbacks are treated as gaps.
- Concealment is explicit llm-rtc PLC frames or NetEq's non-silent concealed
  samples. FEC output is not counted as concealment.
- Playout delay is packet residence time for llm-rtc and the per-sample
  `jitterBufferDelay / jitterBufferEmittedCount` delta for NetEq.
- Audio quality uses time-aligned STOI, SI-SDR, and segmental SNR against the
  common pre-encode reference signal.
- NetEq figures are medians of three runs. llm-rtc CPU is the median of 21
  deterministic offline replays.

CPU and memory describe the receive implementation used by the harness:
llm-rtc's jitter/decode loop versus the Node + libwebrtc receiver process.
They are useful operational footprints, but not isolated microbenchmarks of
the jitter-buffer classes alone.

NetEq concealment excludes `silentConcealedSamples` so startup and deliberate
post-stream silence do not count as audible packet-loss concealment.
