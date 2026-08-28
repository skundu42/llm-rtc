#!/bin/sh
set -eu

RESULTS_DIR=${1:-/results}
mkdir -p "$RESULTS_DIR"

espeak-ng -s 165 -w /tmp/benchmark-speech.wav \
  "Real time conversation depends on steady timing as much as raw speed. \
The receiver must absorb ordinary network variation without making every response feel late. \
When a packet disappears, forward error correction or concealment should preserve a continuous voice. \
This deterministic passage supplies varied speech sounds for an objective comparison of two playout engines. \
Real time conversation depends on steady timing as much as raw speed. \
The receiver must absorb ordinary network variation without making every response feel late."

ffmpeg -hide_banner -loglevel error -y -i /tmp/benchmark-speech.wav \
  -f s16le -acodec pcm_s16le -ac 1 -ar 48000 -t 10 "$RESULTS_DIR/reference.pcm"

for profile in clean moderate severe; do
  profile_dir="$RESULTS_DIR/$profile"
  mkdir -p "$profile_dir"
  /work/target/release/examples/neteq_trace_sender \
    llm-only "$profile" "$RESULTS_DIR/reference.pcm" "$profile_dir"
  node /work/benchmarks/neteq-comparison/neteq_receiver.js \
    "$profile" "$RESULTS_DIR/reference.pcm" "$profile_dir" 3
done

python3 /work/benchmarks/neteq-comparison/analyze.py "$RESULTS_DIR"
