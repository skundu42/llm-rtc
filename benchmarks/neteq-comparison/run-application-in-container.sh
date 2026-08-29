#!/bin/sh
set -eu

RESULTS_DIR=${1:-/results/application}
mkdir -p "$RESULTS_DIR"

if [ ! -s "$RESULTS_DIR/speech-reference.pcm" ]; then
  espeak-ng -s 165 -w /tmp/application-speech.wav \
    "Real time voice agents need steady timing. \
The receiver must absorb network jitter. \
Error correction should preserve clear and continuous conversation."
  ffmpeg -hide_banner -loglevel error -y -i /tmp/application-speech.wav \
    -af apad -f s16le -acodec pcm_s16le -ac 1 -ar 48000 -t 10 \
    "$RESULTS_DIR/speech-reference.pcm"
  printf '%s\n' 'espeak-ng 165 wpm' >"$RESULTS_DIR/speech-source.txt"
fi

speech_dir="$RESULTS_DIR/speech"
mkdir -p "$speech_dir"
/work/target/release/examples/neteq_trace_sender \
  llm-only severe "$RESULTS_DIR/speech-reference.pcm" "$speech_dir" 115 5
node /work/benchmarks/neteq-comparison/neteq_receiver.js \
  severe "$RESULTS_DIR/speech-reference.pcm" "$speech_dir" 3

if [ "${APPLICATION_ONLY:-}" = "speech" ]; then
  exit 0
fi

phrases="stop wait-please hold-on cancel-that question"
for phrase in $phrases; do
  case "$phrase" in
    stop) text="Stop" ;;
    wait-please) text="Wait please" ;;
    hold-on) text="Hold on" ;;
    cancel-that) text="Cancel that" ;;
    question) text="I have a question" ;;
  esac
  espeak-ng -s 180 -w "/tmp/barge-$phrase.wav" "$text"
  ffmpeg -hide_banner -loglevel error -y -i "/tmp/barge-$phrase.wav" \
    -af "silenceremove=start_periods=1:start_duration=0.01:start_threshold=-45dB" \
    -f s16le -acodec pcm_s16le -ac 1 -ar 48000 -t 1.2 "/tmp/barge-$phrase.pcm"
done

python3 - <<'PY'
from pathlib import Path
import numpy as np

sample_rate = 48_000
output = np.zeros(sample_rate * 10, dtype=np.int16)
items = [
    ("stop", 0.8),
    ("wait-please", 2.6),
    ("hold-on", 4.4),
    ("cancel-that", 6.2),
    ("question", 8.0),
]
for name, offset_seconds in items:
    phrase = np.fromfile(f"/tmp/barge-{name}.pcm", dtype="<i2")
    start = round(offset_seconds * sample_rate)
    count = min(len(phrase), len(output) - start)
    mixed = output[start:start + count].astype(np.int32) + phrase[:count].astype(np.int32)
    output[start:start + count] = np.clip(mixed, -32768, 32767).astype(np.int16)
output.tofile(Path("/results/application/barge-reference.pcm"))
PY

barge_dir="$RESULTS_DIR/barge"
mkdir -p "$barge_dir"
/work/target/release/examples/neteq_trace_sender \
  llm-only severe "$RESULTS_DIR/barge-reference.pcm" "$barge_dir" 115 5
node /work/benchmarks/neteq-comparison/neteq_receiver.js \
  severe "$RESULTS_DIR/barge-reference.pcm" "$barge_dir" 3

for calls in 1 10 25 50; do
  llm_load_dir="$RESULTS_DIR/load/llm-$calls"
  neteq_load_dir="$RESULTS_DIR/load/neteq-$calls"
  mkdir -p "$llm_load_dir" "$neteq_load_dir"
  /work/target/release/examples/neteq_trace_sender \
    llm-load severe "$RESULTS_DIR/speech-reference.pcm" "$llm_load_dir" 115 5 "$calls" 7
  node /work/benchmarks/neteq-comparison/neteq_receiver.js \
    severe "$RESULTS_DIR/speech-reference.pcm" "$neteq_load_dir" 3 "$calls" load
done
