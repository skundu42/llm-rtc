#!/bin/sh
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
RESULTS_DIR="$REPO_DIR/benchmarks/neteq-comparison/results/application"
mkdir -p "$RESULTS_DIR"

say -v Samantha -r 165 -o /tmp/llm-rtc-application-speech.aiff \
  "Real time voice agents need steady timing. \
The receiver must absorb network jitter. \
Error correction should preserve clear and continuous conversation."
ffmpeg -hide_banner -loglevel error -y -i /tmp/llm-rtc-application-speech.aiff \
  -af apad -f s16le -acodec pcm_s16le -ac 1 -ar 48000 -t 10 \
  "$RESULTS_DIR/speech-reference.pcm"
printf '%s\n' 'macOS Samantha 165 wpm' >"$RESULTS_DIR/speech-source.txt"

docker desktop start >/dev/null
docker build \
  -f "$REPO_DIR/benchmarks/neteq-comparison/Dockerfile" \
  -t llm-rtc-neteq-benchmark:local \
  "$REPO_DIR"

docker run --rm \
  --entrypoint /bin/sh \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$RESULTS_DIR:/results/application" \
  llm-rtc-neteq-benchmark:local \
  /work/benchmarks/neteq-comparison/run-application-in-container.sh \
  /results/application

uv run "$REPO_DIR/benchmarks/neteq-comparison/application_benchmark.py" "$RESULTS_DIR"
