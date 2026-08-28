FROM rust:1.95-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    cmake \
    espeak-ng \
    ffmpeg \
    libclang-dev \
    libopus-dev \
    libwebrtc-audio-processing-dev \
    nodejs \
    npm \
    pkg-config \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY benchmarks/neteq-comparison/package.json benchmarks/neteq-comparison/package-lock.json* /work/benchmarks/neteq-comparison/
RUN cd /work/benchmarks/neteq-comparison && npm install --omit=dev
RUN pip3 install --break-system-packages --no-cache-dir numpy scipy pystoi

COPY . /work
RUN cargo build --release -p llm-rtc-sdk --example neteq_trace_sender

ENTRYPOINT ["/bin/sh", "/work/benchmarks/neteq-comparison/run-in-container.sh"]
