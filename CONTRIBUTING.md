# Contributing to llm-rtc

Thanks for your interest in contributing! This project is early stage, so
the best first step for anything non-trivial is to open an issue describing
the change.

## Setting up

Requirements:

- Rust (stable toolchain)
- Build deps (Debian/Ubuntu): `sudo apt-get install -y cmake pkg-config libwebrtc-audio-processing-dev`
- Python 3 + maturin, only if you touch the `python/` bindings

Build and run the test suite:

```sh
cargo build --workspace
cargo test --workspace
```

## Before submitting a PR

Run all three checks locally; CI enforces exactly these:

```sh
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## Guidelines

- Keep the low-latency focus: changes should not trade away end-to-end
  latency for robustness without a strong reason and a tunable knob.
- Add or update tests for behavior changes.
- Keep config defaults voice-first; if you change a default, say so in the PR.
- Update `README.md` or `docs/` when you add features or change public APIs.

## Submitting

1. Fork and create a feature branch from `main`.
2. Make your changes and verify the checks above pass.
3. Open a PR with a short description of the what and the why.
4. CI runs on every PR; keep it green.

## License

By contributing you agree that your contributions are licensed under the
same license as the project (MIT OR Apache-2.0).
