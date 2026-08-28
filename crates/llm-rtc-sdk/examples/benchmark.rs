//! llm-rtc performance benchmark.
//!
//! Measures the metrics that matter for a low-latency voice LLM library:
//!   A. Codec CPU efficiency (realtime factor): how many seconds of audio can
//!      we encode/decode per second of wall time.
//!   B. Compression efficiency: achieved Opus bitrate vs configured, packet
//!      sizes, and DTX silence suppression.
//!   C. End-to-end pipeline latency: single-frame algorithmic latency through
//!      process -> encode -> jitter -> decode.
//!   D. Jitter buffer robustness under simulated network jitter + loss.
//!   E. Raw throughput (frames/sec, packets/sec).
//!
//! Run with: cargo run -p llm-rtc-sdk --example benchmark
#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use llm_rtc_core::audio::codec::{CodecConfig, OpusDecoder, OpusEncoder};
use llm_rtc_core::audio::jitter::{AudioPacket, JitterBuffer, JitterBufferConfig, JitterStats};
use llm_rtc_core::audio::pipeline::{AudioPipeline, AudioPipelineConfig};

const SAMPLE_RATE: u32 = 48_000;

/// Generate `seconds` of speech-like audio: harmonic-rich, with natural
/// amplitude modulation and periodic silence gaps (to exercise DTX).
fn speech_like(seconds: f32, sample_rate: u32) -> Vec<i16> {
    let n = (seconds * sample_rate as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Two harmonics (200 Hz + 400 Hz) for a speech-like spectrum.
            let mut s = 0.6 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.4 * (2.0 * std::f32::consts::PI * 400.0 * t).sin();
            // Amplitude envelope: 200 ms on, 100 ms off (speech cadence).
            let env = if (t * 3.333).fract() < 0.66 {
                1.0
            } else {
                0.02
            };
            s *= env;
            // Slow vibrato for realism.
            s *= 0.8 + 0.2 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
            (s * 8000.0) as i16
        })
        .collect()
}

fn fmt_bps(bps: f64) -> String {
    if bps >= 1_000_000.0 {
        format!("{:.2} Mbps", bps / 1_000_000.0)
    } else {
        format!("{:.1} kbps", bps / 1_000.0)
    }
}

fn bench_codec() {
    println!("\n=== A. CODEC CPU EFFICIENCY (realtime factor) ===");
    let audio = speech_like(10.0, SAMPLE_RATE); // 10 s of audio
    let cfg = CodecConfig::default();

    // Encode throughput.
    let encoder = OpusEncoder::new(cfg.clone()).unwrap();
    let spf = encoder.samples_per_frame();
    let frames: Vec<&[i16]> = audio.chunks(spf).collect();
    let n_frames = frames.len();

    let start = Instant::now();
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(n_frames);
    for f in &frames {
        packets.push(encoder.encode(f).unwrap());
    }
    let enc_wall = start.elapsed().as_secs_f64();
    let audio_secs = n_frames as f64 * spf as f64 / SAMPLE_RATE as f64;
    println!("encode: {n_frames} frames ({audio_secs:.1} s audio) in {enc_wall:.3} s  ");
    println!(
        "  realtime factor = {:.2}x   ({:.0} frames/s, {:.0} packets/s)",
        audio_secs / enc_wall,
        n_frames as f64 / enc_wall,
        n_frames as f64 / enc_wall,
    );

    // Decode throughput.
    let mut decoder = OpusDecoder::new(cfg).unwrap();
    let start = Instant::now();
    let mut decoded_samples = 0usize;
    for p in &packets {
        decoded_samples += decoder.decode(p).unwrap().len();
    }
    let dec_wall = start.elapsed().as_secs_f64();
    let dec_audio_secs = decoded_samples as f64 / SAMPLE_RATE as f64;
    println!("decode: {n_frames} packets in {dec_wall:.3} s  ");
    println!(
        "  realtime factor = {:.2}x   ({:.0} packets/s)",
        dec_audio_secs / dec_wall,
        n_frames as f64 / dec_wall,
    );
}

fn bench_compression() {
    println!("\n=== B. COMPRESSION EFFICIENCY ===");
    let cfg = CodecConfig::default();
    let encoder = OpusEncoder::new(cfg.clone()).unwrap();
    let spf = encoder.samples_per_frame();

    // Continuous speech.
    let audio = speech_like(10.0, SAMPLE_RATE);
    let frames: Vec<&[i16]> = audio.chunks(spf).collect();
    let mut total_bytes = 0usize;
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for f in &frames {
        let pkt = encoder.encode(f).unwrap();
        total_bytes += pkt.len();
        packets.push(pkt);
    }
    let audio_secs = frames.len() as f64 * spf as f64 / SAMPLE_RATE as f64;
    let bps = total_bytes as f64 * 8.0 / audio_secs;
    let min_pkt = packets.iter().map(|p| p.len()).min().unwrap();
    let max_pkt = packets.iter().map(|p| p.len()).max().unwrap();
    let avg_pkt = total_bytes as f64 / packets.len() as f64;
    println!("continuous speech (10 s):");
    println!(
        "  total encoded = {:.1} KB, achieved bitrate = {}",
        total_bytes as f64 / 1024.0,
        fmt_bps(bps)
    );
    println!(
        "  packet size: min={min_pkt} B, avg={avg_pkt:.1} B, max={max_pkt} B (configured 24 kbps)"
    );

    // DTX: 50% silence should suppress packets.
    let silent = vec![0i16; spf * 250]; // 5 s of pure silence
    let enc2 = OpusEncoder::new(CodecConfig {
        use_dtx: true,
        ..cfg.clone()
    })
    .unwrap();
    let mut dtx_packets = 0usize;
    let mut dtx_bytes = 0usize;
    for f in silent.chunks(spf) {
        let mut frame = f.to_vec();
        frame.resize(spf, 0);
        let pkt = enc2.encode(&frame).unwrap();
        dtx_packets += 1;
        dtx_bytes += pkt.len();
    }
    println!("DTX on 5 s of silence:");
    println!(
        "  {dtx_packets} frames sent, {dtx_bytes} bytes total ({:.1} B/s)",
        dtx_bytes as f64 / 5.0
    );
    println!("  (DTX should collapse silence to near-zero bitrate)");
}

fn bench_latency() {
    println!("\n=== C. PIPELINE COMPUTE COST ===");
    // This section measures synchronous compute cost, not wall-clock playout.
    // Disable startup buffering and use one synthetic RTP instant so the
    // jitter buffer does not intentionally pace the tight benchmark loop.
    let mut cfg = AudioPipelineConfig::default();
    cfg.jitter.target_latency_ms = 0;
    let mut pipe = AudioPipeline::new(cfg).unwrap();
    let codec_spf = 960; // 20 ms @ 48 kHz

    // Warm up the processor (it has internal state).
    let warm = vec![0i16; codec_spf];
    for _ in 0..20 {
        let mut w = warm.clone();
        let _ = pipe.process_outgoing(&mut w).unwrap();
    }

    // Measure single-frame latency: process -> encode -> jitter -> decode.
    let frame = speech_like(0.02, SAMPLE_RATE); // exactly one 20 ms frame
    let n = 1000;
    let start = Instant::now();
    let mut decoded_frames = 0usize;
    let mut seq = 0u16;
    for _ in 0..n {
        let mut f = frame.clone();
        let packets = pipe.process_outgoing(&mut f).unwrap();
        for pkt in &packets {
            let ap = AudioPacket {
                sequence_number: seq,
                timestamp: 0,
                payload: pkt.clone(),
            };
            seq = seq.wrapping_add(1);
            pipe.push_incoming(ap);
        }
        // Pop whatever is ready (low-latency jitter policy).
        while let Ok(Some(_)) = pipe.pop_decoded() {
            decoded_frames += 1;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let per_frame_us = wall * 1_000_000.0 / n as f64;
    println!("{n} frames round-tripped in {wall:.3} s");
    println!("  per-frame compute time = {per_frame_us:.1} us");
    println!(
        "  ({:.0} frames/s full pipeline throughput)",
        n as f64 / wall
    );
    println!("  decoded {decoded_frames}/{n} frames (jitter/low-latency policy)");
}

fn bench_jitter() {
    println!("\n=== D. JITTER BUFFER UNDER NETWORK STRESS ===");
    let cfg = JitterBufferConfig::default();
    let spf = 960;

    // Simulate a network: 500 frames, 30 ms mean jitter, 5% loss.
    let n = 500;
    let frame_ms = 20.0;
    let jitter_ms = 30.0;
    let loss_rate = 0.05;
    let mut rng_state = 0x12345678u32;
    let mut rng = move || {
        // xorshift
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        rng_state as f64 / u32::MAX as f64
    };

    // Arrival timeline: base 20ms spacing + gaussian-ish jitter.
    let mut arrival = vec![0.0f64; n];
    let mut t = 0.0;
    for slot in arrival.iter_mut() {
        let j = (rng() - 0.5) * 2.0 * jitter_ms;
        t += frame_ms + j;
        *slot = t;
        if rng() < loss_rate {
            *slot = f64::NEG_INFINITY; // lost
        }
    }

    // Mock clock: advance in lock-step with the playout timeline so the
    // jitter buffer's wall-clock grace windows align with the synthetic
    // network timing (deterministic, no real sleeping).
    let clock_t = std::sync::Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO));
    let clock = {
        let t = clock_t.clone();
        Box::new(move || std::time::Instant::now() + *t.lock().unwrap())
    };
    let mut jb = JitterBuffer::with_clock(cfg, clock);

    // Feed packets in arrival order; pop on a 20ms playout clock.
    let mut playout_t = 0.0f64;
    let mut emitted = 0u64;
    let mut max_added_latency = 0.0f64;
    let mut idx = 0usize;

    // Drain condition: all network frames delivered AND buffer drained.
    let buffered =
        |st: &JitterStats| st.packets_in > st.packets_out + st.packets_dropped + st.packets_late;

    // Playout runs for a bounded window: n frames spacing + margin to drain.
    let max_ticks = n + 50;
    let mut ticks = 0usize;
    while (idx < n || buffered(&jb.stats())) && ticks < max_ticks {
        // Deliver all packets whose arrival time has passed.
        while idx < n && arrival[idx] <= playout_t {
            if arrival[idx] == f64::NEG_INFINITY {
                idx += 1; // lost on the network
                continue;
            }
            let ap = AudioPacket {
                sequence_number: idx as u16,
                timestamp: (idx as u32) * spf as u32,
                payload: vec![1, 2, 3], // placeholder payload
            };
            jb.push(ap);
            idx += 1;
        }
        // Advance the mock clock to the current playout instant.
        *clock_t.lock().unwrap() = std::time::Duration::from_millis(playout_t as u64);
        // Pop at playout clock.
        while let Some(p) = jb.pop() {
            let arrival_delay = playout_t - arrival[p.sequence_number as usize];
            max_added_latency = max_added_latency.max(arrival_delay);
            emitted += 1;
        }
        playout_t += frame_ms;
        ticks += 1;
    }

    let st = jb.stats();
    println!(
        "simulated network: {n} frames, {jitter_ms} ms jitter, {:.0}% loss",
        loss_rate * 100.0
    );
    println!(
        "  emitted (played out) = {emitted} ({:.1}%)",
        emitted as f64 / n as f64 * 100.0
    );
    println!(
        "  dropped (lost/skipped) = {}, late = {}",
        st.packets_dropped, st.packets_late
    );
    println!("  jitter estimate = {:.1} ms", st.current_jitter_ms);
    println!("  max added latency = {max_added_latency:.1} ms");
    println!("  (low-latency policy: drops late packets rather than adding delay)");
}

fn bench_throughput() {
    println!("\n=== E. RAW THROUGHPUT ===");
    let cfg = CodecConfig::default();
    let encoder = OpusEncoder::new(cfg.clone()).unwrap();
    let mut decoder = OpusDecoder::new(cfg).unwrap();
    let spf = encoder.samples_per_frame();
    let audio = speech_like(1.0, SAMPLE_RATE);

    // Max encode rate (single-threaded, pure encode).
    let start = Instant::now();
    let mut count = 0usize;
    loop {
        for f in audio.chunks(spf) {
            encoder.encode(f).unwrap();
            count += 1;
        }
        if start.elapsed().as_secs_f64() >= 1.0 {
            break;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let frames_per_sec = count as f64 / wall;
    println!(
        "encode: {:.0} frames/s ({:.0} packets/s)",
        frames_per_sec, frames_per_sec
    );
    println!(
        "  = {:.0}x realtime (one 20ms frame each)",
        frames_per_sec * 0.020
    );

    // Max decode rate.
    let pkt = encoder.encode(&audio[..spf]).unwrap();
    let start = Instant::now();
    let mut count = 0usize;
    loop {
        decoder.decode(&pkt).unwrap();
        count += 1;
        if start.elapsed().as_secs_f64() >= 1.0 {
            break;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let fps = count as f64 / wall;
    println!(
        "decode: {:.0} packets/s ({:.0}x realtime)",
        fps,
        fps * 0.020
    );
}

fn main() {
    println!("llm-rtc benchmark");
    println!("=================");
    println!("sample rate: {SAMPLE_RATE} Hz, mono, 20 ms frames (default config)");

    bench_codec();
    bench_compression();
    bench_latency();
    bench_jitter();
    bench_throughput();

    println!("\nDone.");
}
