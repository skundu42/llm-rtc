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

use std::hint::black_box;
use std::time::Instant;

use llm_rtc_core::audio::codec::{CodecConfig, OpusDecoder, OpusEncoder};
use llm_rtc_core::audio::jitter::{AudioPacket, JitterBuffer, JitterBufferConfig, PlayoutEvent};
use llm_rtc_core::audio::pipeline::{AudioPipeline, AudioPipelineConfig};
use llm_rtc_core::audio::processor::{AudioProcessor, ProcessorConfig};

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
    let avg_capacity =
        packets.iter().map(Vec::capacity).sum::<usize>() as f64 / packets.len() as f64;
    println!("continuous speech (10 s):");
    println!(
        "  total encoded = {:.1} KB, achieved bitrate = {}",
        total_bytes as f64 / 1024.0,
        fmt_bps(bps)
    );
    println!(
        "  packet size: min={min_pkt} B, avg={avg_pkt:.1} B, max={max_pkt} B (configured 24 kbps)"
    );
    println!("  allocated packet capacity: avg={avg_capacity:.1} B");

    // DTX: 50% silence should suppress packets.
    let silent = vec![0i16; SAMPLE_RATE as usize * 5];
    let enc2 = OpusEncoder::new(CodecConfig {
        use_dtx: true,
        ..cfg.clone()
    })
    .unwrap();
    let mut dtx_packets = 0usize;
    let mut dtx_bytes = 0usize;
    for frame in silent.chunks_exact(spf) {
        let pkt = enc2.encode(frame).unwrap();
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
    let frame_ms = cfg.codec.frame_size_ms;
    let codec_spf =
        (cfg.codec.sample_rate as f32 * frame_ms / 1_000.0) as usize * cfg.codec.channels as usize;
    let mut pipe = AudioPipeline::new(cfg).unwrap();

    // Warm up the processor (it has internal state).
    let warm = vec![0i16; codec_spf];
    for _ in 0..20 {
        let mut w = warm.clone();
        let _ = pipe.process_outgoing(&mut w).unwrap();
    }

    // Measure single-frame latency: process -> encode -> jitter -> decode.
    let frame = speech_like(frame_ms / 1_000.0, SAMPLE_RATE);
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

    // Capture APIs may deliver sub-frame chunks. They should be coalesced
    // before processing/encoding rather than padded into extra packets.
    let mut fragmented_cfg = AudioPipelineConfig::default();
    fragmented_cfg.jitter.target_latency_ms = 0;
    let mut fragmented_pipe = AudioPipeline::new(fragmented_cfg).unwrap();
    let mut fragments: Vec<_> = frame
        .chunks(codec_spf / 4)
        .map(|chunk| chunk.to_vec())
        .collect();
    let start = Instant::now();
    let mut fragmented_packets = 0usize;
    for _ in 0..n {
        for fragment in &mut fragments {
            fragmented_packets += fragmented_pipe.process_outgoing(fragment).unwrap().len();
        }
    }
    let fragmented_wall = start.elapsed().as_secs_f64();
    println!(
        "  fragmented capture = {fragmented_packets} packets, {:.1} us/source frame",
        fragmented_wall * 1_000_000.0 / n as f64
    );

    // File and telephony adapters may deliver audio in large batches even
    // though the processor consumes fixed 10 ms frames internally.
    let render_batch = speech_like(1.0, SAMPLE_RATE);
    let render_frames = render_batch.len() / codec_spf;
    let render_batches = 200;
    let mut render_processor = AudioProcessor::new(ProcessorConfig::default()).unwrap();
    render_processor.process_render(&render_batch).unwrap();
    let start = Instant::now();
    for _ in 0..render_batches {
        render_processor.process_render(&render_batch).unwrap();
    }
    println!(
        "  batched AEC render = {:.1} us/frame",
        start.elapsed().as_secs_f64() * 1_000_000.0 / (render_frames * render_batches) as f64
    );

    let mut capture_batch = render_batch;
    let mut capture_processor = AudioProcessor::new(ProcessorConfig::default()).unwrap();
    capture_processor.process(&mut [0]).unwrap();
    capture_processor.process(&mut capture_batch).unwrap();
    let start = Instant::now();
    for _ in 0..render_batches {
        capture_processor.process(&mut capture_batch).unwrap();
    }
    println!(
        "  partial batched capture = {:.1} us/frame",
        start.elapsed().as_secs_f64() * 1_000_000.0 / (render_frames * render_batches) as f64
    );
}

fn bench_jitter() {
    println!("\n=== D. JITTER BUFFER UNDER NETWORK STRESS ===");
    let cfg = JitterBufferConfig::default();
    let frame_ms = f64::from(cfg.frame_size_ms);
    let spf = (f64::from(cfg.sample_rate) * frame_ms / 1_000.0) as usize;

    // Simulate ten seconds of audio with 30 ms jitter and 5% loss.
    let n = (10_000.0 / frame_ms) as usize;
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

    // Independent network jitter around each source timestamp. Sort by
    // arrival because jitter can reorder packets.
    let base_delay_ms = 50.0;
    let mut arrival_by_frame = vec![None; n];
    let mut arrivals = Vec::with_capacity(n);
    for (frame, slot) in arrival_by_frame.iter_mut().enumerate() {
        if rng() < loss_rate {
            continue;
        }
        let jitter = (rng() - 0.5) * 2.0 * jitter_ms;
        let arrival = (frame as f64 * frame_ms + base_delay_ms + jitter).max(0.0);
        *slot = Some(arrival);
        arrivals.push((arrival, frame));
    }
    arrivals.sort_by(|left, right| left.0.total_cmp(&right.0));

    // Mock clock: advance in lock-step with the playout timeline so the
    // jitter buffer's wall-clock grace windows align with the synthetic
    // network timing (deterministic, no real sleeping).
    let clock_t = std::sync::Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO));
    let clock_base = std::time::Instant::now();
    let clock = {
        let t = clock_t.clone();
        Box::new(move || clock_base + *t.lock().unwrap())
    };
    let mut jb = JitterBuffer::with_clock(cfg, clock);

    // Feed packets in arrival order; pop on the configured playout clock.
    let mut playout_t = 0.0f64;
    let mut normal = 0u64;
    let mut recovered = 0u64;
    let mut missing = 0u64;
    let mut max_added_latency = 0.0f64;
    let mut next_arrival = 0usize;

    // Match the SDK's half-frame recovery polling and leave a one-second tail
    // for the final buffered packet.
    let poll_ms = (frame_ms / 2.0).max(1.0);
    let max_ticks =
        ((n as f64 * frame_ms + base_delay_ms + jitter_ms + 1_000.0) / poll_ms) as usize;
    let mut ticks = 0usize;
    while (next_arrival < arrivals.len() || jb.has_pending()) && ticks < max_ticks {
        // Deliver all packets whose arrival time has passed.
        while next_arrival < arrivals.len() && arrivals[next_arrival].0 <= playout_t {
            let frame = arrivals[next_arrival].1;
            let ap = AudioPacket {
                sequence_number: frame as u16,
                timestamp: (frame as u32) * spf as u32,
                payload: vec![1, 2, 3], // placeholder payload
            };
            jb.push(ap);
            next_arrival += 1;
        }
        // Advance the mock clock to the current playout instant.
        *clock_t.lock().unwrap() = std::time::Duration::from_millis(playout_t as u64);
        if let Some(event) = jb.pop_event() {
            match event {
                PlayoutEvent::Packet(packet) => {
                    let arrival = arrival_by_frame[packet.sequence_number as usize]
                        .expect("played packet has an arrival time");
                    max_added_latency = max_added_latency.max(playout_t - arrival);
                    normal += 1;
                }
                PlayoutEvent::RecoveredWithNextPacket { .. } => {
                    recovered += 1;
                }
                PlayoutEvent::Missing { .. } => missing += 1,
            }
        }
        playout_t += poll_ms;
        ticks += 1;
    }

    let st = jb.stats();
    let emitted = normal + recovered + missing;
    println!(
        "simulated network: {n} frames, {jitter_ms} ms jitter, {:.0}% loss",
        loss_rate * 100.0
    );
    println!(
        "  emitted slots = {emitted} ({:.1}%), normal={normal}, fec={recovered}, plc={missing}",
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
    let frame_seconds = f64::from(cfg.frame_size_ms) / 1_000.0;
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
        "  = {:.0}x realtime ({:.0} ms frames)",
        frames_per_sec * frame_seconds,
        frame_seconds * 1_000.0,
    );

    let now = std::time::Instant::now();
    let mut jitter = JitterBuffer::with_clock(
        JitterBufferConfig {
            target_latency_ms: 0,
            ..JitterBufferConfig::default()
        },
        Box::new(move || now),
    );
    let packets = 1_000_000;
    let start = Instant::now();
    for sequence in 0..packets {
        black_box(jitter.push(AudioPacket {
            sequence_number: sequence as u16,
            timestamp: 0,
            payload: Vec::new(),
        }));
        black_box(jitter.pop());
    }
    println!(
        "jitter push/pop: {:.1} ns/packet",
        start.elapsed().as_secs_f64() * 1_000_000_000.0 / packets as f64
    );

    let batches = 100_000;
    let batch_size = 8;
    let start = Instant::now();
    for _ in 0..batches {
        jitter.clear();
        for sequence in (0..batch_size).rev() {
            black_box(jitter.push(AudioPacket {
                sequence_number: sequence as u16,
                timestamp: 0,
                payload: Vec::new(),
            }));
        }
        while black_box(jitter.pop()).is_some() {}
    }
    println!(
        "jitter 8-packet reordered burst: {:.1} ns/packet",
        start.elapsed().as_secs_f64() * 1_000_000_000.0 / (batches * batch_size) as f64
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
        fps * frame_seconds
    );
}

fn main() {
    println!("llm-rtc benchmark");
    println!("=================");
    println!(
        "sample rate: {SAMPLE_RATE} Hz, mono, {} ms frames (default config)",
        CodecConfig::default().frame_size_ms
    );

    bench_codec();
    bench_compression();
    bench_latency();
    bench_jitter();
    bench_throughput();

    println!("\nDone.");
}
