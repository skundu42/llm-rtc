//! End-to-end low-latency voice pipeline demo, without a network peer.
//!
//! This example exercises the full audio round-trip that a voice LLM call
//! performs, purely in-process:
//!
//! ```text
//! mic PCM -> process (AEC/NS/AGC) -> Opus encode -> (network)
//!        -> jitter buffer -> Opus decode -> playout PCM
//! ```
//!
//! A 440 Hz sine wave stands in for the microphone signal. The encoded
//! packets are looped straight back into the jitter buffer as if they had
//! just arrived over the wire, then popped and decoded again.
//!
//! Run with:
//! ```sh
//! cargo run -p llm-rtc-sdk --example voice_pipeline
//! ```

// NOTE: `anyhow` is not a dependency of this crate, so we use the std
// error type instead; the example logic is identical.
use llm_rtc_core::audio::codec::CodecConfig;
use llm_rtc_core::audio::jitter::{AudioPacket, JitterBufferConfig};
use llm_rtc_core::audio::pipeline::{AudioPipeline, AudioPipelineConfig};
use llm_rtc_core::audio::processor::ProcessorConfig;
use llm_rtc_sdk::session::{SessionConfig, VoiceLlmSession};

const SAMPLE_RATE: u32 = 48_000;
/// 440 Hz concert-A tone.
const TONE_HZ: f32 = 440.0;
/// Total duration of the generated tone.
const DURATION_MS: u32 = 300;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("llm-rtc voice pipeline round-trip demo");
    println!("======================================");

    // The session type is what a real voice LLM app would hold; here we
    // create one briefly to show the wiring, then run the raw pipeline
    // loopback below (no network peer is needed for that).
    let _session = VoiceLlmSession::new(SessionConfig::default()).await?;
    println!("VoiceLlmSession created (default config)");

    // Build the full-duplex audio pipeline with default settings:
    // 48 kHz Opus, 10 ms frames, adaptive jitter buffer, AEC/NS/AGC.
    let codec = CodecConfig::default();
    let frame_ms = codec.frame_size_ms as u64;
    let samples_per_frame = (SAMPLE_RATE as f32 * codec.frame_size_ms / 1_000.0) as u32;
    let config = AudioPipelineConfig {
        codec,
        jitter: JitterBufferConfig::default(),
        processor: ProcessorConfig::default(),
    };
    let mut pipeline = AudioPipeline::new(config)?;
    println!("AudioPipeline initialized");

    // Generate a 440 Hz sine wave at 50% full scale (300 ms, mono).
    let input_samples = (SAMPLE_RATE as u64 * DURATION_MS as u64 / 1000) as usize;
    let mut pcm: Vec<i16> = (0..input_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (TONE_HZ * std::f32::consts::TAU * t).sin() * (i16::MAX as f32 / 2.0)
        })
        .map(|s| s as i16)
        .collect();
    println!(
        "Generated {} Hz sine wave: {} samples ({} ms)",
        TONE_HZ, input_samples, DURATION_MS
    );

    // Outgoing path: process the mic audio (AEC/NS/AGC) and encode it into
    // Opus packets, exactly what would hit the network.
    let packets = pipeline.process_outgoing(&mut pcm)?;
    println!("Encoded into {} Opus packet(s)", packets.len());

    // Simulate the network and the receiving side in a streaming fashion:
    // each encoded payload is wrapped in an AudioPacket (sequence number +
    // RTP timestamp) and pushed into the jitter buffer as it "arrives". A
    // The media clock independently asks for each frame at its RTP deadline,
    // matching the scheduling used by VoiceLlmSession.
    let mut decoded_samples = 0usize;
    let mut decoded_frames = 0usize;
    for (seq, payload) in packets.iter().enumerate() {
        let packet = AudioPacket {
            sequence_number: seq as u16,
            timestamp: (seq as u32) * samples_per_frame,
            payload: payload.clone(),
        };
        pipeline.push_incoming(packet);

        tokio::time::sleep(std::time::Duration::from_millis(frame_ms)).await;
        if let Some(frame) = pipeline.pop_decoded()? {
            decoded_samples += frame.len();
            decoded_frames += 1;
        }
    }

    // The initial target depth leaves a small tail after the final arrival.
    // Continue the media clock until every buffered RTP frame is played.
    while pipeline.has_pending_playout() {
        tokio::time::sleep(std::time::Duration::from_millis(frame_ms)).await;
        if let Some(frame) = pipeline.pop_decoded()? {
            decoded_samples += frame.len();
            decoded_frames += 1;
        }
    }

    // Report the round-trip summary.
    let jitter = pipeline.jitter_stats();
    let proc = pipeline.processor_stats();

    println!();
    println!("Round-trip summary");
    println!("------------------");
    println!("input samples:        {}", input_samples);
    println!("outgoing packets:     {}", packets.len());
    println!("decoded frames:       {}", decoded_frames);
    println!("decoded samples:      {}", decoded_samples);
    println!();
    println!("jitter stats:");
    println!("  packets in:         {}", jitter.packets_in);
    println!("  packets out:        {}", jitter.packets_out);
    println!("  packets dropped:    {}", jitter.packets_dropped);
    println!("  packets late:       {}", jitter.packets_late);
    println!("  jitter estimate:    {:.2} ms", jitter.current_jitter_ms);
    println!();
    println!("processor stats:");
    println!("  has voice:          {}", proc.has_voice);
    println!("  has echo:           {}", proc.has_echo);
    println!("  rms:                {:.2} dBFS", proc.rms_dbfs);
    println!("  speech probability: {:.3}", proc.speech_probability);

    Ok(())
}
