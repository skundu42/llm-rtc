//! Full-duplex audio pipeline composing codec, jitter buffer, and processor.
//!
//! The pipeline wires the individual audio building blocks into the two data
//! paths a voice LLM conversation needs:
//!
//! * **Outgoing (near end)** — microphone PCM flows through the
//!   [`AudioProcessor`] (AEC/NS/AGC/VAD) and is then encoded with Opus into
//!   network-ready packets: `mic -> process -> encode -> network`.
//! * **Incoming (far end)** — received Opus packets are reordered and paced
//!   by the [`JitterBuffer`], then decoded back to PCM for playout:
//!   `network -> jitter -> decode -> playout`.
//!
//! The AEC needs to know what is being played out the speakers, so rendered
//! audio must be fed back via [`AudioPipeline::on_render`] (or supplied
//! directly as an echo reference when processing outgoing frames).

use thiserror::Error;
use tracing::debug;

use crate::audio::codec::{CodecConfig, OpusDecoder, OpusEncoder};
use crate::audio::jitter::{
    AudioPacket, JitterBuffer, JitterBufferConfig, JitterStats, PlayoutEvent,
};
use crate::audio::processor::{
    AudioProcessor, ProcessorConfig, ProcessorStats, FRAME_SAMPLES as PROCESSOR_FRAME_SAMPLES,
};

/// Errors produced by the audio pipeline.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// The codec failed to initialize, encode, or decode.
    #[error("codec error: {0}")]
    Codec(#[from] crate::audio::codec::CodecError),

    /// The audio processor failed to initialize or process a frame.
    #[error("audio processor error: {0}")]
    Processor(#[from] crate::audio::processor::ProcessorError),
}

/// Convenience alias for pipeline results.
pub type Result<T> = std::result::Result<T, PipelineError>;

/// Configuration for the full-duplex audio pipeline.
///
/// Groups the sub-configurations of every component the pipeline owns so a
/// caller can construct the whole graph from one struct.
#[derive(Debug, Clone, Default)]
pub struct AudioPipelineConfig {
    /// Opus encoder/decoder settings (sample rate, bitrate, DTX, FEC, ...).
    pub codec: CodecConfig,
    /// Jitter buffer depth and pacing settings.
    pub jitter: JitterBufferConfig,
    /// AEC/NS/AGC/VAD settings for the microphone path.
    pub processor: ProcessorConfig,
}

/// Full-duplex audio pipeline for one WebRTC audio session.
///
/// Owns the encoder, decoder, jitter buffer, and audio processor, and exposes
/// the two directional paths (outgoing mic audio, incoming playout audio) as
/// simple, non-blocking, frame-oriented methods.
pub struct AudioPipeline {
    /// Opus encoder for the outgoing microphone path.
    encoder: OpusEncoder,
    /// Opus decoder for the incoming playout path.
    decoder: OpusDecoder,
    /// Reorders and paces incoming network packets before decoding.
    jitter: JitterBuffer,
    /// AEC/NS/AGC/VAD applied to the outgoing microphone path.
    processor: AudioProcessor,
    /// Incomplete microphone input awaiting one processor/codec-aligned block.
    pending_outgoing: Vec<i16>,
    /// Smallest block divisible by both the processor and codec frame sizes.
    outgoing_block_samples: usize,
}

fn least_common_multiple(left: usize, right: usize) -> usize {
    let (mut a, mut b) = (left, right);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    left / a * right
}

impl AudioPipeline {
    /// Build a pipeline from a configuration.
    ///
    /// Constructing the codec components validates the codec configuration,
    /// so invalid settings (bad sample rates, unsupported frame durations)
    /// surface here rather than on the first audio frame.
    pub fn new(config: AudioPipelineConfig) -> Result<Self> {
        let encoder = OpusEncoder::new(config.codec.clone())?;
        let decoder = OpusDecoder::new(config.codec.clone())?;
        let jitter = JitterBuffer::new(config.jitter);
        let processor = AudioProcessor::new(config.processor)?;
        let outgoing_block_samples =
            least_common_multiple(encoder.samples_per_frame(), PROCESSOR_FRAME_SAMPLES);

        debug!("audio pipeline initialized");

        Ok(Self {
            encoder,
            decoder,
            jitter,
            processor,
            pending_outgoing: Vec::new(),
            outgoing_block_samples,
        })
    }

    /// Process one outgoing microphone frame and encode it.
    ///
    /// The PCM is first run through the audio processor (noise suppression,
    /// AGC, and AEC driven by the previously fed render signal), then encoded
    /// into Opus packets. Incomplete chunks are retained until a full aligned
    /// processing block is available. Returns the packets ready for the network.
    pub fn process_outgoing(&mut self, mic_pcm: &mut [i16]) -> Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();
        self.process_outgoing_into(mic_pcm, &mut packets)?;
        Ok(packets)
    }

    /// Process outgoing audio into caller-owned packet storage.
    ///
    /// This is equivalent to [`AudioPipeline::process_outgoing`], but reuses
    /// the outer `Vec` allocation across real-time capture callbacks.
    pub fn process_outgoing_into(
        &mut self,
        mic_pcm: &mut [i16],
        packets: &mut Vec<Vec<u8>>,
    ) -> Result<()> {
        packets.clear();
        if self.pending_outgoing.is_empty()
            && mic_pcm.len().is_multiple_of(self.outgoing_block_samples)
        {
            self.processor.process(mic_pcm)?;
            self.encoder.encode_frames_into(mic_pcm, packets)?;
            return Ok(());
        }

        self.pending_outgoing.extend_from_slice(mic_pcm);
        let complete_samples =
            self.pending_outgoing.len() / self.outgoing_block_samples * self.outgoing_block_samples;
        if complete_samples == 0 {
            return Ok(());
        }

        let mut complete = if complete_samples == self.pending_outgoing.len() {
            std::mem::take(&mut self.pending_outgoing)
        } else {
            self.pending_outgoing.drain(..complete_samples).collect()
        };
        self.processor.process(&mut complete)?;
        self.encoder.encode_frames_into(&complete, packets)?;
        if self.pending_outgoing.is_empty() {
            complete.clear();
            self.pending_outgoing = complete;
        }
        Ok(())
    }

    /// Process one outgoing microphone frame with an explicit echo reference.
    ///
    /// Same as [`AudioPipeline::process_outgoing`], but the far-end playout
    /// signal is supplied directly as the AEC reference instead of relying on
    /// audio previously queued via [`AudioPipeline::on_render`].
    pub fn process_outgoing_with_reference(
        &mut self,
        mic_pcm: &mut [i16],
        far_end: &[i16],
    ) -> Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();
        self.process_outgoing_with_reference_into(mic_pcm, far_end, &mut packets)?;
        Ok(packets)
    }

    /// Process referenced outgoing audio into caller-owned packet storage.
    pub fn process_outgoing_with_reference_into(
        &mut self,
        mic_pcm: &mut [i16],
        far_end: &[i16],
        packets: &mut Vec<Vec<u8>>,
    ) -> Result<()> {
        self.processor.process_render(far_end)?;
        self.process_outgoing_into(mic_pcm, packets)
    }

    /// Feed one playout frame to the AEC reference.
    ///
    /// The audio being played out the speakers must be known to the echo
    /// canceller so it can subtract it from the microphone signal.
    pub fn on_render(&mut self, playout_pcm: &mut [i16]) -> Result<()> {
        self.processor.process_render(playout_pcm)?;
        Ok(())
    }

    /// Push a received network packet into the jitter buffer.
    ///
    /// Returns `true` if the packet was accepted, `false` if it was dropped
    /// (e.g. duplicate, too late, or the buffer is overflowing).
    pub fn push_incoming(&mut self, packet: AudioPacket) -> bool {
        self.jitter.push(packet)
    }

    /// Pop and decode the next in-order packet, if one is ready.
    ///
    /// Returns `None` when the jitter buffer has nothing to play out at the
    /// current RTP deadline. Missing slots return concealed PCM via PLC or FEC.
    pub fn pop_decoded(&mut self) -> Result<Option<Vec<i16>>> {
        let Some(event) = self.jitter.pop_event() else {
            return Ok(None);
        };

        let pcm = match event {
            PlayoutEvent::Packet(packet) => self.decoder.decode(&packet.payload)?,
            PlayoutEvent::Missing { .. } => self.decoder.decode(&[])?,
            PlayoutEvent::RecoveredWithNextPacket { next_packet, .. }
                if self.decoder.config().use_fec =>
            {
                self.decoder.decode_fec(&next_packet.payload)?
            }
            PlayoutEvent::RecoveredWithNextPacket { .. } => self.decoder.decode(&[])?,
        };
        Ok(Some(pcm))
    }

    /// Whether received packets are still waiting for playout.
    pub fn has_pending_playout(&self) -> bool {
        self.jitter.has_pending()
    }

    /// Wall-clock deadline of the next buffered playout slot.
    pub fn next_playout_deadline(&self) -> Option<std::time::Instant> {
        self.jitter.next_deadline()
    }

    /// Snapshot of the jitter buffer statistics.
    pub fn jitter_stats(&self) -> JitterStats {
        self.jitter.stats()
    }

    /// Snapshot of the audio processor statistics.
    pub fn processor_stats(&self) -> ProcessorStats {
        self.processor.stats()
    }

    /// Reset the pipeline state.
    ///
    /// Drops all buffered network audio and resets the audio processor's
    /// adaptive filters (useful when switching devices or reconnecting).
    pub fn reset(&mut self) {
        self.jitter.clear();
        self.pending_outgoing.clear();
        self.processor.reset();
        debug!("audio pipeline reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Samples per 10 ms frame at 48 kHz mono.
    const FRAME_SAMPLES: usize = 480;

    /// Generate one frame of a 440 Hz sine wave at ~50% full scale.
    fn sine_frame(sample_rate: u32, offset: &mut usize) -> Vec<i16> {
        (0..FRAME_SAMPLES)
            .map(|i| {
                let t = ((*offset + i) as f32) / sample_rate as f32;
                (440.0 * std::f32::consts::TAU * t).sin() * (i16::MAX as f32 / 2.0)
            })
            .map(|s| s as i16)
            .collect()
    }

    fn pipeline_with_clock(
        config: AudioPipelineConfig,
        now: Box<dyn Fn() -> Instant + Send + Sync>,
    ) -> AudioPipeline {
        let encoder = OpusEncoder::new(config.codec.clone()).unwrap();
        let outgoing_block_samples =
            least_common_multiple(encoder.samples_per_frame(), PROCESSOR_FRAME_SAMPLES);
        AudioPipeline {
            encoder,
            decoder: OpusDecoder::new(config.codec.clone()).unwrap(),
            jitter: JitterBuffer::with_clock(config.jitter, now),
            processor: AudioProcessor::new(config.processor).unwrap(),
            pending_outgoing: Vec::new(),
            outgoing_block_samples,
        }
    }

    #[test]
    fn new_succeeds_with_default_config() {
        let pipeline = AudioPipeline::new(AudioPipelineConfig::default());
        assert!(pipeline.is_ok());
    }

    #[test]
    fn process_outgoing_encodes_sine_wave() {
        let mut pipeline = AudioPipeline::new(AudioPipelineConfig::default()).unwrap();
        let mut frame = sine_frame(48_000, &mut 0);
        let packets = pipeline.process_outgoing(&mut frame).unwrap();

        assert!(!packets.is_empty(), "expected at least one packet");
        assert!(packets.iter().all(|p| !p.is_empty()));
    }

    #[test]
    fn process_outgoing_into_reuses_packet_storage() {
        let mut pipeline = AudioPipeline::new(AudioPipelineConfig::default()).unwrap();
        let mut frame = sine_frame(48_000, &mut 0);
        let mut packets = Vec::new();

        pipeline
            .process_outgoing_into(&mut frame, &mut packets)
            .unwrap();
        let capacity = packets.capacity();
        pipeline
            .process_outgoing_into(&mut frame, &mut packets)
            .unwrap();

        assert_eq!(packets.len(), 1);
        assert_eq!(packets.capacity(), capacity);
    }

    #[test]
    fn partial_outgoing_chunks_wait_for_a_complete_codec_block() {
        const CHUNK_SAMPLES: usize = 240;

        for frame_ms in [10.0, 20.0] {
            let config = AudioPipelineConfig {
                codec: CodecConfig {
                    frame_size_ms: frame_ms,
                    ..CodecConfig::default()
                },
                jitter: JitterBufferConfig {
                    frame_size_ms: frame_ms as u32,
                    ..JitterBufferConfig::default()
                },
                ..AudioPipelineConfig::default()
            };
            let expected_samples = (48_000.0 * frame_ms / 1_000.0) as usize;
            let chunks = expected_samples / CHUNK_SAMPLES;
            let mut pipeline = AudioPipeline::new(config.clone()).unwrap();
            let mut packets = Vec::new();

            for index in 0..chunks {
                let mut chunk = vec![1_000; CHUNK_SAMPLES];
                let produced = pipeline.process_outgoing(&mut chunk).unwrap();
                if index + 1 < chunks {
                    assert!(produced.is_empty());
                } else {
                    packets = produced;
                }
            }

            assert_eq!(packets.len(), 1);
            let mut decoder = OpusDecoder::new(config.codec).unwrap();
            assert_eq!(decoder.decode(&packets[0]).unwrap().len(), expected_samples);
        }

        assert_eq!(least_common_multiple(720, PROCESSOR_FRAME_SAMPLES), 1_440);

        let mut pipeline = AudioPipeline::new(AudioPipelineConfig::default()).unwrap();
        let mut half_frame = vec![1_000; CHUNK_SAMPLES];
        assert!(pipeline
            .process_outgoing(&mut half_frame)
            .unwrap()
            .is_empty());
        pipeline.reset();
        assert!(pipeline
            .process_outgoing(&mut half_frame)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn push_incoming_then_pop_decoded_round_trips() {
        let config = AudioPipelineConfig {
            jitter: JitterBufferConfig {
                target_latency_ms: 0,
                ..JitterBufferConfig::default()
            },
            ..AudioPipelineConfig::default()
        };
        let mut pipeline = AudioPipeline::new(config.clone()).unwrap();

        // Encode a sine wave with a standalone encoder using the same config.
        let encoder = OpusEncoder::new(config.codec.clone()).unwrap();
        let pcm = sine_frame(48_000, &mut 0);
        let payload = encoder.encode(&pcm).unwrap();

        let packet = AudioPacket {
            sequence_number: 0,
            timestamp: 0,
            payload,
        };
        assert!(pipeline.push_incoming(packet));

        let decoded = pipeline.pop_decoded().unwrap().expect("packet expected");
        assert_eq!(decoded.len(), FRAME_SAMPLES);
    }

    #[test]
    fn stats_are_accessible() {
        let mut pipeline = AudioPipeline::new(AudioPipelineConfig::default()).unwrap();
        let _jitter = pipeline.jitter_stats();
        let _proc = pipeline.processor_stats();

        // Reset should also be callable without panicking.
        pipeline.reset();
    }

    #[test]
    fn consecutive_losses_use_plc_then_fec_and_preserve_next_packet() {
        let config = AudioPipelineConfig {
            jitter: JitterBufferConfig {
                target_latency_ms: 0,
                max_latency_ms: 200,
                ..JitterBufferConfig::default()
            },
            ..AudioPipelineConfig::default()
        };
        let encoder = OpusEncoder::new(config.codec.clone()).unwrap();
        let mut offset = 0;
        let mut payloads = Vec::new();
        for _ in 0..4 {
            let frame = sine_frame(48_000, &mut offset);
            offset += FRAME_SAMPLES;
            payloads.push(encoder.encode(&frame).unwrap());
        }

        let base = Instant::now();
        let elapsed = Arc::new(Mutex::new(Duration::ZERO));
        let clock_elapsed = Arc::clone(&elapsed);
        let mut pipeline = pipeline_with_clock(
            config,
            Box::new(move || base + *clock_elapsed.lock().unwrap()),
        );

        // Packets 1 and 2 are lost. Slot 1 has no adjacent future packet and
        // uses PLC; slot 2 can recover from packet 3's in-band FEC data.
        for seq in [0u16, 3] {
            assert!(pipeline.push_incoming(AudioPacket {
                sequence_number: seq,
                timestamp: u32::from(seq) * FRAME_SAMPLES as u32,
                payload: payloads[usize::from(seq)].clone(),
            }));
        }

        for expected_slot in 0..4 {
            let decoded = pipeline
                .pop_decoded()
                .unwrap()
                .unwrap_or_else(|| panic!("slot {expected_slot} should be due"));
            assert_eq!(decoded.len(), FRAME_SAMPLES);
            *elapsed.lock().unwrap() += Duration::from_millis(10);
        }

        let stats = pipeline.jitter_stats();
        assert_eq!(stats.packets_dropped, 2);
        assert_eq!(stats.packets_out, 2);
        assert!(!pipeline.has_pending_playout());
    }

    /// The deterministic severe benchmark trace must still produce one
    /// decoded frame per RTP slot. Payload loss may select FEC/PLC, but buffer
    /// pressure must never silently shorten the stream.
    #[test]
    fn severe_jitter_trace_preserves_decoded_frame_continuity() {
        const CONTENT_FRAMES: usize = 1_000;
        const GUARD_FRAMES: usize = 10;

        let config = AudioPipelineConfig {
            jitter: JitterBufferConfig {
                target_latency_ms: 5,
                max_latency_ms: 115,
                ..JitterBufferConfig::default()
            },
            ..AudioPipelineConfig::default()
        };
        let encoder = OpusEncoder::new(config.codec.clone()).unwrap();
        let base = Instant::now();
        let elapsed = Arc::new(Mutex::new(Duration::ZERO));
        let clock_elapsed = Arc::clone(&elapsed);
        let mut pipeline = pipeline_with_clock(
            config,
            Box::new(move || base + *clock_elapsed.lock().unwrap()),
        );

        let mut random_state = 0xa5a5_5a5a_u32;
        let mut random = || {
            random_state ^= random_state << 13;
            random_state ^= random_state >> 17;
            random_state ^= random_state << 5;
            f64::from(random_state) / f64::from(u32::MAX)
        };
        let mut offset = 0;
        let mut arrivals = Vec::new();
        for frame_index in 0..(CONTENT_FRAMES + GUARD_FRAMES) {
            let is_guard = frame_index >= CONTENT_FRAMES;
            let jitter_ms = if is_guard {
                0.0
            } else {
                (random() * 2.0 - 1.0) * 40.0
            };
            let dropped = !is_guard && frame_index != 0 && random() < 0.10;
            let frame = if is_guard {
                vec![0; FRAME_SAMPLES]
            } else {
                let frame = sine_frame(48_000, &mut offset);
                offset += FRAME_SAMPLES;
                frame
            };
            let payload = encoder.encode(&frame).unwrap();
            if !dropped {
                arrivals.push((
                    frame_index as f64 * 10.0 + 50.0 + jitter_ms,
                    AudioPacket {
                        sequence_number: 10_000_u16.wrapping_add(frame_index as u16),
                        timestamp: 900_000_u32.wrapping_add((frame_index * FRAME_SAMPLES) as u32),
                        payload,
                    },
                ));
            }
        }
        arrivals.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());

        let end_ms = arrivals.last().unwrap().0 + 2_000.0;
        let mut next_arrival = 0;
        let mut decoded_frames = 0;
        let mut tick_ms = 0.0;
        while tick_ms <= end_ms && decoded_frames < CONTENT_FRAMES {
            while next_arrival < arrivals.len() && arrivals[next_arrival].0 <= tick_ms {
                let (arrival_ms, packet) = &arrivals[next_arrival];
                *elapsed.lock().unwrap() = Duration::from_secs_f64(*arrival_ms / 1_000.0);
                pipeline.push_incoming(packet.clone());
                next_arrival += 1;
            }
            *elapsed.lock().unwrap() = Duration::from_secs_f64(tick_ms / 1_000.0);
            if pipeline.pop_decoded().unwrap().is_some() {
                decoded_frames += 1;
            }
            // Model the SDK's 2x catch-up ceiling after an empty-buffer loss.
            tick_ms += 5.0;
        }

        assert_eq!(
            decoded_frames, CONTENT_FRAMES,
            "severe trace silently lost a decoded playout slot"
        );
        assert!(pipeline.jitter_stats().packets_late < CONTENT_FRAMES as u64 / 10);
        assert!(
            tick_ms <= 10_100.0,
            "severe trace accumulated excessive turn-ingress delay: {tick_ms} ms"
        );
    }
}
