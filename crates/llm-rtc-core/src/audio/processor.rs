//! WebRTC audio processing for the microphone path (voice LLM pipeline).
//!
//! Clean microphone input is the single cheapest way to improve ASR/LLM
//! accuracy: echo from the speaker leaks words into the transcript, background
//! noise mangles phonemes, and a low recording level wastes bits on quantization
//! noise. This module wraps the `webrtc-audio-processing` crate (the same
//! engine Chrome uses) to apply, in order:
//!
//! * **AEC** — acoustic echo cancellation, using the far-end playback signal
//!   as a reference so the user's own speech is not cancelled.
//! * **NS** — noise suppression, with a tunable suppression level.
//! * **AGC** — automatic gain control to a target level (positive dBFs
//!   convention: `3` means −3 dBFs).
//! * **VAD** — voice activity detection, exposed via [`AudioProcessor::stats`].
//!
//! # Fixed engine parameters
//!
//! The underlying engine is **fixed** at 48 kHz and 10 ms frames (exactly
//! [`NUM_SAMPLES_PER_FRAME`] = 480 samples per frame). It cannot be
//! reconfigured. Because the rest of llm-rtc speaks 16-bit PCM and callers
//! naturally produce arbitrary chunk sizes, this wrapper maintains an
//! internal `i16` accumulator: input samples are buffered, converted to
//! `f32` (scaled by 1/32768), and complete 480-sample frames are handed to
//! the engine. Processed samples are converted back to `i16` and written
//! **in place** into the caller's buffer.
//!
//! # Streaming / partial frames
//!
//! `process` and `process_with_reference` are streaming-safe: if fewer than
//! 480 samples are available after buffering, nothing is fed to the engine
//! and the samples are simply retained until a later call completes the
//! frame. Output delivery is also streaming-safe: if the engine produced more
//! processed samples than fit in the current input buffer, the remainder is
//! carried over and delivered at the front of the next call. The processed
//! output stream is a 1:1, in-order transform of the input stream, so this
//! carry-over preserves sample alignment across calls.

use thiserror::Error;
use tracing::debug;

use webrtc_audio_processing as wap;

/// Samples per engine frame: 10 ms @ 48 kHz, mono (interleaved, so 480 total).
const FRAME_SAMPLES: usize = wap::NUM_SAMPLES_PER_FRAME as usize;

/// Fixed capture rate of the underlying engine (Hz).
pub const SAMPLE_RATE_HZ: u32 = 48_000;

/// Errors produced by the audio processor.
#[derive(Debug, Error)]
pub enum ProcessorError {
    /// The engine rejected the initialization or runtime configuration.
    #[error("audio processor configuration failed: {0}")]
    Config(#[from] wap::Error),

    /// The engine rejected a capture or render frame.
    #[error("audio processing failed on frame: {0}")]
    Process(#[source] wap::Error),
}

/// Convenience alias for processor results.
pub type Result<T> = std::result::Result<T, ProcessorError>;

/// Tuning knobs for the WebRTC audio processing pipeline.
///
/// Defaults enable everything (AEC, NS, AGC, VAD) with settings chosen for
/// interactive voice conversations with an LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    /// Acoustic echo cancellation (needs a far-end reference via
    /// [`AudioProcessor::process_with_reference`] or `process_render` to work).
    pub enable_aec: bool,
    /// Stationary + transient noise suppression.
    pub enable_ns: bool,
    /// Automatic gain control to a fixed target level.
    pub enable_agc: bool,
    /// Voice activity detection (surfaced through `stats()`).
    pub enable_vad: bool,
    /// AGC target level in dBFs. Note the engine's **positive** convention:
    /// `3` means −3 dBFs (i.e. peak speech at ~71% of full scale).
    pub agc_target_level_dbfs: i32,
    /// Maximum gain the AGC will apply during compression, in dB.
    pub agc_compression_gain_db: i32,
    /// Noise suppression strength.
    pub ns_level: wap::NoiseSuppressionLevel,
    /// AEC suppression strength.
    pub aec_suppression: wap::EchoCancellationSuppressionLevel,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            enable_aec: true,
            enable_ns: true,
            enable_agc: true,
            enable_vad: true,
            agc_target_level_dbfs: 3,
            agc_compression_gain_db: 9,
            ns_level: wap::NoiseSuppressionLevel::Moderate,
            aec_suppression: wap::EchoCancellationSuppressionLevel::Moderate,
        }
    }
}

impl ProcessorConfig {
    /// Translate this config into the engine's runtime `Config`, enabling
    /// only the requested sub-modules.
    fn to_engine_config(&self) -> wap::Config {
        wap::Config {
            echo_cancellation: self.enable_aec.then_some(wap::EchoCancellation {
                suppression_level: self.aec_suppression,
                enable_extended_filter: true,
                enable_delay_agnostic: true,
                stream_delay_ms: None,
            }),
            gain_control: self.enable_agc.then_some(wap::GainControl {
                mode: wap::GainControlMode::AdaptiveDigital,
                target_level_dbfs: self.agc_target_level_dbfs,
                compression_gain_db: self.agc_compression_gain_db,
                enable_limiter: true,
            }),
            noise_suppression: self.enable_ns.then_some(wap::NoiseSuppression {
                suppression_level: self.ns_level,
            }),
            voice_detection: self.enable_vad.then_some(wap::VoiceDetection {
                detection_likelihood: wap::VoiceDetectionLikelihood::Low,
            }),
            enable_transient_suppressor: false,
            enable_high_pass_filter: true,
        }
    }
}

/// Snapshot of engine state, useful for gating barge-in and logging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessorStats {
    /// Voice activity detected in the most recently processed capture frame.
    pub has_voice: bool,
    /// Echo (far-end leakage) detected in the capture path.
    pub has_echo: bool,
    /// Short-term RMS level of the capture signal, in dBFS.
    pub rms_dbfs: f64,
    /// Engine speech probability estimate in `[0, 1]`.
    pub speech_probability: f32,
}

/// Wraps the WebRTC audio processing engine with `i16` buffering.
///
/// The engine itself is fixed at 48 kHz / 10 ms / 480-sample frames; this
/// struct adapts arbitrary-size `i16` chunks onto that frame grid. It is not
/// `Sync`; keep one instance per capture stream (typically inside the audio
/// task that owns the microphone).
pub struct AudioProcessor {
    /// The underlying engine. Cloning it would share internal state, so we
    /// own exactly one and recreate it on `reset()`.
    processor: wap::Processor,
    /// Initialization parameters, retained so `reset()` can rebuild the engine.
    init: wap::InitializationConfig,
    /// Runtime configuration, retained so `reset()` can re-apply it.
    config: ProcessorConfig,
    /// Raw near-end (microphone) samples awaiting a complete 480-sample frame.
    pending_capture: Vec<i16>,
    /// Far-end (playback) samples awaiting a complete 480-sample frame.
    pending_render: Vec<i16>,
    /// Processed output samples that did not fit in the caller's buffer on
    /// the previous call; delivered at the front of the next call.
    carry_out: Vec<i16>,
}

// Manual impl: the engine handle does not implement `Debug`.
impl std::fmt::Debug for AudioProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioProcessor")
            .field("config", &self.config)
            .field("pending_capture", &self.pending_capture.len())
            .field("pending_render", &self.pending_render.len())
            .field("carry_out", &self.carry_out.len())
            .finish()
    }
}

impl AudioProcessor {
    /// Create a processor with the given configuration.
    ///
    /// The engine is initialized mono-in / mono-out at its fixed 48 kHz rate.
    /// The runtime [`wap::Config`] is applied immediately via `set_config` so
    /// the very first processed frame already has AEC/NS/AGC/VAD active.
    pub fn new(config: ProcessorConfig) -> Result<Self> {
        let init = wap::InitializationConfig {
            num_capture_channels: 1,
            num_render_channels: 1,
            enable_experimental_agc: config.enable_agc,
            enable_intelligibility_enhancer: false,
        };

        let mut processor = wap::Processor::new(&init)?;
        processor.set_config(config.to_engine_config());

        debug!(
            aec = config.enable_aec,
            ns = config.enable_ns,
            agc = config.enable_agc,
            vad = config.enable_vad,
            "audio processor initialized (48 kHz, {} samples/frame)",
            FRAME_SAMPLES
        );

        Ok(Self {
            processor,
            init,
            config,
            pending_capture: Vec::new(),
            pending_render: Vec::new(),
            carry_out: Vec::new(),
        })
    }

    /// Process a near-end (microphone) chunk.
    ///
    /// Samples are buffered and fed to the engine in complete 480-sample
    /// frames; processed samples are written back into `near_end` in place.
    /// If the buffer holds fewer than 480 samples after this call, nothing is
    /// processed and the partial data is retained for the next call — so
    /// callers may pass any chunk size (e.g. 160-sample Opus frames).
    pub fn process(&mut self, near_end: &mut [i16]) -> Result<()> {
        self.pending_capture.extend_from_slice(near_end);
        let produced = self.drain_capture_frames()?;
        self.deliver(produced, near_end);
        Ok(())
    }

    /// Process a near-end chunk together with the far-end (playback)
    /// reference required for echo cancellation.
    ///
    /// `far_end` should contain the samples the speaker played while
    /// `near_end` was captured (same length is typical, but any size is
    /// buffered the same way). The far-end reference is consumed before the
    /// capture frames are processed so the AEC aligns against the most recent
    /// playback.
    pub fn process_with_reference(
        &mut self,
        near_end: &mut [i16],
        far_end: &[i16],
    ) -> Result<()> {
        self.pending_render.extend_from_slice(far_end);
        while self.pending_render.len() >= FRAME_SAMPLES {
            let mut frame = [0.0f32; FRAME_SAMPLES];
            Self::fill_frame(&mut frame, &mut self.pending_render);
            self.processor
                .process_render_frame(&mut frame)
                .map_err(ProcessorError::Process)?;
        }

        self.process(near_end)
    }

    /// Feed a playback chunk to the AEC reference path without processing any
    /// capture audio.
    ///
    /// Use this when playback and capture arrive on separate paths (e.g. the
    /// render callback runs on a different task than the microphone reader);
    /// partial frames are buffered until 480 samples accumulate.
    pub fn process_render(&mut self, far_end: &mut [i16]) -> Result<()> {
        self.pending_render.extend_from_slice(far_end);
        while self.pending_render.len() >= FRAME_SAMPLES {
            let mut frame = [0.0f32; FRAME_SAMPLES];
            Self::fill_frame(&mut frame, &mut self.pending_render);
            self.processor
                .process_render_frame(&mut frame)
                .map_err(ProcessorError::Process)?;
        }
        Ok(())
    }

    /// Drop and recreate the internal engine, clearing all buffered audio.
    ///
    /// Call this when the audio route changes (device switch, headset
    /// connect) or after a stream discontinuity, so stale AEC delay estimates
    /// and NS noise models do not corrupt the new stream.
    pub fn reset(&mut self) {
        self.pending_capture.clear();
        self.pending_render.clear();
        self.carry_out.clear();

        // Rebuild from the retained initialization parameters. On failure
        // (which should be impossible — the same config was accepted at
        // construction) keep the existing engine rather than lose the stream.
        match wap::Processor::new(&self.init) {
            Ok(mut processor) => {
                processor.set_config(self.config.to_engine_config());
                self.processor = processor;
                debug!("audio processor reset");
            }
            Err(e) => tracing::warn!("audio processor reset failed, keeping old engine: {e}"),
        }
    }

    /// Snapshot of the engine's most recent statistics.
    ///
    /// The engine reports `None` for a metric until it has enough data to
    /// judge (e.g. VAD needs a couple of frames); those fall back to neutral
    /// values rather than panicking in the audio hot path.
    pub fn stats(&self) -> ProcessorStats {
        let s = self.processor.get_stats();
        ProcessorStats {
            has_voice: s.has_voice.unwrap_or(false),
            has_echo: s.has_echo.unwrap_or(false),
            rms_dbfs: s
                .rms_dbfs
                .map(f64::from)
                .unwrap_or(f64::NEG_INFINITY),
            speech_probability: s.speech_probability.unwrap_or(0.0) as f32,
        }
    }

    /// Process every complete 480-sample capture frame currently buffered and
    /// return the processed samples as `i16`.
    fn drain_capture_frames(&mut self) -> Result<Vec<i16>> {
        let frames = self.pending_capture.len() / FRAME_SAMPLES;
        if frames == 0 {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(frames * FRAME_SAMPLES);
        for _ in 0..frames {
            let mut frame = [0.0f32; FRAME_SAMPLES];
            Self::fill_frame(&mut frame, &mut self.pending_capture);
            self.processor
                .process_capture_frame(&mut frame)
                .map_err(ProcessorError::Process)?;
            out.extend(frame.iter().map(|&s| f32_to_i16(s)));
        }
        Ok(out)
    }

    /// Move the front 480 buffered samples into `frame`, converting
    /// `i16 -> f32` (full scale = 1.0) and consuming them from the buffer.
    fn fill_frame(frame: &mut [f32; FRAME_SAMPLES], buffer: &mut Vec<i16>) {
        for (dst, src) in frame.iter_mut().zip(buffer.drain(..FRAME_SAMPLES)) {
            *dst = src as f32 / 32768.0;
        }
    }

    /// Deliver processed samples into the caller's buffer, carrying any
    /// surplus over to the next call.
    fn deliver(&mut self, mut produced: Vec<i16>, near_end: &mut [i16]) {
        if !self.carry_out.is_empty() {
            let mut carried = std::mem::take(&mut self.carry_out);
            carried.append(&mut produced);
            produced = carried;
        }

        let n = produced.len().min(near_end.len());
        near_end[..n].copy_from_slice(&produced[..n]);
        if produced.len() > n {
            self.carry_out = produced[n..].to_vec();
        }
    }
}

/// Convert an engine output sample back to `i16`, saturating on overflow.
fn f32_to_i16(s: f32) -> i16 {
    (s * 32768.0).clamp(-32768.0, 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f32::consts::PI;

    /// A 440 Hz sine at roughly -18 dBFs, as a stand-in for speech.
    fn sine(len: usize, amplitude: i16) -> Vec<i16> {
        (0..len)
            .map(|i| {
                (amplitude as f32 * (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE_HZ as f32).sin())
                    as i16
            })
            .collect()
    }

    /// (a) Config defaults and translation into engine config.
    #[test]
    fn config_defaults_and_translation() {
        let cfg = ProcessorConfig::default();
        assert!(cfg.enable_aec && cfg.enable_ns && cfg.enable_agc && cfg.enable_vad);
        assert_eq!(cfg.agc_target_level_dbfs, 3);
        assert_eq!(cfg.agc_compression_gain_db, 9);
        assert_eq!(cfg.ns_level, wap::NoiseSuppressionLevel::Moderate);
        assert_eq!(
            cfg.aec_suppression,
            wap::EchoCancellationSuppressionLevel::Moderate
        );

        let engine = cfg.to_engine_config();
        assert!(engine.echo_cancellation.is_some());
        assert!(engine.noise_suppression.is_some());
        assert!(engine.gain_control.is_some());
        assert!(engine.voice_detection.is_some());
        assert!(engine.enable_high_pass_filter);

        // Disabled modules must translate to `None`.
        let off = ProcessorConfig {
            enable_aec: false,
            enable_agc: false,
            ..Default::default()
        }
        .to_engine_config();
        assert!(off.echo_cancellation.is_none());
        assert!(off.gain_control.is_none());
        assert!(off.noise_suppression.is_some());
    }

    /// (b) A sine wave through NS/AGC comes out with the same length and
    /// non-silent content.
    #[test]
    fn sine_wave_processing_produces_full_length_output() {
        let mut proc = AudioProcessor::new(ProcessorConfig {
            enable_aec: false, // no render reference in this test
            ..Default::default()
        })
        .expect("processor construction");

        // Two full frames (960 samples) so processing definitely happens.
        let mut samples = sine(2 * FRAME_SAMPLES, 8_000);
        proc.process(&mut samples).expect("processing");

        assert_eq!(samples.len(), 2 * FRAME_SAMPLES);
        assert!(
            samples.iter().any(|&s| s != 0),
            "processed output must not be silent"
        );
    }

    /// (c) Processing with a far-end reference (AEC path) runs without error.
    #[test]
    fn process_with_reference_runs() {
        let mut proc = AudioProcessor::new(ProcessorConfig::default()).expect("processor");

        let mut near = sine(3 * FRAME_SAMPLES, 8_000);
        let far = sine(3 * FRAME_SAMPLES, 4_000);
        proc.process_with_reference(&mut near, &far)
            .expect("process_with_reference");
        assert_eq!(near.len(), 3 * FRAME_SAMPLES);
    }

    /// (d) `reset()` recreates the engine and processing continues to work.
    #[test]
    fn reset_rebuilds_engine() {
        let mut proc = AudioProcessor::new(ProcessorConfig::default()).expect("processor");

        let mut first = sine(FRAME_SAMPLES, 8_000);
        proc.process(&mut first).expect("pre-reset processing");

        proc.reset();

        let mut second = sine(FRAME_SAMPLES, 8_000);
        proc.process(&mut second).expect("post-reset processing");
        assert_eq!(second.len(), FRAME_SAMPLES);
    }

    /// (e) Partial frames are buffered: a short chunk produces no output,
    /// and the remainder is processed once the frame completes.
    #[test]
    fn partial_frames_are_buffered() {
        let mut proc = AudioProcessor::new(ProcessorConfig {
            enable_aec: false,
            ..Default::default()
        })
        .expect("processor");

        // 100 samples: less than one 480-sample frame, nothing to process.
        let mut short = sine(100, 8_000);
        proc.process(&mut short).expect("partial chunk");

        // Complete the frame: 100 + 380 = 480 samples, exactly one frame.
        let mut rest = sine(380, 8_000);
        proc.process(&mut rest).expect("completing chunk");

        // Call 1 delivered no output; call 2's buffer only fit 380 of the
        // 480 processed samples, so 100 remain carried over for next time.
        assert_eq!(proc.carry_out.len(), 100);
        // Buffer must be empty again — every buffered sample was consumed.
        assert_eq!(proc.pending_capture.len(), 0);
    }
}
