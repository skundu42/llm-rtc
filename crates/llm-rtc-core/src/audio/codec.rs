//! Opus audio codec tuned for low-latency voice (LLM speech) streaming.
//!
//! This module wraps the safe `opus` crate bindings around libopus with
//! settings chosen for a real-time conversational pipeline:
//!
//! * `Application::Voip` — optimizes intelligibility of speech over fidelity.
//! * Low bitrate (default 24 kbps) — small packets fit a single MTU and keep
//!   serialization/encryption overhead minimal.
//! * DTX (Discontinuous Transmission) — silence is not transmitted, saving
//!   bandwidth and CPU during pauses in conversation.
//! * In-band FEC — the encoder embeds a low-bandwidth copy of the previous
//!   frame so the decoder can reconstruct after a single lost packet without
//!   waiting for a retransmission.
//! * Complexity 0 — the fastest libopus operating point; encode latency and
//!   CPU burn matter far more than a fraction of a dB of quality here.
//!
//! All PCM in this module is 16-bit interleaved (mono by default). No
//! `unsafe` is used anywhere in this module; the `opus` crate owns the FFI
//! boundary.

use std::sync::Mutex;

use opus::{Application, Bitrate, Channels, Signal};
use thiserror::Error;

/// Largest possible Opus packet (per RFC 6716) at any bitrate/settings.
const MAX_PACKET_SIZE: usize = 1275;

/// Valid Opus frame durations in milliseconds.
///
/// libopus only accepts frames of exactly these lengths; anything else
/// returns `OPUS_BAD_ARG`. We validate eagerly at construction time so a bad
/// config surfaces immediately instead of on the first `encode` call in the
/// audio hot path.
const VALID_FRAME_DURATIONS_MS: [f32; 6] = [2.5, 5.0, 10.0, 20.0, 40.0, 60.0];

/// Sample rates (Hz) supported by libopus.
const VALID_SAMPLE_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

/// Errors produced by the codec layer.
#[derive(Debug, Error)]
pub enum CodecError {
    /// The [`CodecConfig`] is not usable by libopus (bad rate, duration, etc.).
    #[error("invalid codec configuration: {0}")]
    InvalidConfig(String),

    /// libopus rejected the frame or ran out of buffer space while encoding.
    #[error("opus encode failed: {0}")]
    Encode(#[source] opus::Error),

    /// libopus rejected the packet or the output buffer was too small.
    #[error("opus decode failed: {0}")]
    Decode(#[source] opus::Error),

    /// The input PCM length does not match the configured frame size.
    #[error(
        "pcm input is {actual} samples, expected exactly one frame of {expected} samples \
         ({channels} channel(s) interleaved)"
    )]
    FrameSizeMismatch {
        /// Number of samples actually supplied.
        actual: usize,
        /// Expected samples per frame (all channels interleaved).
        expected: usize,
        /// Configured channel count.
        channels: u8,
    },

    /// The internal encoder mutex was poisoned by a panic in another thread.
    #[error("encoder lock poisoned by a prior panic")]
    Poisoned,
}

/// Convenience alias for codec results.
pub type Result<T> = std::result::Result<T, CodecError>;

/// Tuning knobs for the Opus codec, optimized for low-latency voice.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecConfig {
    /// Sample rate in Hz. libopus supports 8/12/16/24/48 kHz.
    /// Defaults to 48 kHz, the WebRTC standard.
    pub sample_rate: u32,

    /// Channel count: 1 = mono (default, best for voice), 2 = stereo.
    pub channels: u8,

    /// Target bitrate in bits/second. 24 kbps gives clean speech at very
    /// small packet sizes; the Opus "sweet spot" for VoIP is 16–32 kbps.
    pub bitrate: u32,

    /// Frame duration in milliseconds. 20 ms is the classic WebRTC trade-off
    /// between packet overhead and delay; use 10 ms for lower latency.
    pub frame_size_ms: f32,

    /// Enable Discontinuous Transmission: don't send packets during silence
    /// (the decoder synthesizes comfort noise). Greatly reduces bandwidth
    /// when one side of a conversation is listening.
    pub use_dtx: bool,

    /// Enable in-band forward error correction: each packet carries a
    /// low-bitrate copy of the previous frame, letting the decoder recover
    /// from isolated packet losses without concealment artifacts.
    pub use_fec: bool,

    /// Encoder computational complexity, 0 (fastest) to 10 (best quality).
    /// Default 0 prioritizes minimal encode latency for real-time voice.
    pub complexity: u8,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 1,
            bitrate: 24_000,
            frame_size_ms: 20.0,
            use_dtx: true,
            use_fec: true,
            complexity: 0,
        }
    }
}

impl CodecConfig {
    /// Validate the configuration against libopus's constraints.
    ///
    /// Called by [`OpusEncoder::new`] and [`OpusDecoder::new`] so that invalid
    /// settings fail fast at startup rather than mid-stream.
    fn validate(&self) -> Result<()> {
        if !VALID_SAMPLE_RATES.contains(&self.sample_rate) {
            return Err(CodecError::InvalidConfig(format!(
                "sample_rate {} Hz is not supported (expected one of {VALID_SAMPLE_RATES:?})",
                self.sample_rate
            )));
        }

        if self.channels != 1 && self.channels != 2 {
            return Err(CodecError::InvalidConfig(format!(
                "channels must be 1 (mono) or 2 (stereo), got {}",
                self.channels
            )));
        }

        // libopus hard limits: 500 bps .. 512 kbps.
        if !(500..=512_000).contains(&self.bitrate) {
            return Err(CodecError::InvalidConfig(format!(
                "bitrate {} bps out of range [500, 512000]",
                self.bitrate
            )));
        }

        if !VALID_FRAME_DURATIONS_MS
            .iter()
            .any(|d| (d - self.frame_size_ms).abs() < f32::EPSILON)
        {
            return Err(CodecError::InvalidConfig(format!(
                "frame_size_ms {} is invalid (expected one of {VALID_FRAME_DURATIONS_MS:?})",
                self.frame_size_ms
            )));
        }

        if self.complexity > 10 {
            return Err(CodecError::InvalidConfig(format!(
                "complexity {} out of range [0, 10]",
                self.complexity
            )));
        }

        Ok(())
    }

    /// Number of PCM samples per frame, counting all channels interleaved.
    ///
    /// For 20 ms @ 48 kHz mono this is 960.
    fn samples_per_frame(&self) -> usize {
        (self.frame_size_ms * self.sample_rate as f32 / 1000.0) as usize * self.channels as usize
    }

    /// Map the channel count onto libopus's enum.
    fn opus_channels(&self) -> Channels {
        if self.channels == 2 {
            Channels::Stereo
        } else {
            Channels::Mono
        }
    }
}

/// Opus encoder for real-time voice.
///
/// Cheap to clone-safe use from multiple threads: the underlying libopus
/// encoder sits behind a mutex so [`OpusEncoder::encode`] only needs `&self`
/// (encoder state is mutated by libopus on every call).
#[derive(Debug)]
pub struct OpusEncoder {
    /// Validated configuration this encoder was built from.
    config: CodecConfig,
    /// PCM samples per frame (all channels interleaved).
    samples_per_frame: usize,
    /// The libopus encoder. `Mutex` gives interior mutability while keeping
    /// `encode(&self)` ergonomic for callers in the audio pipeline.
    encoder: Mutex<opus::Encoder>,
}

impl OpusEncoder {
    /// Create an encoder pre-tuned for low-latency voice.
    ///
    /// Applies (in order): VOIP application mode, target bitrate, minimal
    /// complexity, DTX, in-band FEC with a 10% assumed loss rate (libopus
    /// only spends bits on FEC when it expects loss), and voice signal
    /// detection hints.
    pub fn new(config: CodecConfig) -> Result<Self> {
        config.validate()?;

        let mut encoder = opus::Encoder::new(
            config.sample_rate,
            config.opus_channels(),
            // Voip mode: speech-optimized psychoacoustics, lowest complexity.
            Application::Voip,
        )
        .map_err(CodecError::Encode)?;

        // Target bitrate drives packet size directly; smaller packets mean
        // less time on the wire and in the jitter buffer.
        encoder
            .set_bitrate(Bitrate::Bits(config.bitrate as i32))
            .map_err(CodecError::Encode)?;

        // Complexity 0 = fewest CPU cycles / fastest encode. For clean speech
        // at VOIP mode the quality delta vs. higher settings is negligible.
        encoder
            .set_complexity(config.complexity as i32)
            .map_err(CodecError::Encode)?;

        // DTX suppresses packets during silence; the decoder fills in with
        // comfort noise. Essential for conversational turn-taking.
        encoder.set_dtx(config.use_dtx).map_err(CodecError::Encode)?;

        // In-band FEC embeds a loss-resilient copy of the prior frame. It
        // only engages when the encoder believes packets may be lost, so we
        // assume a conservative 10% loss rate.
        encoder
            .set_inband_fec(config.use_fec)
            .map_err(CodecError::Encode)?;
        if config.use_fec {
            encoder
                .set_packet_loss_perc(10)
                .map_err(CodecError::Encode)?;
        }

        // Hint that the input is speech so libopus picks its voice-optimized
        // modes (e.g. CELT layer tuning) instead of relying on detection.
        encoder.set_signal(Signal::Voice).map_err(CodecError::Encode)?;

        let samples_per_frame = config.samples_per_frame();
        tracing::debug!(
            sample_rate = config.sample_rate,
            channels = config.channels,
            bitrate = config.bitrate,
            frame_size_ms = config.frame_size_ms,
            samples_per_frame,
            dtx = config.use_dtx,
            fec = config.use_fec,
            complexity = config.complexity,
            "opus voice encoder initialized"
        );

        Ok(Self {
            config,
            samples_per_frame,
            encoder: Mutex::new(encoder),
        })
    }

    /// The validated configuration in use.
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }

    /// Number of PCM samples expected per [`OpusEncoder::encode`] call.
    pub fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }

    /// Encode exactly one frame of interleaved PCM into an Opus packet.
    ///
    /// `pcm` must contain exactly [`OpusEncoder::samples_per_frame`] samples
    /// (all channels interleaved). Returns the encoded packet bytes, which
    /// may be as small as a single byte when DTX suppresses silence.
    pub fn encode(&self, pcm: &[i16]) -> Result<Vec<u8>> {
        if pcm.len() != self.samples_per_frame {
            return Err(CodecError::FrameSizeMismatch {
                actual: pcm.len(),
                expected: self.samples_per_frame,
                channels: self.config.channels,
            });
        }

        let mut encoder = self.encoder.lock().map_err(|_| CodecError::Poisoned)?;
        // encode_vec allocates the output; MAX_PACKET_SIZE is the hard
        // ceiling for any Opus packet so a single call can never truncate.
        let packet = encoder
            .encode_vec(pcm, MAX_PACKET_SIZE)
            .map_err(CodecError::Encode)?;
        tracing::trace!(bytes = packet.len(), "encoded opus frame");
        Ok(packet)
    }

    /// Encode a PCM buffer by splitting it into frame-size chunks.
    ///
    /// Returns one packet per frame. If the buffer is not an exact multiple
    /// of the frame size, the trailing partial frame is zero-padded to a full
    /// frame so the decoder can reconstruct continuous audio (silence at the
    /// tail is preferable to dropping the tail of an utterance).
    pub fn encode_frames(&mut self, pcm: &[i16]) -> Result<Vec<Vec<u8>>> {
        let frame = self.samples_per_frame;
        let mut packets = Vec::with_capacity(pcm.len() / frame.max(1) + 1);

        for chunk in pcm.chunks(frame) {
            // The final chunk may be short; pad with silence to a full frame.
            let mut frame_pcm = chunk.to_vec();
            frame_pcm.resize(frame, 0);
            packets.push(self.encode(&frame_pcm)?);
        }

        tracing::trace!(
            frames = packets.len(),
            input_samples = pcm.len(),
            "encoded pcm buffer into opus frames"
        );
        Ok(packets)
    }
}

/// Opus decoder for real-time voice.
///
/// Mirrors [`OpusEncoder`]: same frame sizing, plus packet-loss concealment
/// and FEC-based recovery for robustness on lossy networks.
#[derive(Debug)]
pub struct OpusDecoder {
    /// Validated configuration this decoder was built from.
    config: CodecConfig,
    /// Expected PCM samples per frame (all channels interleaved).
    samples_per_frame: usize,
    /// The libopus decoder.
    decoder: opus::Decoder,
}

impl OpusDecoder {
    /// Create a decoder matching the given configuration.
    pub fn new(config: CodecConfig) -> Result<Self> {
        config.validate()?;

        let samples_per_frame = config.samples_per_frame();
        let decoder =
            opus::Decoder::new(config.sample_rate, config.opus_channels()).map_err(|e| {
                CodecError::InvalidConfig(format!("failed to create opus decoder: {e}"))
            })?;

        tracing::debug!(
            sample_rate = config.sample_rate,
            channels = config.channels,
            samples_per_frame,
            "opus voice decoder initialized"
        );

        Ok(Self {
            config,
            samples_per_frame,
            decoder,
        })
    }

    /// The validated configuration in use.
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }

    /// Number of PCM samples produced per full frame.
    pub fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }

    /// Decode one Opus packet into interleaved 16-bit PCM.
    ///
    /// An empty `packet` is treated as a lost packet: libopus's packet-loss
    /// concealment (PLC) synthesizes plausible continuation audio, which is
    /// exactly what a jitter buffer wants on a late/missing packet.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<i16>> {
        let out = self.decode_impl(packet, false)?;
        tracing::trace!(packet_bytes = packet.len(), samples = out.len(), "decoded opus packet");
        Ok(out)
    }

    /// Decode a packet using in-band FEC / loss concealment.
    ///
    /// Call this when a packet is known to be missing but the *next* packet
    /// has arrived: libopus reconstructs the missing frame from the FEC data
    /// embedded in the following packet, producing far cleaner speech than
    /// pure concealment. Passing an empty slice invokes plain PLC.
    pub fn decode_fec(&mut self, packet: &[u8]) -> Result<Vec<i16>> {
        let out = self.decode_impl(packet, true)?;
        tracing::trace!(
            packet_bytes = packet.len(),
            samples = out.len(),
            "decoded opus packet with fec/concealment"
        );
        Ok(out)
    }

    /// Shared decode path: size the output buffer from the packet header when
    /// possible (packets may legally carry a different duration than the
    /// configured frame), falling back to the configured frame size for
    /// empty/PLC packets.
    fn decode_impl(&mut self, packet: &[u8], fec: bool) -> Result<Vec<i16>> {
        let channels = self.config.channels as usize;

        // Samples *per channel* according to the packet's own TOC header.
        let per_channel = if packet.is_empty() {
            self.samples_per_frame / channels
        } else {
            self.decoder
                .get_nb_samples(packet)
                .map_err(CodecError::Decode)?
        };

        let mut output = vec![0i16; per_channel * channels];
        let decoded = self
            .decoder
            .decode(packet, &mut output, fec)
            .map_err(CodecError::Decode)?;

        // `decode` returns samples per channel; keep only what was written.
        output.truncate(decoded * channels);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 440 Hz sine wave at a comfortably audible amplitude.
    fn sine_frame(freq: f32, sample_rate: u32, len: usize, amplitude: i16) -> Vec<i16> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (amplitude as f32 * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn default_config_matches_voice_tuning() {
        let cfg = CodecConfig::default();
        assert_eq!(cfg.sample_rate, 48_000);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.bitrate, 24_000);
        assert_eq!(cfg.frame_size_ms, 20.0);
        assert!(cfg.use_dtx);
        assert!(cfg.use_fec);
        assert_eq!(cfg.complexity, 0);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(OpusEncoder::new(CodecConfig {
            sample_rate: 44_100,
            ..CodecConfig::default()
        })
        .is_err());

        assert!(OpusEncoder::new(CodecConfig {
            frame_size_ms: 15.0,
            ..CodecConfig::default()
        })
        .is_err());

        assert!(OpusEncoder::new(CodecConfig {
            channels: 5,
            ..CodecConfig::default()
        })
        .is_err());

        assert!(OpusDecoder::new(CodecConfig {
            complexity: 11,
            ..CodecConfig::default()
        })
        .is_err());
    }

    #[test]
    fn encode_then_decode_sine_roundtrip() {
        let config = CodecConfig::default();
        let samples_per_frame = config.samples_per_frame(); // 960 @ 48 kHz mono

        let encoder = OpusEncoder::new(config.clone()).expect("encoder");
        let mut decoder = OpusDecoder::new(config).expect("decoder");

        let pcm = sine_frame(440.0, 48_000, samples_per_frame, 8_000);
        assert_eq!(pcm.len(), encoder.samples_per_frame());

        let packet = encoder.encode(&pcm).expect("encode");
        assert!(!packet.is_empty(), "encoded packet must not be empty");
        // 24 kbps over 20 ms should stay far below the 1275-byte ceiling.
        assert!(packet.len() <= MAX_PACKET_SIZE);

        let decoded = decoder.decode(&packet).expect("decode");
        assert_eq!(
            decoded.len(),
            samples_per_frame,
            "decoded output must match one full frame"
        );

        // Non-silent: the reconstructed wave must retain real energy.
        let peak = decoded.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 1_000, "decoded sine wave appears silent (peak {peak})");
    }

    #[test]
    fn encode_frames_splits_into_frame_chunks() {
        let config = CodecConfig::default();
        let frame = config.samples_per_frame(); // 960

        let mut encoder = OpusEncoder::new(config).expect("encoder");

        // 2.5 frames worth of audio: two full frames + a padded partial.
        let pcm = sine_frame(440.0, 48_000, frame * 2 + frame / 2, 8_000);
        let packets = encoder.encode_frames(&pcm).expect("encode_frames");

        assert_eq!(packets.len(), 3, "partial tail frame must become its own packet");
        assert!(packets.iter().all(|p| !p.is_empty()));

        // Decoding the stream yields 3 full frames of audio.
        let mut decoder = OpusDecoder::new(CodecConfig::default()).expect("decoder");
        let total: usize = packets
            .iter()
            .map(|p| decoder.decode(p).expect("decode").len())
            .sum();
        assert_eq!(total, frame * 3);
    }

    #[test]
    fn encode_rejects_wrong_frame_size() {
        let encoder = OpusEncoder::new(CodecConfig::default()).expect("encoder");
        let too_short = vec![0i16; encoder.samples_per_frame() - 1];
        assert!(matches!(
            encoder.encode(&too_short),
            Err(CodecError::FrameSizeMismatch { .. })
        ));
    }

    #[test]
    fn dtx_produces_smaller_packets_on_silence() {
        let frame = CodecConfig::default().samples_per_frame();
        let silence = vec![0i16; frame];

        // DTX encoder: after a few silent frames libopus stops sending real
        // packets entirely, emitting 1-byte "keepalive" markers instead.
        let mut dtx_encoder = OpusEncoder::new(CodecConfig {
            use_dtx: true,
            use_fec: false,
            ..CodecConfig::default()
        })
        .expect("dtx encoder");
        let dtx_packets = dtx_encoder.encode_frames(&silence.repeat(20)).expect("encode");
        let dtx_min = dtx_packets.iter().map(|p| p.len()).min().unwrap();

        // No-DTX baseline: every silent frame still gets a real (if tiny)
        // packet; libopus never emits the 1-byte DTX marker.
        let mut plain_encoder = OpusEncoder::new(CodecConfig {
            use_dtx: false,
            use_fec: false,
            ..CodecConfig::default()
        })
        .expect("plain encoder");
        let plain_packets = plain_encoder.encode_frames(&silence.repeat(20)).expect("encode");
        let plain_min = plain_packets.iter().map(|p| p.len()).min().unwrap();

        assert!(
            dtx_min <= 2,
            "expected DTX to emit near-empty packets on silence, got min {dtx_min} bytes"
        );
        assert!(
            dtx_min < plain_min,
            "DTX (min {dtx_min}) should beat no-DTX (min {plain_min}) on silence"
        );
    }

    #[test]
    fn decode_fec_conceals_lost_packet() {
        let config = CodecConfig::default();
        let frame = config.samples_per_frame();

        let encoder = OpusEncoder::new(config.clone()).expect("encoder");
        let mut decoder = OpusDecoder::new(config).expect("decoder");

        // Prime the encoder/decoder with a real frame, then "lose" the next
        // packet: an empty packet asks libopus for loss concealment.
        let first = encoder.encode(&sine_frame(440.0, 48_000, frame, 8_000)).expect("encode");
        let _next = encoder.encode(&sine_frame(440.0, 48_000, frame, 8_000)).expect("encode");

        assert_eq!(decoder.decode(&first).expect("decode").len(), frame);

        let concealed = decoder.decode_fec(&[]).expect("decode_fec");
        assert_eq!(
            concealed.len(),
            frame,
            "concealed frame must produce a full frame of audio"
        );
    }
}
