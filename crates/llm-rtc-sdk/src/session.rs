//! High-level full-duplex voice LLM session.
//!
//! This module is the facade of llm-rtc: it ties together the low-level
//! WebRTC engine ([`PeerConnectionHandle`]) and the real-time audio
//! pipeline ([`AudioPipeline`]) into a single first-class object, a
//! [`VoiceLlmSession`], representing a full-duplex voice call between a
//! user (microphone + speaker) and a remote voice LLM endpoint.
//!
//! Typical usage:
//!
//! 1. Build a [`SessionConfig`] (or use the default).
//! 2. `VoiceLlmSession::new(config).await` to create the peer connection
//!    and the audio pipeline.
//! 3. Register `on_remote_audio` to receive decoded PCM playout frames.
//! 4. Exchange SDP with the remote endpoint via `create_offer` /
//!    `create_answer` / `set_remote_description`.
//! 5. Feed microphone frames to `send_audio` (or
//!    `send_audio_with_reference` when playout audio is available for
//!    echo cancellation).
//! 6. `close().await` when the session is over.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use llm_rtc_core::audio::jitter::AudioPacket;
use llm_rtc_core::audio::pipeline::PipelineError;
use llm_rtc_core::audio::pipeline::{AudioPipeline, AudioPipelineConfig};
use llm_rtc_core::peer::{PeerConfig, PeerConnectionHandle, RemoteTrack};
use thiserror::Error;
use tracing::debug;
use webrtc::media::Sample;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

/// Errors produced by a [`VoiceLlmSession`].
#[derive(Debug, Error)]
pub enum SessionError {
    /// The audio pipeline (codec, jitter buffer, or processor) failed.
    #[error("pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// The underlying peer connection failed.
    #[error("peer connection error: {0}")]
    Peer(String),

    /// Writing encoded audio to the local track failed.
    #[error("track write error: {0}")]
    Track(String),
}

/// Convenience alias for session results.
pub type Result<T> = std::result::Result<T, SessionError>;

/// Configuration for a [`VoiceLlmSession`].
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// WebRTC peer connection settings (ICE servers, codec, policies).
    pub peer: PeerConfig,
    /// Audio pipeline settings (codec, jitter buffer, AEC/NS/AGC).
    pub pipeline: AudioPipelineConfig,
    /// PCM sample rate in Hz for the microphone/playout path.
    pub sample_rate: u32,
    /// Number of channels in the PCM path (1 = mono).
    pub channels: u8,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            peer: PeerConfig::default(),
            pipeline: AudioPipelineConfig::default(),
            sample_rate: 48_000,
            channels: 1,
        }
    }
}

/// A full-duplex voice session between a user and a remote voice LLM endpoint.
///
/// The session owns the peer connection (which negotiates and transports the
/// audio) and the audio pipeline (which performs echo cancellation, noise
/// suppression, AGC, Opus encode/decode, and jitter buffering).
///
/// The pipeline is shared between the outgoing path (`send_audio`, called
/// from the application) and the incoming path (the background receiver task
/// spawned by [`VoiceLlmSession::on_remote_audio`]), so it is kept behind an
/// `Arc<Mutex<...>>`.
pub struct VoiceLlmSession {
    /// The WebRTC peer connection handle.
    peer: PeerConnectionHandle,
    /// The local audio track that outgoing Opus packets are written to.
    local_track: Arc<TrackLocalStaticSample>,
    /// Shared audio pipeline (AEC/NS/AGC + Opus + jitter buffer).
    pipeline: Arc<Mutex<AudioPipeline>>,
    /// Duration of one Opus frame (used for RTP packetization timing).
    frame_duration: Duration,
}

impl VoiceLlmSession {
    /// Create a new voice LLM session.
    ///
    /// This constructs the underlying peer connection and the audio
    /// pipeline, and adds the outgoing Opus track.
    pub async fn new(config: SessionConfig) -> Result<Self> {
        debug!("creating voice LLM session");

        // Align the pipeline codec and jitter buffer with the requested
        // PCM format so encode/decode expectations stay consistent.
        let mut pipeline_config = config.pipeline;
        pipeline_config.codec.sample_rate = config.sample_rate;
        pipeline_config.codec.channels = config.channels;
        pipeline_config.jitter.sample_rate = config.sample_rate;

        let peer = PeerConnectionHandle::new(config.peer)
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))?;

        let local_track = peer
            .add_opus_track(0)
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))?;

        let frame_duration =
            Duration::from_secs_f64(pipeline_config.codec.frame_size_ms as f64 / 1000.0);

        let pipeline = AudioPipeline::new(pipeline_config)?;

        debug!("voice LLM session created");
        Ok(Self {
            peer,
            local_track,
            pipeline: Arc::new(Mutex::new(pipeline)),
            frame_duration,
        })
    }

    /// Register a callback invoked whenever the peer connection state
    /// changes (e.g. `Connected`, `Failed`, `Closed`).
    pub fn on_connection_state_change(
        &self,
        cb: impl Fn(RTCPeerConnectionState) + Send + Sync + 'static,
    ) {
        self.peer.on_connection_state_change(cb);
    }

    /// Register a callback invoked with decoded PCM whenever remote audio
    /// arrives.
    ///
    /// Internally this spawns a background task per incoming track that:
    ///
    /// 1. reads RTP packets off the remote track,
    /// 2. pushes them into the jitter buffer,
    /// 3. pops and decodes in-order frames,
    /// 4. invokes `cb` with the decoded PCM samples.
    ///
    /// The task exits when the track is closed (read fails).
    pub fn on_remote_audio(&self, cb: impl Fn(Vec<i16>) + Send + Sync + 'static) {
        let cb = Arc::new(cb);
        let pipeline = Arc::clone(&self.pipeline);

        self.peer.on_track(move |track: Arc<RemoteTrack>| {
            debug!("remote audio track added; starting receiver task");
            let pipeline = Arc::clone(&pipeline);
            let cb = Arc::clone(&cb);

            tokio::spawn(async move {
                // Max UDP payload size; RTP payloads are always smaller.
                let mut buf = vec![0u8; 1500];

                loop {
                    let (header, n) = match track.read_rtp(&mut buf).await {
                        Ok((header, n)) => (header, n),
                        Err(e) => {
                            debug!("remote audio track closed: {e}");
                            break;
                        }
                    };

                    let packet = AudioPacket {
                        sequence_number: header.sequence_number,
                        timestamp: header.timestamp,
                        payload: buf[..n].to_vec(),
                    };

                    // Push into the jitter buffer and try to pop the next
                    // in-order decoded frame. The lock is only held across
                    // synchronous pipeline calls.
                    let decoded = {
                        let mut pipeline = pipeline.lock().expect("pipeline mutex poisoned");
                        if !pipeline.push_incoming(packet) {
                            debug!("jitter buffer dropped late/duplicate packet");
                            continue;
                        }
                        match pipeline.pop_decoded() {
                            Ok(decoded) => decoded,
                            Err(e) => {
                                debug!("decode failed: {e}");
                                continue;
                            }
                        }
                    };

                    if let Some(pcm) = decoded {
                        cb(pcm);
                    }
                }

                debug!("remote audio receiver task finished");
            });
        });
    }

    /// Send one frame of microphone audio to the remote endpoint.
    ///
    /// The frame is processed by the audio pipeline (AEC/NS/AGC) and Opus
    /// encoded; the resulting packets are written to the local track.
    pub async fn send_audio(&mut self, mic_pcm: &mut [i16]) -> Result<()> {
        let packets = {
            let mut pipeline = self.pipeline.lock().expect("pipeline mutex poisoned");
            pipeline.process_outgoing(mic_pcm)?
        };
        self.write_packets(packets).await
    }

    /// Send one frame of microphone audio with an explicit far-end (speaker)
    /// reference for echo cancellation.
    ///
    /// Use this when the application knows exactly what audio is being
    /// played out, instead of relying on `on_render` to feed the AEC.
    pub async fn send_audio_with_reference(
        &mut self,
        mic_pcm: &mut [i16],
        far_end: &[i16],
    ) -> Result<()> {
        let packets = {
            let mut pipeline = self.pipeline.lock().expect("pipeline mutex poisoned");
            pipeline.process_outgoing_with_reference(mic_pcm, far_end)?
        };
        self.write_packets(packets).await
    }

    /// Write encoded Opus packets to the local WebRTC track.
    async fn write_packets(&self, packets: Vec<Vec<u8>>) -> Result<()> {
        for packet in packets {
            let sample = Sample {
                data: packet.into(),
                timestamp: SystemTime::now(),
                duration: self.frame_duration,
                packet_timestamp: 0,
                prev_dropped_packets: 0,
                prev_padding_packets: 0,
            };
            self.local_track
                .write_sample(&sample)
                .await
                .map_err(|e| SessionError::Track(e.to_string()))?;
        }
        Ok(())
    }

    /// Create an SDP offer to send to the remote endpoint.
    pub async fn create_offer(&self) -> Result<RTCSessionDescription> {
        self.peer
            .create_offer()
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))
    }

    /// Create an SDP answer in response to a remote offer.
    pub async fn create_answer(&self) -> Result<RTCSessionDescription> {
        self.peer
            .create_answer()
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))
    }

    /// Apply the remote session description (offer or answer).
    pub async fn set_remote_description(&self, sdp: RTCSessionDescription) -> Result<()> {
        self.peer
            .set_remote_description(sdp)
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))
    }

    /// Apply the local session description (offer or answer).
    pub async fn set_local_description(&self, sdp: RTCSessionDescription) -> Result<()> {
        self.peer
            .set_local_description(sdp)
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))
    }

    /// Snapshot of the jitter buffer statistics.
    pub fn jitter_stats(&self) -> llm_rtc_core::audio::jitter::JitterStats {
        self.pipeline
            .lock()
            .expect("pipeline mutex poisoned")
            .jitter_stats()
    }

    /// Snapshot of the audio processor statistics (AEC/NS/AGC).
    pub fn processor_stats(&self) -> llm_rtc_core::audio::processor::ProcessorStats {
        self.pipeline
            .lock()
            .expect("pipeline mutex poisoned")
            .processor_stats()
    }

    /// Close the session, shutting down the peer connection. Background
    /// receiver tasks exit once their tracks fail to read.
    pub async fn close(&self) -> Result<()> {
        debug!("closing voice LLM session");
        self.peer
            .close()
            .await
            .map_err(|e| SessionError::Peer(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh session should be created successfully with default settings.
    #[tokio::test]
    async fn new_session_with_default_config_succeeds() {
        let session = VoiceLlmSession::new(SessionConfig::default())
            .await
            .expect("session creation should succeed");
        session.close().await.expect("close should succeed");
    }

    /// create_offer should produce a non-empty SDP blob.
    #[tokio::test]
    async fn create_offer_produces_non_empty_sdp() {
        let session = VoiceLlmSession::new(SessionConfig::default())
            .await
            .expect("session creation should succeed");
        let offer = session
            .create_offer()
            .await
            .expect("create_offer should succeed");
        assert!(!offer.sdp.is_empty(), "offer SDP should not be empty");
        session.close().await.expect("close should succeed");
    }

    /// Sending a sine-wave microphone frame through the pipeline should not
    /// error, even before the connection is negotiated (writes are buffered
    /// / no-oped until the track is bound).
    #[tokio::test]
    async fn send_audio_with_sine_wave_does_not_error() {
        let mut session = VoiceLlmSession::new(SessionConfig::default())
            .await
            .expect("session creation should succeed");

        // One 20 ms frame of a 440 Hz sine wave at 48 kHz mono: 960 samples.
        let frame_size = 48_000 * 20 / 1000;
        let mut frame: Vec<i16> = (0..frame_size)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                (f64::sin(2.0 * std::f64::consts::PI * 440.0 * t) * 8_000.0) as i16
            })
            .collect();

        session
            .send_audio(&mut frame)
            .await
            .expect("send_audio should succeed");

        session.close().await.expect("close should succeed");
    }
}
