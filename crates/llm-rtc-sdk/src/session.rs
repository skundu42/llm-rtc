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
use tokio::sync::{watch, Notify};
use tokio::task::JoinHandle;
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
    /// Reusable outer storage for encoded packets.
    outgoing_packets: Vec<Vec<u8>>,
    /// Duration of one Opus frame (used for RTP packetization timing).
    frame_duration: Duration,
}

type RemoteAudioCallback = Arc<dyn Fn(Vec<i16>) + Send + Sync + 'static>;

/// Run playout on RTP deadlines instead of coupling it to packet arrival.
/// Once the receiver exits, the task keeps sleeping and decoding until every
/// buffered tail packet has been played.
fn spawn_playout_task(
    pipeline: Arc<Mutex<AudioPipeline>>,
    cb: RemoteAudioCallback,
    frame_duration: Duration,
    mut receiver_done: watch::Receiver<bool>,
    packet_ready: Arc<Notify>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let recovery_interval = frame_duration.div_f64(2.0).max(Duration::from_millis(1));
        let mut next_decode_at = tokio::time::Instant::now();

        loop {
            let (deadline, pending) = {
                let pipeline = pipeline.lock().expect("pipeline mutex poisoned");
                (
                    pipeline.next_playout_deadline(),
                    pipeline.has_pending_playout(),
                )
            };

            if *receiver_done.borrow() && !pending {
                break;
            }

            let Some(deadline) = deadline else {
                tokio::select! {
                    _ = packet_ready.notified() => {}
                    changed = receiver_done.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                }
                continue;
            };

            let wake_at = tokio::time::Instant::from_std(deadline).max(next_decode_at);
            tokio::select! {
                _ = tokio::time::sleep_until(wake_at) => {}
                _ = packet_ready.notified() => continue,
            }

            let (decoded, pending) = {
                let mut pipeline = pipeline.lock().expect("pipeline mutex poisoned");
                let decoded = match pipeline.pop_decoded() {
                    Ok(decoded) => decoded,
                    Err(e) => {
                        debug!("playout decode failed: {e}");
                        None
                    }
                };
                (decoded, pipeline.has_pending_playout())
            };

            if let Some(pcm) = decoded {
                cb(pcm);
                // A delayed task may catch up, but never emits faster than 2x
                // the media rate and therefore does not burst stale audio.
                next_decode_at = tokio::time::Instant::now() + recovery_interval;
            }

            if *receiver_done.borrow() && !pending {
                break;
            }
        }

        debug!("remote audio playout task finished");
    })
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
            outgoing_packets: Vec::new(),
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
    /// Internally this spawns two background tasks per incoming track:
    ///
    /// 1. reads RTP packets off the remote track,
    /// 2. pushes them into the jitter buffer,
    /// 3. sleeps independently until each RTP playout deadline,
    /// 4. pops RTP-deadline events and applies normal decode, FEC, or PLC,
    /// 5. invokes `cb` with the decoded PCM samples.
    ///
    /// The receiver exits when the track closes; playout exits after draining
    /// the buffered tail.
    pub fn on_remote_audio(&self, cb: impl Fn(Vec<i16>) + Send + Sync + 'static) {
        let cb: RemoteAudioCallback = Arc::new(cb);
        let pipeline = Arc::clone(&self.pipeline);
        let frame_duration = self.frame_duration;

        self.peer.on_track(move |track: Arc<RemoteTrack>| {
            debug!("remote audio track added; starting receive and playout tasks");
            let pipeline = Arc::clone(&pipeline);
            let cb = Arc::clone(&cb);
            let (receiver_done_tx, receiver_done_rx) = watch::channel(false);
            let packet_ready = Arc::new(Notify::new());

            spawn_playout_task(
                Arc::clone(&pipeline),
                cb,
                frame_duration,
                receiver_done_rx,
                Arc::clone(&packet_ready),
            );

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

                    // Packet arrival only feeds the jitter buffer. A separate
                    // media-clock task performs deadline-driven playout.
                    let wake_playout = {
                        let mut pipeline = pipeline.lock().expect("pipeline mutex poisoned");
                        let previous_deadline = pipeline.next_playout_deadline();
                        if !pipeline.push_incoming(packet) {
                            debug!("jitter buffer dropped late/duplicate packet");
                        }
                        pipeline.next_playout_deadline() != previous_deadline
                    };
                    if wake_playout {
                        packet_ready.notify_one();
                    }
                }

                let _ = receiver_done_tx.send(true);
                debug!("remote audio receiver task finished");
            });
        });
    }

    /// Send one frame of microphone audio to the remote endpoint.
    ///
    /// The frame is processed by the audio pipeline (AEC/NS/AGC) and Opus
    /// encoded; the resulting packets are written to the local track.
    pub async fn send_audio(&mut self, mic_pcm: &mut [i16]) -> Result<()> {
        let mut packets = std::mem::take(&mut self.outgoing_packets);
        let encoded = {
            let mut pipeline = self.pipeline.lock().expect("pipeline mutex poisoned");
            pipeline.process_outgoing_into(mic_pcm, &mut packets)
        };
        if let Err(error) = encoded {
            self.outgoing_packets = packets;
            return Err(error.into());
        }
        let result = self.write_packets(&mut packets).await;
        self.outgoing_packets = packets;
        result
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
        let mut packets = std::mem::take(&mut self.outgoing_packets);
        let encoded = {
            let mut pipeline = self.pipeline.lock().expect("pipeline mutex poisoned");
            pipeline.process_outgoing_with_reference_into(mic_pcm, far_end, &mut packets)
        };
        if let Err(error) = encoded {
            self.outgoing_packets = packets;
            return Err(error.into());
        }
        let result = self.write_packets(&mut packets).await;
        self.outgoing_packets = packets;
        result
    }

    /// Write encoded Opus packets to the local WebRTC track.
    async fn write_packets(&self, packets: &mut Vec<Vec<u8>>) -> Result<()> {
        for packet in packets.drain(..) {
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

        // One default 10 ms frame of a 440 Hz sine wave at 48 kHz mono.
        let frame_size = 48_000 * 10 / 1000;
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
        let packet_capacity = session.outgoing_packets.capacity();
        session
            .send_audio(&mut frame)
            .await
            .expect("second send_audio should succeed");
        assert_eq!(session.outgoing_packets.capacity(), packet_capacity);

        session.close().await.expect("close should succeed");
    }

    /// Playout continues after packet ingestion has stopped, so the final
    /// buffered frames are delivered on their RTP deadlines rather than being
    /// stranded waiting for another network arrival.
    #[tokio::test]
    async fn playout_task_drains_buffered_tail_after_receiver_finishes() {
        let config = AudioPipelineConfig {
            jitter: llm_rtc_core::audio::jitter::JitterBufferConfig {
                target_latency_ms: 20,
                max_latency_ms: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let codec_config = config.codec.clone();
        let frame_duration =
            Duration::from_secs_f64(f64::from(codec_config.frame_size_ms) / 1_000.0);
        let pipeline = Arc::new(Mutex::new(AudioPipeline::new(config).unwrap()));
        let encoder = llm_rtc_core::audio::codec::OpusEncoder::new(codec_config).unwrap();
        let samples_per_frame = encoder.samples_per_frame();
        let frame = vec![1_000i16; samples_per_frame];

        let (receiver_done_tx, receiver_done_rx) = watch::channel(false);
        let (frames_tx, mut frames_rx) = tokio::sync::mpsc::unbounded_channel();
        let cb: RemoteAudioCallback = Arc::new(move |pcm| {
            let _ = frames_tx.send(pcm);
        });
        let packet_ready = Arc::new(Notify::new());
        let task = spawn_playout_task(
            Arc::clone(&pipeline),
            cb,
            frame_duration,
            receiver_done_rx,
            Arc::clone(&packet_ready),
        );

        for seq in 0..2u16 {
            let packet = AudioPacket {
                sequence_number: seq,
                timestamp: u32::from(seq) * samples_per_frame as u32,
                payload: encoder.encode(&frame).unwrap(),
            };
            assert!(pipeline.lock().unwrap().push_incoming(packet));
        }
        packet_ready.notify_one();
        receiver_done_tx.send(true).unwrap();

        for _ in 0..2 {
            let pcm = tokio::time::timeout(Duration::from_millis(200), frames_rx.recv())
                .await
                .expect("playout frame timed out")
                .expect("playout task closed the callback channel");
            assert_eq!(pcm.len(), samples_per_frame);
        }
        tokio::time::timeout(Duration::from_millis(200), task)
            .await
            .expect("playout task did not drain and stop")
            .expect("playout task panicked");
        assert!(!pipeline.lock().unwrap().has_pending_playout());
    }
}
