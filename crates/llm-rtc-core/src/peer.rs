//! WebRTC peer connection module.
//!
//! Wraps the `webrtc` crate (webrtc-rs) to establish a `PeerConnection`
//! with a remote client and carry Opus audio with minimal negotiation
//! overhead.

use std::sync::Arc;

use anyhow::Result;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_OPUS};
use webrtc::api::APIBuilder;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::policy::bundle_policy::RTCBundlePolicy;
use webrtc::peer_connection::policy::ice_transport_policy::RTCIceTransportPolicy;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp::header::Header;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::rtp_transceiver::rtp_transceiver_direction::RTCRtpTransceiverDirection;
use webrtc::rtp_transceiver::RTCRtpTransceiverInit;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_remote::TrackRemote;

/// Configuration for creating a [`PeerConnectionHandle`].
#[derive(Debug, Clone)]
pub struct PeerConfig {
    /// STUN/TURN server URLs, e.g. `stun:stun.l.google.com:19302`.
    pub ice_servers: Vec<String>,
    /// MIME type of the audio codec to negotiate. Defaults to Opus.
    pub codec_mime_type: String,
    /// ICE transport policy. `"all"` (default) or `"relay"`.
    pub ice_transport_policy: String,
    /// Bundle policy. `"balanced"` (default), `"max-compat"`, or `"max-bundle"`.
    pub bundle_policy: String,
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self {
            ice_servers: vec!["stun:stun.l.google.com:19302".to_string()],
            codec_mime_type: MIME_TYPE_OPUS.to_string(),
            ice_transport_policy: "all".to_string(),
            bundle_policy: "balanced".to_string(),
        }
    }
}

/// Handle around a remote incoming audio track.
pub struct RemoteTrack {
    /// The underlying remote track, exposed for advanced use.
    pub track: Arc<TrackRemote>,
}

impl RemoteTrack {
    /// Read the next RTP packet into `buf`.
    ///
    /// Returns the RTP packet header and the payload size written into
    /// `buf` (payload occupies the first `size` bytes of `buf`).
    pub async fn read_rtp(&self, buf: &mut [u8]) -> Result<(Header, usize)> {
        let (pkt, _) = self.track.read(buf).await?;
        let n = pkt.payload.len();
        Ok((pkt.header, n))
    }
}

/// Handle wrapping a WebRTC [`RTCPeerConnection`].
pub struct PeerConnectionHandle {
    pc: Arc<RTCPeerConnection>,
}

impl PeerConnectionHandle {
    /// Build a new peer connection from the given config.
    pub async fn new(config: PeerConfig) -> Result<Self> {
        let mut m = MediaEngine::default();
        m.register_default_codecs()?;

        let mut registry = Registry::new();
        registry = register_default_interceptors(registry, &mut m)?;

        let api = APIBuilder::new()
            .with_media_engine(m)
            .with_interceptor_registry(registry)
            .build();

        let ice_servers: Vec<RTCIceServer> = config
            .ice_servers
            .iter()
            .map(|url| RTCIceServer {
                urls: vec![url.clone()],
                ..Default::default()
            })
            .collect();

        let rtc_config = RTCConfiguration {
            ice_servers,
            ice_transport_policy: parse_ice_transport_policy(&config.ice_transport_policy),
            bundle_policy: parse_bundle_policy(&config.bundle_policy),
            ..Default::default()
        };

        let pc = Arc::new(api.new_peer_connection(rtc_config).await?);

        Ok(Self { pc })
    }

    /// Register a callback invoked on peer connection state changes.
    pub fn on_connection_state_change(
        &self,
        cb: impl Fn(RTCPeerConnectionState) + Send + Sync + 'static,
    ) {
        self.pc.on_peer_connection_state_change(Box::new(move |s| {
            cb(s);
            Box::pin(async {})
        }));
    }

    /// Register a callback invoked when a remote track arrives.
    pub fn on_track(&self, cb: impl Fn(Arc<RemoteTrack>) + Send + Sync + 'static) {
        self.pc
            .on_track(Box::new(move |track, _receiver, _streams| {
                let remote = Arc::new(RemoteTrack { track });
                cb(remote);
                Box::pin(async {})
            }));
    }

    /// Create an SDP offer.
    pub async fn create_offer(&self) -> Result<RTCSessionDescription> {
        let offer = self.pc.create_offer(None).await?;
        Ok(offer)
    }

    /// Create an SDP answer.
    pub async fn create_answer(&self) -> Result<RTCSessionDescription> {
        let answer = self.pc.create_answer(None).await?;
        Ok(answer)
    }

    /// Apply a remote session description.
    pub async fn set_remote_description(&self, sdp: RTCSessionDescription) -> Result<()> {
        self.pc.set_remote_description(sdp).await?;
        Ok(())
    }

    /// Apply a local session description.
    pub async fn set_local_description(&self, sdp: RTCSessionDescription) -> Result<()> {
        self.pc.set_local_description(sdp).await?;
        Ok(())
    }

    /// Add a local Opus audio track that callers can write samples into.
    pub async fn add_opus_track(&self, ssid: u32) -> Result<Arc<TrackLocalStaticSample>> {
        let codec = RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_string(),
            ..Default::default()
        };
        let track = Arc::new(TrackLocalStaticSample::new(
            codec,
            format!("audio-{}", ssid),
            format!("llm-rtc-stream-{}", ssid),
        ));

        let init = RTCRtpTransceiverInit {
            direction: RTCRtpTransceiverDirection::Sendonly,
            send_encodings: vec![],
        };
        self.pc
            .add_transceiver_from_track(track.clone(), Some(init))
            .await?;
        Ok(track)
    }

    /// Close the underlying peer connection.
    pub async fn close(&self) -> Result<()> {
        self.pc.close().await?;
        Ok(())
    }
}

fn parse_ice_transport_policy(s: &str) -> RTCIceTransportPolicy {
    if s.eq_ignore_ascii_case("relay") {
        RTCIceTransportPolicy::Relay
    } else {
        RTCIceTransportPolicy::All
    }
}

fn parse_bundle_policy(s: &str) -> RTCBundlePolicy {
    match s.to_ascii_lowercase().as_str() {
        "max-compat" => RTCBundlePolicy::MaxCompat,
        "max-bundle" => RTCBundlePolicy::MaxBundle,
        _ => RTCBundlePolicy::Balanced,
    }
}

#[cfg(test)]
mod tests {
    use webrtc::track::track_local::TrackLocal;

    use super::*;

    #[tokio::test]
    async fn test_new_succeeds_with_default_config() {
        let handle = PeerConnectionHandle::new(PeerConfig::default())
            .await
            .expect("peer connection creation failed");
        handle.close().await.expect("close failed");
    }

    #[tokio::test]
    async fn test_add_opus_track_succeeds() {
        let handle = PeerConnectionHandle::new(PeerConfig::default())
            .await
            .expect("peer connection creation failed");
        let track = handle
            .add_opus_track(42)
            .await
            .expect("add_opus_track failed");
        assert!(!track.id().is_empty());
        handle.close().await.expect("close failed");
    }

    #[tokio::test]
    async fn test_create_offer_produces_non_empty_sdp() {
        let handle = PeerConnectionHandle::new(PeerConfig::default())
            .await
            .expect("peer connection creation failed");
        let offer = handle.create_offer().await.expect("create_offer failed");
        assert!(!offer.sdp.is_empty(), "offer SDP must not be empty");
        handle.close().await.expect("close failed");
    }
}
