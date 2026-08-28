//! SRTP transport configuration.
//!
//! SRTP keying (via DTLS-SRTP) and packet encryption/decryption are performed
//! entirely by the `webrtc` crate inside the peer connection's media
//! pipeline. This module does **not** implement any cryptography; it only
//! exposes configuration knobs (protection profile, RTP/RTCP muxing, and
//! replay protection window size) and validates them.

use thiserror::Error;

/// Errors that can occur while validating SRTP configuration.
///
/// Actual protect/unprotect failures surface from the `webrtc` crate at
/// runtime; this error type only covers local configuration validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SrtpError {
    /// The requested SRTP protection profile is not supported.
    #[error("unsupported SRTP profile: {0} (supported: DTLS_SRTP_SHA2_256, DTLS_SRTP_SHA1_80)")]
    UnsupportedProfile(String),
}

/// Result type for SRTP configuration helpers.
pub type Result<T> = std::result::Result<T, SrtpError>;

/// Configuration for the SRTP transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtpConfig {
    /// SRTP protection profile negotiated via DTLS-SRTP.
    /// Defaults to `"DTLS_SRTP_SHA2_256"` (the strongest supported profile).
    pub profile: String,
    /// Multiplex RTP and RTCP on a single transport ("rtcp-mux").
    /// Defaults to `true`, as required by most modern WebRTC endpoints.
    pub enable_rtp_rtcp_mux: bool,
    /// Size of the SRTP anti-replay window (number of sequence numbers).
    /// Defaults to `64`.
    pub replay_protection_window: usize,
}

impl Default for SrtpConfig {
    fn default() -> Self {
        Self {
            profile: "DTLS_SRTP_SHA2_256".to_string(),
            enable_rtp_rtcp_mux: true,
            replay_protection_window: 64,
        }
    }
}

/// Profiles supported by llm-rtc, mirroring the SDP/DTLS-SRTP profile names.
const SUPPORTED_PROFILES: &[&str] = &["DTLS_SRTP_SHA2_256", "DTLS_SRTP_SHA1_80"];

impl SrtpConfig {
    /// Validate the configuration, returning `Ok(())` when the profile is
    /// supported.
    pub fn validate(&self) -> Result<()> {
        validate_profile(&self.profile)
    }
}

/// Check that an SRTP protection profile name is supported.
fn validate_profile(profile: &str) -> Result<()> {
    if SUPPORTED_PROFILES
        .iter()
        .any(|p| p.eq_ignore_ascii_case(profile))
    {
        tracing::debug!("SRTP profile: {profile}");
        Ok(())
    } else {
        tracing::debug!("rejected unsupported SRTP profile: {profile}");
        Err(SrtpError::UnsupportedProfile(profile.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = SrtpConfig::default();
        assert_eq!(cfg.profile, "DTLS_SRTP_SHA2_256");
        assert!(cfg.enable_rtp_rtcp_mux);
        assert_eq!(cfg.replay_protection_window, 64);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn profile_validation() {
        assert!(validate_profile("DTLS_SRTP_SHA2_256").is_ok());
        assert!(validate_profile("dtls_srtp_sha1_80").is_ok());
        assert!(matches!(
            validate_profile("DTLS_SRTP_AES128_GCM"),
            Err(SrtpError::UnsupportedProfile(_))
        ));
    }
}
