//! DTLS transport configuration.
//!
//! The DTLS handshake itself (certificate generation, key exchange, and
//! certificate verification) is performed entirely by the `webrtc` crate as
//! part of the `RTCPeerConnection` setup. This module does **not** implement
//! any cryptography; it only exposes the configuration knobs that llm-rtc
//! forwards to the underlying webrtc-rs DTLS transport, plus helpers for
//! validating user-supplied settings.

use thiserror::Error;

/// Supported DTLS certificate fingerprint (hash) algorithms.
///
/// Mirrors the algorithms accepted in SDP `a=fingerprint` lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlsFingerprintAlgorithm {
    /// SHA-256 (recommended, the default).
    Sha256,
    /// SHA-1 (legacy, kept for interoperability with old endpoints).
    Sha1,
}

impl DtlsFingerprintAlgorithm {
    /// SDP-style name of the algorithm (e.g. `"sha-256"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sha256 => "sha-256",
            Self::Sha1 => "sha-1",
        }
    }
}

/// Errors that can occur while validating DTLS configuration.
///
/// Note: actual handshake failures surface from the `webrtc` crate at
/// connection time; this error type only covers local configuration
/// validation performed by llm-rtc.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DtlsError {
    /// The requested fingerprint algorithm is not supported.
    #[error("unsupported DTLS fingerprint algorithm: {0} (supported: sha-256, sha-1)")]
    UnsupportedFingerprintAlgorithm(String),
}

/// Result type for DTLS configuration helpers.
pub type Result<T> = std::result::Result<T, DtlsError>;

/// Configuration for the DTLS transport.
///
/// The handshake itself is handled by the `webrtc` crate; this struct only
/// carries the settings llm-rtc passes through (or uses for validation) when
/// setting up a peer connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtlsConfig {
    /// Fingerprint hash algorithm advertised/verified in SDP.
    /// Defaults to `"sha-256"`.
    pub fingerprint_algorithm: String,
    /// Skip certificate verification (insecure; development only).
    /// Defaults to `false`.
    pub insecure_skip_verify: bool,
}

impl Default for DtlsConfig {
    fn default() -> Self {
        Self {
            fingerprint_algorithm: "sha-256".to_string(),
            insecure_skip_verify: false,
        }
    }
}

impl DtlsConfig {
    /// Validate the configuration, returning a typed algorithm enum.
    pub fn validate(&self) -> Result<DtlsFingerprintAlgorithm> {
        fingerprint_algorithm_enum(&self.fingerprint_algorithm)
    }
}

/// Map an SDP-style algorithm name to [`DtlsFingerprintAlgorithm`].
///
/// Accepts `"sha-256"` and `"sha-1"` (case-insensitive). This documents the
/// set of algorithms llm-rtc supports; anything else is rejected.
pub fn fingerprint_algorithm_enum(alg: &str) -> Result<DtlsFingerprintAlgorithm> {
    match alg.to_ascii_lowercase().as_str() {
        "sha-256" => {
            tracing::debug!("DTLS fingerprint algorithm: sha-256");
            Ok(DtlsFingerprintAlgorithm::Sha256)
        }
        "sha-1" => {
            tracing::debug!("DTLS fingerprint algorithm: sha-1");
            Ok(DtlsFingerprintAlgorithm::Sha1)
        }
        other => {
            tracing::debug!("rejected unsupported DTLS fingerprint algorithm: {other}");
            Err(DtlsError::UnsupportedFingerprintAlgorithm(
                other.to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = DtlsConfig::default();
        assert_eq!(cfg.fingerprint_algorithm, "sha-256");
        assert!(!cfg.insecure_skip_verify);
        assert_eq!(cfg.validate(), Ok(DtlsFingerprintAlgorithm::Sha256));
    }

    #[test]
    fn fingerprint_parsing() {
        assert_eq!(
            fingerprint_algorithm_enum("sha-256"),
            Ok(DtlsFingerprintAlgorithm::Sha256)
        );
        assert_eq!(
            fingerprint_algorithm_enum("sha-1"),
            Ok(DtlsFingerprintAlgorithm::Sha1)
        );
        assert!(matches!(
            fingerprint_algorithm_enum("sha-512"),
            Err(DtlsError::UnsupportedFingerprintAlgorithm(_))
        ));
    }
}
