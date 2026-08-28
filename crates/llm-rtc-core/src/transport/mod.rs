//! Transport layer configuration for WebRTC (DTLS, SRTP).
//!
//! The webrtc crate implements DTLS/SRTP internally; these modules expose
//! configuration and stats knobs.

pub mod dtls;
pub mod srtp;
