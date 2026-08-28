//! WebRTC peer-side demo: create a voice session and print its SDP offer.
//!
//! This shows the signaling half of llm-rtc. In a real deployment the
//! printed offer would be sent to the remote endpoint (a voice LLM server),
//! which answers with its own SDP that you'd apply via
//! `set_remote_description`.
//!
//! Run with:
//! ```sh
//! cargo run -p llm-rtc-sdk --example peer_offer
//! ```

use llm_rtc_sdk::session::{SessionConfig, VoiceLlmSession};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Creating voice LLM session...");

    let session = VoiceLlmSession::new(SessionConfig::default()).await?;

    // Build the SDP offer describing our Opus audio track and ICE setup.
    let offer = session.create_offer().await?;

    println!();
    println!("SDP offer:");
    println!("----------");
    println!("{}", offer.sdp);

    session.close().await?;
    Ok(())
}
