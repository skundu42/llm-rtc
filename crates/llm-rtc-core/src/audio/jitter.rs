//! Low-latency adaptive jitter buffer for voice LLM streaming.
//!
//! The jitter buffer sits between the network and the Opus decoder. It
//! reorders out-of-order RTP audio packets, absorbs network jitter, and hands
//! packets to the decoder strictly in sequence order.
//!
//! Design goals (low-latency-first):
//! * Keep the buffer shallow — buffered depth is trimmed to [`JitterBufferConfig::max_latency_ms`].
//! * Adapt quickly — inter-arrival jitter is estimated continuously with the
//!   RFC 3550 algorithm and exposed via [`JitterStats::current_jitter_ms`].
//! * Prefer dropping late packets over adding delay — a missing packet is only
//!   waited on for [`JitterBufferConfig::target_latency_ms`]; after that the
//!   gap is declared lost and playback advances to the next available packet.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use thiserror::Error;
use tracing::{debug, trace};

/// Errors produced by jitter buffer configuration validation.
#[derive(Debug, Error)]
pub enum JitterError {
    /// Target latency must never exceed the hard maximum latency.
    #[error("target latency ({target_latency_ms} ms) exceeds max latency ({max_latency_ms} ms)")]
    InvalidLatency {
        /// Requested target latency in milliseconds.
        target_latency_ms: u32,
        /// Configured hard maximum latency in milliseconds.
        max_latency_ms: u32,
    },
    /// RTP clock rate must be positive.
    #[error("sample_rate must be greater than zero")]
    InvalidSampleRate,
    /// Frame duration must be positive.
    #[error("frame_size_ms must be greater than zero")]
    InvalidFrameSize,
    /// Buffer capacity must hold at least one packet.
    #[error("max_packets must be at least one")]
    InvalidMaxPackets,
}

/// Tunable parameters for the jitter buffer.
///
/// Defaults are tuned for interactive voice LLM conversations, where a user
/// perceives anything above ~100 ms of added buffering as a laggy turn.
#[derive(Debug, Clone)]
pub struct JitterBufferConfig {
    /// Hard ceiling on buffered audio depth. Oldest packets are dropped when
    /// the buffer would exceed this.
    pub max_latency_ms: u32,
    /// Playout target. Doubles as the maximum time `pop()` will wait for a
    /// missing (in-sequence) packet before declaring it lost and skipping it.
    pub target_latency_ms: u32,
    /// Hard ceiling on the number of buffered packets (safety valve against
    /// memory blowups during long stalls).
    pub max_packets: usize,
    /// RTP clock rate in Hz (Opus uses 48000).
    pub sample_rate: u32,
    /// Duration of one encoded frame in milliseconds (Opus frames are
    /// typically 20 ms).
    pub frame_size_ms: u32,
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 60,
            target_latency_ms: 40,
            max_packets: 100,
            sample_rate: 48_000,
            frame_size_ms: 20,
        }
    }
}

impl JitterBufferConfig {
    /// Strictly validate the configuration.
    ///
    /// [`JitterBuffer::new`] silently sanitizes invalid values instead of
    /// failing; callers that prefer explicit errors can run this first.
    pub fn validate(&self) -> Result<(), JitterError> {
        if self.target_latency_ms > self.max_latency_ms {
            return Err(JitterError::InvalidLatency {
                target_latency_ms: self.target_latency_ms,
                max_latency_ms: self.max_latency_ms,
            });
        }
        if self.sample_rate == 0 {
            return Err(JitterError::InvalidSampleRate);
        }
        if self.frame_size_ms == 0 {
            return Err(JitterError::InvalidFrameSize);
        }
        if self.max_packets == 0 {
            return Err(JitterError::InvalidMaxPackets);
        }
        Ok(())
    }
}

/// One encoded audio frame straight off the network (RTP payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPacket {
    /// RTP sequence number. Wraps around at `u16::MAX`; the buffer extends it
    /// internally to a 64-bit counter so ordering survives the wrap.
    pub sequence_number: u16,
    /// RTP timestamp in `sample_rate` ticks. Wraps at `u32::MAX`.
    pub timestamp: u32,
    /// Encoded Opus payload bytes.
    pub payload: Vec<u8>,
}

/// Running counters and the adaptive jitter estimate.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Packets accepted into the buffer (duplicates and late arrivals excluded).
    pub packets_in: u64,
    /// Packets handed to the decoder in order.
    pub packets_out: u64,
    /// Packets discarded as duplicates, lost gaps (skipped after deadline), or
    /// capacity/latency evictions.
    pub packets_dropped: u64,
    /// Packets that arrived after their playout slot had already been skipped.
    pub packets_late: u64,
    /// Current RFC 3550 inter-arrival jitter estimate, in milliseconds.
    pub current_jitter_ms: f32,
}

/// Low-latency, adaptive jitter buffer.
///
/// Backed by a [`BTreeMap`] keyed by *extended* (wrap-safe) sequence numbers.
/// Raw `u16` sequence numbers cannot be compared directly across the
/// `u16::MAX -> 0` wrap, so each incoming number is unfolded against the
/// current sequence reference into a monotonically increasing `i64`, which
/// restores total order across stream lifetimes.
pub struct JitterBuffer {
    /// Sanitized configuration.
    config: JitterBufferConfig,
    /// Buffered packets keyed by extended sequence number (total order across u16 wrap).
    packets: BTreeMap<i64, AudioPacket>,
    /// Next extended sequence number the decoder expects.
    next_seq: i64,
    /// True once the first packet anchored `next_seq` to the stream.
    started: bool,
    /// When the current sequence gap was first noticed by `pop()`.
    gap_detected_at: Option<Instant>,
    /// RFC 3550 jitter estimate, in RTP timestamp units.
    jitter: f64,
    /// Arrival time of the previously pushed packet (for jitter deltas).
    last_arrival: Option<Instant>,
    /// RTP timestamp of the previously pushed packet.
    last_rtp_ts: Option<u32>,
    /// Injectable time source. Defaults to [`Instant::now`]; used for the gap
    /// grace window and inter-arrival jitter so timing can be mocked in tests
    /// and benchmarks.
    now: Box<dyn Fn() -> Instant + Send + Sync>,
    // -- statistics --
    packets_in: u64,
    packets_out: u64,
    packets_dropped: u64,
    packets_late: u64,
}

impl JitterBuffer {
    /// Create a jitter buffer. Invalid config values are sanitized (see
    /// [`JitterBufferConfig::validate`] for strict checking).
    pub fn new(config: JitterBufferConfig) -> Self {
        Self::with_clock(config, Box::new(Instant::now))
    }

    /// Create a jitter buffer with an injectable clock.
    ///
    /// `now` is called whenever the buffer needs the current time (gap grace
    /// windows and inter-arrival jitter). Supply a mock clock for deterministic
    /// tests and benchmarks.
    pub fn with_clock(
        config: JitterBufferConfig,
        now: Box<dyn Fn() -> Instant + Send + Sync>,
    ) -> Self {
        Self {
            config: JitterBufferConfig {
                max_latency_ms: config.max_latency_ms,
                // The target must never exceed the hard ceiling.
                target_latency_ms: config.target_latency_ms.min(config.max_latency_ms),
                max_packets: config.max_packets.max(1),
                sample_rate: config.sample_rate.max(1),
                frame_size_ms: config.frame_size_ms.max(1),
            },
            packets: BTreeMap::new(),
            next_seq: 0,
            started: false,
            gap_detected_at: None,
            jitter: 0.0,
            last_arrival: None,
            last_rtp_ts: None,
            now,
            packets_in: 0,
            packets_out: 0,
            packets_dropped: 0,
            packets_late: 0,
        }
    }

    /// Insert a packet, keeping the buffer ordered by sequence number.
    ///
    /// * Duplicates are ignored.
    /// * Packets arriving after their playout slot was skipped are counted as
    ///   late and dropped (low-latency policy: never rewind the decoder).
    /// * Inserting may evict the oldest packets to respect the latency and
    ///   capacity limits.
    pub fn push(&mut self, packet: AudioPacket) {
        self.update_jitter(&packet);

        if !self.started {
            // Anchor the extended sequence space on the first packet seen.
            self.next_seq = i64::from(packet.sequence_number);
            self.started = true;
        }

        // Unfold the raw u16 into the extended sequence space so comparisons
        // are correct even across the u16 wraparound.
        let ext = extend_seq(packet.sequence_number, self.next_seq);
        if ext < self.next_seq {
            // Its playout slot is already gone — playing it now would add
            // delay or force a decoder rewind, so drop it.
            self.packets_late += 1;
            debug!(
                seq = packet.sequence_number,
                next = self.next_seq,
                "jitter buffer: dropping late packet"
            );
            return;
        }
        if self.packets.contains_key(&ext) {
            self.packets_dropped += 1;
            debug!(
                seq = packet.sequence_number,
                "jitter buffer: duplicate packet"
            );
            return;
        }

        trace!(
            seq = packet.sequence_number,
            ts = packet.timestamp,
            "jitter buffer: push"
        );
        self.packets.insert(ext, packet);
        self.packets_in += 1;

        self.enforce_limits();
    }

    /// Pop the next in-order packet for decoding.
    ///
    /// Low-latency gap policy: if the expected packet is missing, wait at most
    /// [`JitterBufferConfig::target_latency_ms`] for it to arrive. Once the
    /// deadline passes, declare it lost, advance past the gap, and return the
    /// next available packet instead of stalling playout.
    pub fn pop(&mut self) -> Option<AudioPacket> {
        if !self.started || self.packets.is_empty() {
            return None;
        }

        // Fast path: the expected packet is buffered.
        if let Some(pkt) = self.packets.remove(&self.next_seq) {
            self.gap_detected_at = None;
            self.next_seq += 1;
            self.packets_out += 1;
            trace!(seq = pkt.sequence_number, "jitter buffer: pop");
            return Some(pkt);
        }

        // Gap: `next_seq` never arrived (yet).
        let now = (self.now)();
        let deadline = self.skip_deadline();
        let expired = self
            .gap_detected_at
            .is_some_and(|t| now.duration_since(t) >= deadline);
        if !expired {
            // Start (or continue) the grace window for the straggler.
            if self.gap_detected_at.is_none() {
                debug!(
                    missing = self.next_seq as u16,
                    "jitter buffer: gap detected"
                );
                self.gap_detected_at = Some(now);
            }
            return None;
        }

        // Deadline passed: the missing packet(s) are lost. Jump to the oldest
        // packet we actually have — never wait longer than the target latency.
        let next_key = *self.packets.keys().next().expect("non-empty checked above");
        let skipped = (next_key - self.next_seq) as u64;
        self.next_seq = next_key;
        self.gap_detected_at = None;
        self.packets_dropped += skipped;
        debug!(
            skipped,
            resumed_at = next_key as u16,
            "jitter buffer: gap deadline passed, skipping lost packets"
        );

        // Hand the resumed packet straight to the decoder.
        let pkt = self
            .packets
            .remove(&self.next_seq)
            .expect("next_key was just taken from the map");
        self.next_seq += 1;
        self.packets_out += 1;
        Some(pkt)
    }

    /// Snapshot of counters and the adaptive jitter estimate.
    pub fn stats(&self) -> JitterStats {
        JitterStats {
            packets_in: self.packets_in,
            packets_out: self.packets_out,
            packets_dropped: self.packets_dropped,
            packets_late: self.packets_late,
            current_jitter_ms: self.current_jitter_ms(),
        }
    }

    /// Reset all state for a new stream (keeps the configuration).
    pub fn clear(&mut self) {
        self.packets.clear();
        self.started = false;
        self.next_seq = 0;
        self.gap_detected_at = None;
        self.jitter = 0.0;
        self.last_arrival = None;
        self.last_rtp_ts = None;
        self.packets_in = 0;
        self.packets_out = 0;
        self.packets_dropped = 0;
        self.packets_late = 0;
        debug!("jitter buffer: cleared for new stream");
    }

    /// Grace window for a missing packet before it is declared lost.
    fn skip_deadline(&self) -> Duration {
        Duration::from_millis(u64::from(self.config.target_latency_ms))
    }

    /// Enforce the packet-count and latency ceilings after an insert.
    fn enforce_limits(&mut self) {
        // Capacity safety valve (protects against unbounded memory use).
        while self.packets.len() > self.config.max_packets {
            self.evict_oldest("capacity");
        }

        // Depth ceiling: approximate buffered audio duration as
        // (buffered frames) * (frame duration). This is the playout latency
        // the oldest buffered packet would add, which is exactly what the
        // low-latency policy wants to bound.
        let frame_ms = u64::from(self.config.frame_size_ms);
        let max_ms = u64::from(self.config.max_latency_ms);
        while self.packets.len() > 1 && self.packets.len() as u64 * frame_ms > max_ms {
            self.evict_oldest("max_latency");
        }
    }

    /// Remove the oldest buffered packet, counting it as dropped.
    fn evict_oldest(&mut self, reason: &str) {
        if let Some((key, pkt)) = self.packets.pop_first() {
            self.packets_dropped += 1;
            debug!(
                seq = pkt.sequence_number,
                reason, "jitter buffer: evicting oldest packet"
            );
            // Never leave `next_seq` pointing below the oldest surviving
            // packet — that would let a dead sequence slot block playout.
            self.next_seq = self.next_seq.max(key);
        }
    }

    /// RFC 3550 §A.1 inter-arrival jitter, in RTP timestamp units:
    ///
    /// `jitter += (|D(i-1,i)| - jitter) / 16`
    ///
    /// where `D` is the difference between the packet arrival spacing and the
    /// RTP timestamp spacing. Both deltas are computed in RTP clock units
    /// (converted with wrapping arithmetic so u32 timestamp wraparound is
    /// handled), and the first packet of a stream only initializes state.
    fn update_jitter(&mut self, packet: &AudioPacket) {
        let now = (self.now)();
        if let (Some(prev_arrival), Some(prev_ts)) = (self.last_arrival, self.last_rtp_ts) {
            // Arrival spacing expressed in RTP ticks (fractional ticks are
            // rounded; the /16 smoothing makes the error negligible).
            let arrival_delta = (now.duration_since(prev_arrival).as_secs_f64()
                * f64::from(self.config.sample_rate))
            .round() as i64;
            // Timestamp spacing, signed and wrap-safe.
            let ts_delta = i64::from(packet.timestamp.wrapping_sub(prev_ts) as i32);
            let d = arrival_delta - ts_delta;
            self.jitter += (d.abs() as f64 - self.jitter) / 16.0;
        }
        self.last_arrival = Some(now);
        self.last_rtp_ts = Some(packet.timestamp);
    }

    /// Current jitter estimate converted from RTP ticks to milliseconds.
    fn current_jitter_ms(&self) -> f32 {
        (self.jitter * 1000.0 / f64::from(self.config.sample_rate)) as f32
    }
}

/// Unfold a raw `u16` sequence number into the extended (`i64`) sequence space
/// relative to `reference`, choosing the representative closest to the
/// reference. This is what makes the `u16::MAX -> 0` wraparound compare
/// correctly: e.g. `extend_seq(0, 65_535) == 65_536`.
#[inline]
fn extend_seq(seq: u16, reference: i64) -> i64 {
    // Signed distance in [-32768, 32767] from the reference's low 16 bits.
    let diff = i64::from(seq.wrapping_sub(reference as u16) as i16);
    reference + diff
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Config with roomy latency limits so test buffers are not trimmed.
    fn cfg(max_latency_ms: u32, target_latency_ms: u32) -> JitterBufferConfig {
        JitterBufferConfig {
            max_latency_ms,
            target_latency_ms,
            ..JitterBufferConfig::default()
        }
    }

    /// Synthetic Opus packet: one payload byte encodes the sequence for easy
    /// assertions, timestamps advance by one 20 ms frame (960 ticks @ 48 kHz).
    fn pkt(seq: u16) -> AudioPacket {
        AudioPacket {
            sequence_number: seq,
            timestamp: u32::from(seq) * 960,
            payload: vec![seq as u8],
        }
    }

    /// A mock clock whose `Instant` advances only when told to, so the gap
    /// grace window can be driven deterministically without real time passing.
    #[derive(Clone)]
    struct MockClock {
        t: std::sync::Arc<std::sync::Mutex<std::time::Duration>>,
    }

    impl MockClock {
        fn new() -> Self {
            Self {
                t: std::sync::Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO)),
            }
        }
        fn advance(&self, d: std::time::Duration) {
            *self.t.lock().unwrap() += d;
        }
        fn now(&self) -> Instant {
            // `Instant::now()` + a fixed offset derived from the mock duration.
            // We only ever compare *relative* instants inside the buffer, so
            // anchoring on a real base instant is safe.
            Instant::now() + *self.t.lock().unwrap()
        }
    }

    /// A jitter buffer driven by a [`MockClock`].
    fn jb_with_clock(cfg: JitterBufferConfig, clock: &MockClock) -> JitterBuffer {
        let c = clock.clone();
        JitterBuffer::with_clock(cfg, Box::new(move || c.now()))
    }

    /// (a) In-order packets pop in order.
    #[test]
    fn in_order_packets_pop_in_order() {
        let mut jb = JitterBuffer::new(cfg(200, 40));
        for seq in 0..3u16 {
            jb.push(pkt(seq));
        }
        for seq in 0..3u16 {
            let out = jb.pop().expect("packet should be available");
            assert_eq!(out.sequence_number, seq);
            assert_eq!(out.payload, vec![seq as u8]);
        }
        assert_eq!(jb.stats().packets_out, 3);
    }

    /// (b) Out-of-order arrivals are reordered into sequence order.
    #[test]
    fn out_of_order_packets_are_reordered() {
        let mut jb = JitterBuffer::new(cfg(200, 40));
        jb.push(pkt(0));
        jb.push(pkt(2));
        jb.push(pkt(1));

        let seqs: Vec<u16> = (0..3)
            .filter_map(|_| jb.pop())
            .map(|p| p.sequence_number)
            .collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        let stats = jb.stats();
        assert_eq!(stats.packets_in, 3);
        assert_eq!(stats.packets_out, 3);
        assert_eq!(stats.packets_dropped, 0);
    }

    /// (c) Packets arriving after their slot was skipped are counted late and
    /// never played. Uses a mock clock so the grace window is deterministic.
    #[test]
    fn late_packets_beyond_max_latency_are_dropped() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(200, 1), &clock); // 1 ms grace window
        jb.push(pkt(1));
        jb.push(pkt(3)); // 2 is missing

        assert_eq!(jb.pop().unwrap().sequence_number, 1);
        assert!(jb.pop().is_none(), "should wait during grace window");

        // Let the gap deadline expire (advance mock clock past 1 ms), then pop.
        clock.advance(Duration::from_millis(10));
        assert_eq!(jb.pop().unwrap().sequence_number, 3);

        // A very late arrival of 2 must be dropped, not replayed.
        jb.push(pkt(2));
        let stats = jb.stats();
        assert_eq!(stats.packets_late, 1);
        assert_eq!(stats.packets_out, 2);
        assert_eq!(stats.packets_dropped, 1, "lost gap counted once");
    }

    /// (d) Duplicate packets are ignored.
    #[test]
    fn duplicates_are_ignored() {
        let mut jb = JitterBuffer::new(cfg(200, 1));
        jb.push(pkt(7));
        jb.push(pkt(7)); // duplicate

        let stats = jb.stats();
        assert_eq!(stats.packets_in, 1, "only one copy accepted");
        assert_eq!(stats.packets_dropped, 1, "duplicate counted as dropped");

        let first = jb.pop().unwrap();
        assert_eq!(first.sequence_number, 7);
        assert!(
            jb.pop().is_none(),
            "duplicate must not produce a second playout"
        );
    }

    /// (e) RFC 3550 adaptive jitter is computed from arrival vs timestamp deltas.
    #[test]
    fn adaptive_jitter_is_computed() {
        let mut jb = JitterBuffer::new(cfg(500, 40));
        // Burst arrival: back-to-back pushes mean arrival spacing (~0 ticks)
        // differs from timestamp spacing (960 ticks), so |D| > 0 and the
        // smoothed jitter estimate must rise above zero.
        for seq in 0..5u16 {
            jb.push(pkt(seq));
        }
        let stats = jb.stats();
        assert!(
            stats.current_jitter_ms > 0.0,
            "jitter should be > 0, got {}",
            stats.current_jitter_ms
        );
        // Converges toward |D| = 960 ticks = 20 ms; sanity-bound the estimate.
        assert!(stats.current_jitter_ms < 25.0);
    }

    /// (f) Low-latency skip: a missing packet is played through once its
    /// deadline passes instead of stalling the decoder.
    #[test]
    fn low_latency_skip_when_gap_deadline_passes() {
        let mut jb = JitterBuffer::new(cfg(200, 1)); // 1 ms grace window
        jb.push(pkt(0));
        jb.push(pkt(2)); // 1 is missing

        assert_eq!(jb.pop().unwrap().sequence_number, 0);
        assert!(jb.pop().is_none(), "grace window: not yet skipping");

        std::thread::sleep(Duration::from_millis(10));
        let out = jb.pop().expect("deadline passed, must skip and play");
        assert_eq!(out.sequence_number, 2, "skipped the lost packet 1");

        let stats = jb.stats();
        assert_eq!(stats.packets_dropped, 1, "lost packet counted as dropped");
        assert_eq!(stats.packets_out, 2);
        assert_eq!(stats.packets_late, 0);
    }

    /// (g) Sequence numbers are ordered correctly across the u16 wraparound.
    #[test]
    fn u16_wraparound_is_handled() {
        let mut jb = JitterBuffer::new(cfg(500, 1));
        for seq in [65_534u16, 65_535, 0, 1] {
            jb.push(pkt(seq));
        }
        let seqs: Vec<u16> = (0..4)
            .filter_map(|_| jb.pop())
            .map(|p| p.sequence_number)
            .collect();
        assert_eq!(seqs, vec![65_534, 65_535, 0, 1]);
    }

    /// (h) `clear()` resets stream state and statistics.
    #[test]
    fn clear_resets_for_new_stream() {
        let mut jb = JitterBuffer::new(cfg(200, 1));
        jb.push(pkt(0));
        jb.push(pkt(1));
        assert!(jb.pop().is_some());

        jb.clear();
        let stats = jb.stats();
        assert_eq!(
            (
                stats.packets_in,
                stats.packets_out,
                stats.packets_dropped,
                stats.packets_late
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(stats.current_jitter_ms, 0.0);
        assert!(jb.pop().is_none());

        // A brand-new sequence space can be anchored after clear().
        jb.push(pkt(100));
        assert_eq!(jb.pop().unwrap().sequence_number, 100);
    }

    /// (i) Config validation and sanitization behave as documented.
    #[test]
    fn config_validation_and_sanitization() {
        let bad = JitterBufferConfig {
            target_latency_ms: 100,
            max_latency_ms: 60,
            ..JitterBufferConfig::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(JitterError::InvalidLatency { .. })
        ));

        // new() sanitizes instead of panicking: target clamped to max.
        let mut jb = JitterBuffer::new(bad);
        jb.push(pkt(0));
        assert!(jb.pop().is_some());
    }
}
