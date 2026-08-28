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
//! * Use RTP timestamps to map every frame onto a stable wall-clock playout
//!   deadline after one initial [`JitterBufferConfig::target_latency_ms`] delay.
//! * Prefer concealment over growing delay — a missing packet produces one
//!   [`PlayoutEvent`] at its scheduled deadline and playback keeps advancing.

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
    /// Initial playout depth. The first RTP timestamp is mapped to its arrival
    /// time plus this delay; later frame deadlines follow the RTP timeline.
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

/// One frame-slot decision made by the jitter buffer at its RTP deadline.
///
/// The next packet is cloned for FEC recovery but remains buffered for normal
/// decoding at its own deadline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlayoutEvent {
    /// The expected packet arrived before its playout deadline.
    Packet(AudioPacket),
    /// The expected packet was absent and no immediately following packet can
    /// provide in-band FEC. The decoder should perform packet-loss concealment.
    Missing {
        /// Sequence number of the lost packet.
        sequence_number: u16,
        /// Predicted RTP timestamp for the lost frame.
        timestamp: u32,
    },
    /// The expected packet was absent, but the immediately following packet is
    /// available and can be used to reconstruct it with Opus in-band FEC.
    RecoveredWithNextPacket {
        /// Sequence number of the lost packet being reconstructed.
        sequence_number: u16,
        /// Predicted RTP timestamp for the lost frame.
        timestamp: u32,
        /// Following packet containing the redundant FEC data. It remains in
        /// the jitter buffer for normal decoding at its own deadline.
        next_packet: AudioPacket,
    },
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
    /// RTP timestamp expected for `next_seq` when that packet is absent.
    next_timestamp: u32,
    /// RTP timestamp mapped onto the first playout deadline.
    anchor_timestamp: u32,
    /// Wall-clock deadline corresponding to `anchor_timestamp`.
    anchor_deadline: Option<Instant>,
    /// True after the first frame slot has been emitted. Before that point an
    /// earlier out-of-order packet may still move the stream anchor backwards.
    playout_started: bool,
    /// RFC 3550 jitter estimate, in RTP timestamp units.
    jitter: f64,
    /// Arrival time of the previously pushed packet (for jitter deltas).
    last_arrival: Option<Instant>,
    /// RTP timestamp of the previously pushed packet.
    last_rtp_ts: Option<u32>,
    /// Injectable time source. Defaults to [`Instant::now`]; used for playout
    /// deadlines and inter-arrival jitter so timing can be mocked in tests.
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
            next_timestamp: 0,
            anchor_timestamp: 0,
            anchor_deadline: None,
            playout_started: false,
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
    pub fn push(&mut self, packet: AudioPacket) -> bool {
        let now = (self.now)();

        if !self.started {
            // Anchor the extended sequence space on the first packet seen.
            self.next_seq = i64::from(packet.sequence_number);
            self.next_timestamp = packet.timestamp;
            self.anchor_timestamp = packet.timestamp;
            self.anchor_deadline = Some(now + self.target_latency());
            self.started = true;
        }

        // Unfold the raw u16 into the extended sequence space so comparisons
        // are correct even across the u16 wraparound.
        let ext = extend_seq(packet.sequence_number, self.next_seq);
        if ext < self.next_seq && self.playout_started {
            // Its playout slot is already gone — playing it now would add
            // delay or force a decoder rewind, so drop it.
            self.packets_late += 1;
            debug!(
                seq = packet.sequence_number,
                next = self.next_seq,
                "jitter buffer: dropping late packet"
            );
            return false;
        }
        if ext < self.next_seq {
            // Playout has not started, so this is an earlier out-of-order
            // packet rather than a late packet. Keep the original wall-clock
            // startup deadline but re-anchor its RTP timestamp and sequence.
            self.next_seq = ext;
            self.next_timestamp = packet.timestamp;
            self.anchor_timestamp = packet.timestamp;
        }
        if self.packets.contains_key(&ext) {
            self.packets_dropped += 1;
            debug!(
                seq = packet.sequence_number,
                "jitter buffer: duplicate packet"
            );
            return false;
        }

        trace!(
            seq = packet.sequence_number,
            ts = packet.timestamp,
            "jitter buffer: push"
        );
        self.packets.insert(ext, packet);
        self.packets_in += 1;
        self.update_jitter(ext, now);

        self.enforce_limits();
        self.packets.contains_key(&ext)
    }

    /// Emit the next frame-slot decision once its RTP playout deadline is due.
    ///
    /// This method never emits early. If the expected packet is missing but a
    /// later packet proves that the RTP sequence has a gap, it emits exactly
    /// one loss event and advances by one frame. Empty buffers produce no loss
    /// events because RTP DTX can legitimately leave timestamp gaps without
    /// consuming sequence numbers.
    pub fn pop_event(&mut self) -> Option<PlayoutEvent> {
        if !self.started || self.packets.is_empty() {
            return None;
        }

        // An actual packet timestamp wins over the frame-size prediction. This
        // preserves deliberate RTP timestamp gaps such as Opus DTX silence.
        let scheduled_timestamp = self
            .packets
            .get(&self.next_seq)
            .map_or(self.next_timestamp, |packet| packet.timestamp);
        let deadline = self.deadline_for(scheduled_timestamp);
        if (self.now)() < deadline {
            return None;
        }

        self.playout_started = true;

        // Fast path: the expected packet was buffered before its deadline.
        if let Some(pkt) = self.packets.remove(&self.next_seq) {
            self.next_seq += 1;
            self.next_timestamp = pkt.timestamp.wrapping_add(self.frame_ticks());
            self.packets_out += 1;
            trace!(seq = pkt.sequence_number, "jitter buffer: pop");
            return Some(PlayoutEvent::Packet(pkt));
        }

        // A later buffered packet proves this sequence slot was lost. Advance
        // one slot only so every missing frame gets one PLC/FEC opportunity.
        let missing_seq = self.next_seq;
        let missing_timestamp = self.next_timestamp;
        let next_key = *self.packets.keys().next().expect("non-empty checked above");
        debug_assert!(next_key > missing_seq);
        self.next_seq += 1;
        self.next_timestamp = missing_timestamp.wrapping_add(self.frame_ticks());
        self.packets_dropped += 1;
        debug!(
            missing = missing_seq as u16,
            "jitter buffer: packet missing at playout deadline"
        );

        if next_key == missing_seq + 1 {
            let next_packet = self
                .packets
                .get(&next_key)
                .expect("next_key was just taken from the map")
                .clone();
            Some(PlayoutEvent::RecoveredWithNextPacket {
                sequence_number: missing_seq as u16,
                timestamp: missing_timestamp,
                next_packet,
            })
        } else {
            Some(PlayoutEvent::Missing {
                sequence_number: missing_seq as u16,
                timestamp: missing_timestamp,
            })
        }
    }

    /// Pop only real packets, preserving the original packet-oriented API.
    ///
    /// Loss events are consumed and returned as `None`; callers that decode
    /// audio should use [`JitterBuffer::pop_event`] through [`AudioPipeline`](crate::audio::pipeline::AudioPipeline)
    /// so FEC and packet-loss concealment are applied.
    pub fn pop(&mut self) -> Option<AudioPacket> {
        match self.pop_event()? {
            PlayoutEvent::Packet(packet) => Some(packet),
            PlayoutEvent::Missing { .. } | PlayoutEvent::RecoveredWithNextPacket { .. } => None,
        }
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

    /// Whether packets are still waiting for their RTP playout deadlines.
    pub fn has_pending(&self) -> bool {
        !self.packets.is_empty()
    }

    /// Reset all state for a new stream (keeps the configuration).
    pub fn clear(&mut self) {
        self.packets.clear();
        self.started = false;
        self.next_seq = 0;
        self.next_timestamp = 0;
        self.anchor_timestamp = 0;
        self.anchor_deadline = None;
        self.playout_started = false;
        self.jitter = 0.0;
        self.last_arrival = None;
        self.last_rtp_ts = None;
        self.packets_in = 0;
        self.packets_out = 0;
        self.packets_dropped = 0;
        self.packets_late = 0;
        debug!("jitter buffer: cleared for new stream");
    }

    /// Initial wall-clock buffering delay.
    fn target_latency(&self) -> Duration {
        Duration::from_millis(u64::from(self.config.target_latency_ms))
    }

    /// RTP timestamp ticks in one configured audio frame.
    fn frame_ticks(&self) -> u32 {
        ((u64::from(self.config.sample_rate) * u64::from(self.config.frame_size_ms)) / 1000) as u32
    }

    /// Stable wall-clock deadline for an RTP timestamp, including wraparound.
    fn deadline_for(&self, timestamp: u32) -> Instant {
        let delta_ticks = timestamp.wrapping_sub(self.anchor_timestamp);
        let delta =
            Duration::from_secs_f64(f64::from(delta_ticks) / f64::from(self.config.sample_rate));
        self.anchor_deadline
            .expect("started jitter buffer has a deadline anchor")
            + delta
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
            let skipped = if key >= self.next_seq {
                (key - self.next_seq + 1) as u64
            } else {
                1
            };
            self.packets_dropped += skipped;
            debug!(
                seq = pkt.sequence_number,
                skipped, reason, "jitter buffer: evicting oldest packet"
            );
            if key >= self.next_seq {
                // Capacity pressure intentionally skips old audio to bound
                // latency. Advance every part of the playout cursor together
                // so the evicted slot is not later counted or concealed again.
                self.next_seq = key + 1;
                self.next_timestamp = pkt.timestamp.wrapping_add(self.frame_ticks());

                // Before startup, make the oldest survivor the new RTP anchor
                // while retaining the original wall-clock startup deadline.
                if !self.playout_started {
                    if let Some((&next_key, next_packet)) = self.packets.first_key_value() {
                        self.next_seq = next_key;
                        self.next_timestamp = next_packet.timestamp;
                        self.anchor_timestamp = next_packet.timestamp;
                    }
                }
            }
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
    fn update_jitter(&mut self, ext: i64, now: Instant) {
        let packet = self
            .packets
            .get(&ext)
            .expect("accepted packet was inserted before jitter update");
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

    fn pkt_at(seq: u16, timestamp: u32) -> AudioPacket {
        AudioPacket {
            sequence_number: seq,
            timestamp,
            payload: vec![seq as u8],
        }
    }

    /// A mock clock whose `Instant` advances only when told to, so the gap
    /// grace window can be driven deterministically without real time passing.
    #[derive(Clone)]
    struct MockClock {
        base: Instant,
        t: std::sync::Arc<std::sync::Mutex<std::time::Duration>>,
    }

    impl MockClock {
        fn new() -> Self {
            Self {
                base: Instant::now(),
                t: std::sync::Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO)),
            }
        }
        fn advance(&self, d: std::time::Duration) {
            *self.t.lock().unwrap() += d;
        }
        fn now(&self) -> Instant {
            self.base + *self.t.lock().unwrap()
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
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(200, 0), &clock);
        for seq in 0..3u16 {
            jb.push(pkt(seq));
        }
        for seq in 0..3u16 {
            let out = jb.pop().expect("packet should be available");
            assert_eq!(out.sequence_number, seq);
            assert_eq!(out.payload, vec![seq as u8]);
            clock.advance(Duration::from_millis(20));
        }
        assert_eq!(jb.stats().packets_out, 3);
    }

    /// (b) Out-of-order arrivals are reordered into sequence order.
    #[test]
    fn out_of_order_packets_are_reordered() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(200, 0), &clock);
        jb.push(pkt(0));
        jb.push(pkt(2));
        jb.push(pkt(1));

        let mut seqs = Vec::new();
        for _ in 0..3 {
            seqs.push(jb.pop().unwrap().sequence_number);
            clock.advance(Duration::from_millis(20));
        }
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
        let mut jb = jb_with_clock(cfg(200, 0), &clock);
        jb.push(pkt(1));
        jb.push(pkt(3)); // 2 is missing

        assert_eq!(jb.pop().unwrap().sequence_number, 1);
        assert!(jb.pop_event().is_none(), "sequence 2 is not due yet");

        clock.advance(Duration::from_millis(20));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::RecoveredWithNextPacket {
                sequence_number: 2,
                ..
            })
        ));

        // A very late arrival of 2 must be dropped, not replayed.
        assert!(!jb.push(pkt(2)));
        clock.advance(Duration::from_millis(20));
        assert_eq!(jb.pop().unwrap().sequence_number, 3);
        let stats = jb.stats();
        assert_eq!(stats.packets_late, 1);
        assert_eq!(stats.packets_out, 2);
        assert_eq!(stats.packets_dropped, 1, "lost gap counted once");
    }

    /// (d) Duplicate packets are ignored.
    #[test]
    fn duplicates_are_ignored() {
        let mut jb = JitterBuffer::new(cfg(200, 0));
        assert!(jb.push(pkt(7)));
        assert!(!jb.push(pkt(7))); // duplicate

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

    /// (f) Startup waits for the target and a missing packet emits an FEC
    /// event at its RTP deadline without growing the playout delay.
    #[test]
    fn low_latency_skip_when_gap_deadline_passes() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(200, 40), &clock);
        jb.push(pkt(0));
        jb.push(pkt(2)); // 1 is missing

        assert!(jb.pop_event().is_none(), "startup target is not due");
        clock.advance(Duration::from_millis(39));
        assert!(jb.pop_event().is_none(), "must hold the full target depth");
        clock.advance(Duration::from_millis(1));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::Packet(AudioPacket {
                sequence_number: 0,
                ..
            }))
        ));

        clock.advance(Duration::from_millis(20));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::RecoveredWithNextPacket {
                sequence_number: 1,
                next_packet: AudioPacket {
                    sequence_number: 2,
                    ..
                },
                ..
            })
        ));

        clock.advance(Duration::from_millis(20));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::Packet(AudioPacket {
                sequence_number: 2,
                ..
            }))
        ));

        let stats = jb.stats();
        assert_eq!(stats.packets_dropped, 1, "lost packet counted as dropped");
        assert_eq!(stats.packets_out, 2);
        assert_eq!(stats.packets_late, 0);
    }

    /// (g) Sequence numbers are ordered correctly across the u16 wraparound.
    #[test]
    fn u16_wraparound_is_handled() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(500, 0), &clock);
        let mut timestamp = u32::MAX - 1_500;
        for seq in [65_534u16, 65_535, 0, 1] {
            jb.push(pkt_at(seq, timestamp));
            timestamp = timestamp.wrapping_add(960);
        }
        let mut seqs = Vec::new();
        for _ in 0..4 {
            seqs.push(jb.pop().unwrap().sequence_number);
            clock.advance(Duration::from_millis(20));
        }
        assert_eq!(seqs, vec![65_534, 65_535, 0, 1]);
    }

    /// (h) `clear()` resets stream state and statistics.
    #[test]
    fn clear_resets_for_new_stream() {
        let mut jb = JitterBuffer::new(cfg(200, 0));
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
        let clock = MockClock::new();
        let mut jb = jb_with_clock(bad, &clock);
        jb.push(pkt(0));
        assert!(jb.pop().is_none());
        clock.advance(Duration::from_millis(60));
        assert!(jb.pop().is_some());
    }

    /// (j) RTP timestamp gaps delay contiguous sequence numbers without
    /// synthesizing packet loss, which is required for sender-side DTX.
    #[test]
    fn rtp_timestamp_gap_controls_deadline() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(2_000, 0), &clock);
        jb.push(pkt_at(10, 1_000));
        jb.push(pkt_at(11, 49_000)); // one second later, sequence is contiguous

        assert_eq!(jb.pop().unwrap().sequence_number, 10);
        clock.advance(Duration::from_millis(999));
        assert!(jb.pop().is_none());
        clock.advance(Duration::from_millis(1));
        assert_eq!(jb.pop().unwrap().sequence_number, 11);
        assert_eq!(jb.stats().packets_dropped, 0);
    }

    /// (k) Multiple consecutive losses emit PLC until the immediately
    /// preceding missing frame can use the next packet's FEC data.
    #[test]
    fn consecutive_losses_emit_missing_then_fec() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(500, 0), &clock);
        jb.push(pkt(0));
        jb.push(pkt(3));

        assert!(matches!(jb.pop_event(), Some(PlayoutEvent::Packet(_))));
        clock.advance(Duration::from_millis(20));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::Missing {
                sequence_number: 1,
                ..
            })
        ));
        clock.advance(Duration::from_millis(20));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::RecoveredWithNextPacket {
                sequence_number: 2,
                ..
            })
        ));
    }

    /// (l) Capacity eviction advances the complete playout cursor and does not
    /// count or conceal the same discarded slot a second time.
    #[test]
    fn startup_eviction_reanchors_without_double_counting_loss() {
        let mut jb = JitterBuffer::new(cfg(40, 0));
        jb.push(pkt(1));
        jb.push(pkt(2));
        assert!(!jb.push(pkt(0))); // new oldest frame exceeds the ceiling

        assert_eq!(jb.stats().packets_dropped, 1);
        assert_eq!(jb.pop().unwrap().sequence_number, 1);
        assert_eq!(jb.stats().packets_dropped, 1);
    }
}
