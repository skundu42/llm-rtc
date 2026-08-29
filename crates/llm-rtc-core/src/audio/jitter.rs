//! Low-latency adaptive jitter buffer for voice LLM streaming.
//!
//! The jitter buffer sits between the network and the Opus decoder. It
//! reorders out-of-order RTP audio packets, absorbs network jitter, and hands
//! packets to the decoder strictly in sequence order.
//!
//! Design goals (low-latency-first):
//! * Keep the buffer shallow — startup delay is capped by
//!   [`JitterBufferConfig::max_latency_ms`].
//! * Adapt quickly — startup delay grows only enough to cover the observed
//!   positive transit-time spread, rather than multiplying an arrival-order
//!   jitter estimate that is distorted by packet reordering.
//! * Use RTP timestamps to map every frame onto a stable wall-clock playout
//!   deadline after one adaptive startup delay rooted at
//!   [`JitterBufferConfig::target_latency_ms`].
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
    /// Hard ceiling on the adaptive startup delay.
    ///
    /// This is intentionally not a packet-residence limit. A packet that
    /// arrives unusually early may wait longer than this while still having a
    /// correct, low-latency RTP deadline; discarding it would reduce quality
    /// without advancing playout.
    pub max_latency_ms: u32,
    /// Initial playout depth. The first RTP timestamp is mapped to its arrival
    /// time plus this delay; later frame deadlines follow the RTP timeline.
    pub target_latency_ms: u32,
    /// Hard ceiling on the number of buffered packets (safety valve against
    /// memory blowups during long stalls).
    pub max_packets: usize,
    /// RTP clock rate in Hz (Opus uses 48000).
    pub sample_rate: u32,
    /// Duration of one encoded frame in milliseconds.
    pub frame_size_ms: u32,
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 120,
            target_latency_ms: 5,
            max_packets: 100,
            sample_rate: 48_000,
            frame_size_ms: 10,
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
    /// capacity evictions.
    pub packets_dropped: u64,
    /// Packets that arrived after their playout slot had already been skipped.
    pub packets_late: u64,
    /// Current RFC 3550 inter-arrival jitter estimate, in milliseconds.
    pub current_jitter_ms: f32,
    /// Current adaptive startup target, in milliseconds.
    pub current_target_latency_ms: f32,
}

#[derive(Debug)]
struct BufferedPacket {
    extended_sequence_number: i64,
    packet: AudioPacket,
}

/// Low-latency, adaptive jitter buffer.
///
/// Buffered packets are kept in a small sorted [`Vec`] by *extended*
/// (wrap-safe) sequence number. Raw `u16` sequence numbers cannot be compared
/// directly across the `u16::MAX -> 0` wrap, so each incoming number is
/// unfolded against the current sequence reference into a monotonically
/// increasing `i64`, which restores total order across stream lifetimes.
pub struct JitterBuffer {
    /// Sanitized configuration.
    config: JitterBufferConfig,
    /// Buffered packets sorted by extended sequence number.
    packets: Vec<BufferedPacket>,
    /// Evicted packet slots retained in sequence order so each one still emits
    /// exactly one `Missing` event at its RTP deadline.
    evicted_slots: BTreeMap<i64, u32>,
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
    /// Startup target after adapting to the observed transit-time spread.
    current_target_latency_ms: f64,
    /// Arrival and RTP timestamp of the first packet, used to compare later
    /// packets' network transit time without requiring synchronized clocks.
    transit_anchor_arrival: Option<Instant>,
    transit_anchor_timestamp: u32,
    /// Largest observed increase in transit time relative to the first packet.
    max_positive_transit_ms: f64,
    /// RFC 3550 jitter estimate, in RTP timestamp units.
    jitter: f64,
    /// Number of transit deltas incorporated into the jitter estimate.
    jitter_samples: u32,
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
            packets: Vec::new(),
            evicted_slots: BTreeMap::new(),
            next_seq: 0,
            started: false,
            next_timestamp: 0,
            anchor_timestamp: 0,
            anchor_deadline: None,
            playout_started: false,
            current_target_latency_ms: f64::from(
                config.target_latency_ms.min(config.max_latency_ms),
            ),
            transit_anchor_arrival: None,
            transit_anchor_timestamp: 0,
            max_positive_transit_ms: 0.0,
            jitter: 0.0,
            jitter_samples: 0,
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
    /// * Inserting may evict far-future packets to respect the capacity limit.
    pub fn push(&mut self, packet: AudioPacket) -> bool {
        let now = (self.now)();

        if !self.started {
            // Anchor the extended sequence space on the first packet seen.
            self.next_seq = i64::from(packet.sequence_number);
            self.next_timestamp = packet.timestamp;
            self.anchor_timestamp = packet.timestamp;
            self.anchor_deadline = Some(now + self.target_latency());
            self.transit_anchor_arrival = Some(now);
            self.transit_anchor_timestamp = packet.timestamp;
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
        let packet_index = self.packet_index(ext);
        if packet_index.is_ok() || self.evicted_slots.contains_key(&ext) {
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
        let timestamp = packet.timestamp;
        self.update_transit_spread(timestamp, now);
        self.packets.insert(
            packet_index.expect_err("duplicate packet returned above"),
            BufferedPacket {
                extended_sequence_number: ext,
                packet,
            },
        );
        self.packets_in += 1;
        self.update_jitter(timestamp, now);
        self.adapt_startup_target();

        self.enforce_limits();
        self.packet_index(ext).is_ok()
    }

    /// Emit the next frame-slot decision once its RTP playout deadline is due.
    ///
    /// This method never emits early. If the expected packet is missing but a
    /// later packet proves that the RTP sequence has a gap, it emits exactly
    /// one loss event and advances by one frame. Empty buffers produce no loss
    /// events because RTP DTX can legitimately leave timestamp gaps without
    /// consuming sequence numbers.
    pub fn pop_event(&mut self) -> Option<PlayoutEvent> {
        let deadline = self.next_deadline()?;
        if (self.now)() < deadline {
            return None;
        }

        self.playout_started = true;

        // Fast path: the expected packet was buffered before its deadline.
        if self
            .packets
            .first()
            .is_some_and(|packet| packet.extended_sequence_number == self.next_seq)
        {
            let buffered = self.packets.remove(0);
            let pkt = buffered.packet;
            self.next_seq += 1;
            self.next_timestamp = pkt.timestamp.wrapping_add(self.frame_ticks());
            self.packets_out += 1;
            trace!(seq = pkt.sequence_number, "jitter buffer: pop");
            return Some(PlayoutEvent::Packet(pkt));
        }

        // A resource-limit eviction is still an explicit playout slot. It was
        // counted as dropped when evicted, so emit PLC without counting it a
        // second time.
        if let Some(timestamp) = self.evicted_slots.remove(&self.next_seq) {
            let missing_seq = self.next_seq;
            self.next_seq += 1;
            self.next_timestamp = timestamp.wrapping_add(self.frame_ticks());
            debug!(
                missing = missing_seq as u16,
                "jitter buffer: emitting missing event for evicted packet"
            );
            return Some(PlayoutEvent::Missing {
                sequence_number: missing_seq as u16,
                timestamp,
            });
        }

        // A later buffered packet proves this sequence slot was lost. Advance
        // one slot only so every missing frame gets one PLC/FEC opportunity.
        let missing_seq = self.next_seq;
        let missing_timestamp = self.next_timestamp;
        let next_packet_key = self
            .packets
            .first()
            .map(|packet| packet.extended_sequence_number);
        let next_evicted_key = self.evicted_slots.keys().next().copied();
        let next_key = match (next_packet_key, next_evicted_key) {
            (Some(packet), Some(evicted)) => packet.min(evicted),
            (Some(packet), None) => packet,
            (None, Some(evicted)) => evicted,
            (None, None) => unreachable!("pending state checked above"),
        };
        debug_assert!(next_key > missing_seq);
        self.next_seq += 1;
        self.next_timestamp = missing_timestamp.wrapping_add(self.frame_ticks());
        self.packets_dropped += 1;
        debug!(
            missing = missing_seq as u16,
            "jitter buffer: packet missing at playout deadline"
        );

        if next_packet_key == Some(missing_seq + 1) {
            let next_packet = self
                .packets
                .first()
                .expect("next_key was just taken from the packet store")
                .packet
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
            current_target_latency_ms: self.current_target_latency_ms as f32,
        }
    }

    /// Whether packets are still waiting for their RTP playout deadlines.
    pub fn has_pending(&self) -> bool {
        !self.packets.is_empty() || !self.evicted_slots.is_empty()
    }

    /// Wall-clock deadline of the next buffered playout slot.
    pub fn next_deadline(&self) -> Option<Instant> {
        if !self.started || !self.has_pending() {
            return None;
        }

        // An actual packet timestamp wins over the frame-size prediction. This
        // preserves deliberate RTP timestamp gaps such as Opus DTX silence.
        let timestamp = self
            .packet_index(self.next_seq)
            .ok()
            .map(|index| &self.packets[index])
            .map(|buffered| buffered.packet.timestamp)
            .or_else(|| self.evicted_slots.get(&self.next_seq).copied())
            .unwrap_or(self.next_timestamp);
        Some(self.deadline_for(timestamp))
    }

    /// Reset all state for a new stream (keeps the configuration).
    pub fn clear(&mut self) {
        self.packets.clear();
        self.evicted_slots.clear();
        self.started = false;
        self.next_seq = 0;
        self.next_timestamp = 0;
        self.anchor_timestamp = 0;
        self.anchor_deadline = None;
        self.playout_started = false;
        self.current_target_latency_ms = f64::from(self.config.target_latency_ms);
        self.transit_anchor_arrival = None;
        self.transit_anchor_timestamp = 0;
        self.max_positive_transit_ms = 0.0;
        self.jitter = 0.0;
        self.jitter_samples = 0;
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
        Duration::from_secs_f64(self.current_target_latency_ms / 1_000.0)
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

    /// Enforce the packet-count safety ceiling after an insert.
    fn enforce_limits(&mut self) {
        // Capacity safety valve (protects against unbounded memory use).
        while self.packets.len() > self.config.max_packets {
            let key = self
                .packets
                .last()
                .map(|packet| packet.extended_sequence_number)
                .expect("capacity overflow implies a packet exists");
            self.evict_packet(key, "capacity");
        }
    }

    /// Remove one buffered packet while preserving its RTP slot for PLC.
    fn evict_packet(&mut self, key: i64, reason: &str) {
        if let Ok(index) = self.packet_index(key) {
            let buffered = self.packets.remove(index);
            let packet = buffered.packet;
            self.packets_dropped += 1;
            self.evicted_slots.insert(key, packet.timestamp);
            debug!(
                seq = packet.sequence_number,
                reason, "jitter buffer: evicting packet and queuing missing event"
            );
        }
    }

    fn packet_index(&self, sequence_number: i64) -> std::result::Result<usize, usize> {
        self.packets
            .binary_search_by_key(&sequence_number, |packet| packet.extended_sequence_number)
    }

    /// Raise the startup target only enough to cover packets whose transit time
    /// exceeded the first packet's. Unlike `4 * RFC3550 jitter`, this signal is
    /// not inflated by arrival reordering and directly estimates the extra
    /// startup depth needed for the current path. Deadlines never move after
    /// playout begins; later spikes are handled by FEC/PLC.
    fn adapt_startup_target(&mut self) {
        if self.playout_started {
            return;
        }
        let desired = f64::from(self.config.target_latency_ms)
            .max(self.max_positive_transit_ms)
            .min(f64::from(self.config.max_latency_ms));
        if desired <= self.current_target_latency_ms {
            return;
        }
        let increase = desired - self.current_target_latency_ms;
        self.current_target_latency_ms = desired;
        if let Some(deadline) = self.anchor_deadline {
            self.anchor_deadline = Some(deadline + Duration::from_secs_f64(increase / 1_000.0));
        }
    }

    /// Update the relative one-way transit spread. RTP and local monotonic
    /// clocks need not share an epoch: subtracting both deltas leaves only the
    /// change in network transit time.
    fn update_transit_spread(&mut self, timestamp: u32, now: Instant) {
        let Some(anchor_arrival) = self.transit_anchor_arrival else {
            return;
        };
        let arrival_delta_ms = now.duration_since(anchor_arrival).as_secs_f64() * 1_000.0;
        let rtp_delta_ms = f64::from(timestamp.wrapping_sub(self.transit_anchor_timestamp))
            * 1_000.0
            / f64::from(self.config.sample_rate);
        self.max_positive_transit_ms = self
            .max_positive_transit_ms
            .max((arrival_delta_ms - rtp_delta_ms).max(0.0));
    }

    /// RFC 3550 §A.1 inter-arrival jitter, in RTP timestamp units:
    ///
    /// `jitter += (|D(i-1,i)| - jitter) / 16`
    ///
    /// where `D` is the difference between the packet arrival spacing and the
    /// RTP timestamp spacing. Both deltas are computed in RTP clock units
    /// (converted with wrapping arithmetic so u32 timestamp wraparound is
    /// handled), and the first packet of a stream only initializes state.
    fn update_jitter(&mut self, timestamp: u32, now: Instant) {
        if let (Some(prev_arrival), Some(prev_ts)) = (self.last_arrival, self.last_rtp_ts) {
            // Arrival spacing expressed in RTP ticks (fractional ticks are
            // rounded; the /16 smoothing makes the error negligible).
            let arrival_delta = (now.duration_since(prev_arrival).as_secs_f64()
                * f64::from(self.config.sample_rate))
            .round() as i64;
            // Timestamp spacing, signed and wrap-safe.
            let ts_delta = i64::from(timestamp.wrapping_sub(prev_ts) as i32);
            let d = arrival_delta - ts_delta;
            self.jitter += (d.abs() as f64 - self.jitter) / 16.0;
            self.jitter_samples += 1;
        }
        self.last_arrival = Some(now);
        self.last_rtp_ts = Some(timestamp);
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
            frame_size_ms: 20,
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
        let defaults = JitterBufferConfig::default();
        assert_eq!(defaults.target_latency_ms, 5);
        assert_eq!(defaults.frame_size_ms, 10);

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

    /// (l) Every capacity eviction remains an explicit missing playout slot.
    #[test]
    fn capacity_eviction_queues_one_missing_event() {
        let clock = MockClock::new();
        let mut config = cfg(200, 0);
        config.max_packets = 2;
        let mut jb = jb_with_clock(config, &clock);
        assert!(jb.push(pkt(0)));
        assert!(jb.push(pkt(1)));
        assert!(!jb.push(pkt(2)), "farthest-future packet is evicted");

        assert_eq!(jb.stats().packets_dropped, 1);
        assert_eq!(jb.pop().unwrap().sequence_number, 0);
        clock.advance(Duration::from_millis(20));
        assert_eq!(jb.pop().unwrap().sequence_number, 1);
        clock.advance(Duration::from_millis(20));
        assert!(matches!(
            jb.pop_event(),
            Some(PlayoutEvent::Missing {
                sequence_number: 2,
                ..
            })
        ));
        assert_eq!(jb.stats().packets_dropped, 1, "eviction counted once");
        assert!(!jb.has_pending());
    }

    /// (m) A valid early packet is retained. Its residence time is not added
    /// latency: the RTP timestamp still places it at the correct deadline.
    #[test]
    fn early_packet_is_not_discarded_by_latency_ceiling() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(100, 0), &clock);
        assert!(jb.push(pkt_at(0, 0)));

        // This packet arrives 200 ms before its RTP deadline. Concealing it at
        // that same deadline would buy no latency, so it must remain buffered.
        assert!(jb.push(pkt_at(1, 9_600)));
        assert_eq!(jb.stats().packets_dropped, 0);
        assert_eq!(jb.pop().unwrap().sequence_number, 0);
        clock.advance(Duration::from_millis(200));
        assert_eq!(jb.pop().unwrap().sequence_number, 1);
    }

    /// (n) Positive transit spread raises the target beyond its configured
    /// floor without being inflated by early or reordered arrivals.
    #[test]
    fn startup_target_adapts_to_jitter() {
        let clock = MockClock::new();
        let mut jb = jb_with_clock(cfg(120, 40), &clock);
        jb.push(pkt(0));
        clock.advance(Duration::from_millis(70));
        jb.push(pkt(1));
        clock.advance(Duration::from_millis(1));
        jb.push(pkt(2));

        let stats = jb.stats();
        assert!(stats.current_jitter_ms > 0.0);
        assert!(stats.current_target_latency_ms > 40.0);
        assert!(stats.current_target_latency_ms <= 120.0);
    }
}
