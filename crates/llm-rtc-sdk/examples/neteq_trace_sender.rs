//! Differential jitter-buffer benchmark driver.
//!
//! This example has two modes:
//! - `llm-only`: replay a deterministic RTP trace through llm-rtc.
//! - `neteq-sender`: send the exact same encoded RTP packets to a libwebrtc
//!   peer, where Chromium NetEq performs receive-side playout.

use std::cmp::Ordering;
use std::fs;
use std::io::{self, BufRead, Write};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use llm_rtc_core::audio::codec::{CodecConfig, OpusDecoder, OpusEncoder};
use llm_rtc_core::audio::jitter::{AudioPacket, JitterBuffer, JitterBufferConfig, PlayoutEvent};
use serde::Serialize;
use tokio::time::{sleep, sleep_until, timeout};
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::{MediaEngine, MIME_TYPE_OPUS};
use webrtc::api::APIBuilder;
use webrtc::ice_transport::ice_gathering_state::RTCIceGatheringState;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp::header::Header;
use webrtc::rtp::packet::Packet;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;
use webrtc::track::track_local::TrackLocalWriter;

const SAMPLE_RATE: u32 = 48_000;
const FRAME_MS: u64 = 10;
const PLAYOUT_POLL_MS: u64 = FRAME_MS / 2;
const SAMPLES_PER_FRAME: usize = 480;
const CONTENT_FRAMES: usize = 1_000;
const GUARD_FRAMES: usize = 10;
const START_SEQUENCE: u16 = 10_000;
const START_TIMESTAMP: u32 = 900_000;
const DEFAULT_TARGET_LATENCY_MS: u32 = 5;

#[derive(Clone, Copy)]
struct NetworkProfile {
    name: &'static str,
    base_delay_ms: f64,
    jitter_ms: f64,
    loss_rate: f64,
    seed: u32,
}

impl NetworkProfile {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "clean" => Ok(Self {
                name: "clean",
                base_delay_ms: 20.0,
                jitter_ms: 2.0,
                loss_rate: 0.0,
                seed: 0x1357_2468,
            }),
            "moderate" => Ok(Self {
                name: "moderate",
                base_delay_ms: 30.0,
                jitter_ms: 15.0,
                loss_rate: 0.05,
                seed: 0x1234_5678,
            }),
            "severe" => Ok(Self {
                name: "severe",
                base_delay_ms: 50.0,
                jitter_ms: 40.0,
                loss_rate: 0.10,
                seed: 0xa5a5_5a5a,
            }),
            other => bail!("unknown profile {other:?}; expected clean, moderate, or severe"),
        }
    }
}

#[derive(Clone)]
struct TracePacket {
    frame_index: usize,
    sequence_number: u16,
    timestamp: u32,
    arrival_ms: f64,
    dropped: bool,
    payload: Vec<u8>,
}

#[derive(Serialize)]
struct TraceRow {
    frame_index: usize,
    sequence_number: u16,
    timestamp: u32,
    source_time_ms: u64,
    arrival_ms: f64,
    dropped: bool,
    payload_bytes: usize,
}

#[derive(Serialize)]
struct LocalMetrics {
    engine: &'static str,
    profile: String,
    max_latency_ms: u32,
    target_latency_ms: u32,
    content_frames: usize,
    network_packets_lost: usize,
    continuity_pct: f64,
    normal_frames: usize,
    fec_frames: usize,
    fec_rate_pct: f64,
    plc_frames: usize,
    plc_rate_pct: f64,
    concealment_rate_pct: f64,
    late_drops: u64,
    adaptive_target_latency_ms: f32,
    p50_playout_delay_ms: f64,
    p95_playout_delay_ms: f64,
    p99_playout_delay_ms: f64,
    first_content_playout_ms: f64,
    last_content_playout_end_ms: f64,
    content_playout_timeline: Vec<PlayoutPoint>,
    median_cpu_ms_per_audio_second: f64,
    peak_rss_mib: f64,
}

#[derive(Clone, Serialize)]
struct PlayoutPoint {
    sample_offset: usize,
    sample_count: usize,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct LoadMetrics {
    engine: &'static str,
    profile: String,
    concurrent_calls: usize,
    repetitions: usize,
    max_latency_ms: u32,
    target_latency_ms: u32,
    median_cpu_ms_per_audio_second_per_call: f64,
    median_batch_wall_ms: f64,
    peak_rss_mib: f64,
}

struct LocalRun {
    pcm: Vec<i16>,
    normal_frames: usize,
    fec_frames: usize,
    plc_frames: usize,
    output_frames: usize,
    delays_ms: Vec<f64>,
    late_drops: u64,
    adaptive_target_latency_ms: f32,
    first_content_playout_ms: f64,
    last_content_playout_end_ms: f64,
    content_playout_timeline: Vec<PlayoutPoint>,
}

fn codec_config() -> CodecConfig {
    CodecConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        bitrate: 24_000,
        frame_size_ms: FRAME_MS as f32,
        use_dtx: false,
        use_fec: true,
        complexity: 0,
    }
}

fn read_pcm(path: &Path) -> Result<Vec<i16>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() % 2 != 0 {
        bail!("PCM input has an odd byte count");
    }
    let mut pcm = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([pair[0], pair[1]]));
    }
    let required = CONTENT_FRAMES * SAMPLES_PER_FRAME;
    if pcm.len() < required {
        bail!(
            "PCM input contains {} samples; benchmark needs at least {required}",
            pcm.len()
        );
    }
    pcm.truncate(required);
    Ok(pcm)
}

fn write_pcm(path: &Path, pcm: &[i16]) -> Result<()> {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for sample in pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn encode_trace(pcm: &[i16], profile: NetworkProfile) -> Result<Vec<TracePacket>> {
    let encoder = OpusEncoder::new(codec_config())?;
    let mut encoded = Vec::with_capacity(CONTENT_FRAMES + GUARD_FRAMES);
    for frame in pcm.chunks_exact(SAMPLES_PER_FRAME) {
        encoded.push(encoder.encode(frame)?);
    }
    let guard = vec![0_i16; SAMPLES_PER_FRAME];
    for _ in 0..GUARD_FRAMES {
        encoded.push(encoder.encode(&guard)?);
    }

    let mut random_state = profile.seed;
    let mut random = || {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        f64::from(random_state) / f64::from(u32::MAX)
    };

    let mut trace = Vec::with_capacity(encoded.len());
    for (frame_index, payload) in encoded.into_iter().enumerate() {
        let is_guard = frame_index >= CONTENT_FRAMES;
        let jitter = if is_guard {
            0.0
        } else {
            (random() * 2.0 - 1.0) * profile.jitter_ms
        };
        // Keep the first packet so both engines anchor on the same RTP slot.
        let dropped = !is_guard && frame_index != 0 && random() < profile.loss_rate;
        let source_time_ms = frame_index as f64 * FRAME_MS as f64;
        let arrival_ms = (source_time_ms + profile.base_delay_ms + jitter).max(0.0);
        trace.push(TracePacket {
            frame_index,
            sequence_number: START_SEQUENCE.wrapping_add(frame_index as u16),
            timestamp: START_TIMESTAMP.wrapping_add((frame_index * SAMPLES_PER_FRAME) as u32),
            arrival_ms,
            dropped,
            payload,
        });
    }
    Ok(trace)
}

fn write_trace(
    path: &Path,
    profile: NetworkProfile,
    trace: &[TracePacket],
    max_latency_ms: u32,
    target_latency_ms: u32,
) -> Result<()> {
    let rows: Vec<_> = trace
        .iter()
        .map(|packet| TraceRow {
            frame_index: packet.frame_index,
            sequence_number: packet.sequence_number,
            timestamp: packet.timestamp,
            source_time_ms: packet.frame_index as u64 * FRAME_MS,
            arrival_ms: packet.arrival_ms,
            dropped: packet.dropped,
            payload_bytes: packet.payload.len(),
        })
        .collect();
    let document = serde_json::json!({
        "profile": profile.name,
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "target_latency_ms": target_latency_ms,
        "max_latency_ms": max_latency_ms,
        "packets": rows,
    });
    fs::write(path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn run_local_once(
    trace: &[TracePacket],
    max_latency_ms: u32,
    target_latency_ms: u32,
    capture_pcm: bool,
) -> Result<LocalRun> {
    let base = Instant::now();
    let clock_offset = Arc::new(Mutex::new(Duration::ZERO));
    let clock = {
        let clock_offset = Arc::clone(&clock_offset);
        Box::new(move || base + *clock_offset.lock().expect("benchmark clock poisoned"))
    };
    let config = JitterBufferConfig {
        max_latency_ms,
        target_latency_ms,
        max_packets: 100,
        sample_rate: SAMPLE_RATE,
        frame_size_ms: FRAME_MS as u32,
    };
    let mut jitter = JitterBuffer::with_clock(config, clock);
    let mut decoder = OpusDecoder::new(codec_config())?;
    let mut arrivals: Vec<_> = trace.iter().filter(|packet| !packet.dropped).collect();
    arrivals.sort_by(|a, b| {
        a.arrival_ms
            .partial_cmp(&b.arrival_ms)
            .unwrap_or(Ordering::Equal)
            .then(a.frame_index.cmp(&b.frame_index))
    });

    let mut next_arrival = 0;
    let mut output = if capture_pcm {
        Vec::with_capacity(CONTENT_FRAMES * SAMPLES_PER_FRAME)
    } else {
        Vec::new()
    };
    let mut normal_frames = 0;
    let mut fec_frames = 0;
    let mut plc_frames = 0;
    let mut output_frames = 0;
    let mut delays_ms = Vec::new();
    let mut first_content_playout_ms = None;
    let mut last_content_playout_end_ms = 0.0;
    let mut content_playout_timeline = Vec::with_capacity(CONTENT_FRAMES);
    let end_ms = trace
        .iter()
        .map(|packet| packet.arrival_ms)
        .fold(0.0, f64::max)
        + 1_000.0;

    let mut tick_ms = 0.0;
    while tick_ms <= end_ms && output_frames < CONTENT_FRAMES {
        while next_arrival < arrivals.len() && arrivals[next_arrival].arrival_ms <= tick_ms {
            let packet = arrivals[next_arrival];
            *clock_offset.lock().expect("benchmark clock poisoned") =
                Duration::from_secs_f64(packet.arrival_ms / 1_000.0);
            jitter.push(AudioPacket {
                sequence_number: packet.sequence_number,
                timestamp: packet.timestamp,
                payload: packet.payload.clone(),
            });
            next_arrival += 1;
        }
        *clock_offset.lock().expect("benchmark clock poisoned") =
            Duration::from_secs_f64(tick_ms / 1_000.0);

        // Match the SDK's controlled recovery clock: emit at most one frame
        // per half-frame poll, even when several RTP deadlines are overdue.
        if let Some(event) = jitter.pop_event() {
            let (sequence_number, pcm, kind) = match event {
                PlayoutEvent::Packet(packet) => {
                    let decoded = decoder.decode(&packet.payload)?;
                    delays_ms.push(
                        tick_ms
                            - trace
                                [usize::from(packet.sequence_number.wrapping_sub(START_SEQUENCE))]
                            .arrival_ms,
                    );
                    (packet.sequence_number, decoded, 0_u8)
                }
                PlayoutEvent::Missing {
                    sequence_number, ..
                } => (sequence_number, decoder.decode(&[])?, 2_u8),
                PlayoutEvent::RecoveredWithNextPacket {
                    sequence_number,
                    next_packet,
                    ..
                } => (
                    sequence_number,
                    decoder.decode_fec(&next_packet.payload)?,
                    1_u8,
                ),
            };
            let frame_index = usize::from(sequence_number.wrapping_sub(START_SEQUENCE));
            if frame_index < CONTENT_FRAMES {
                if capture_pcm {
                    output.extend_from_slice(&pcm[..pcm.len().min(SAMPLES_PER_FRAME)]);
                }
                output_frames += 1;
                first_content_playout_ms.get_or_insert(tick_ms);
                // The full decoded frame is available to ASR at callback time.
                last_content_playout_end_ms = tick_ms;
                content_playout_timeline.push(PlayoutPoint {
                    sample_offset: frame_index * SAMPLES_PER_FRAME,
                    sample_count: SAMPLES_PER_FRAME,
                    elapsed_ms: tick_ms,
                });
                match kind {
                    0 => normal_frames += 1,
                    1 => fec_frames += 1,
                    _ => plc_frames += 1,
                }
            }
        }
        tick_ms += PLAYOUT_POLL_MS as f64;
    }

    let stats = jitter.stats();
    Ok(LocalRun {
        pcm: output,
        normal_frames,
        fec_frames,
        plc_frames,
        output_frames,
        delays_ms,
        late_drops: stats.packets_late,
        adaptive_target_latency_ms: stats.current_target_latency_ms,
        first_content_playout_ms: first_content_playout_ms.unwrap_or(0.0),
        last_content_playout_end_ms,
        content_playout_timeline,
    })
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn peak_rss_mib() -> f64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<f64>().ok())
        .map_or(0.0, |kib| kib / 1_024.0)
}

fn process_cpu_time() -> Result<Duration> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` initializes the pointed-to `rusage` on success, and
    // the pointer is valid for the duration of the call.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error()).context("reading process CPU usage");
    }
    // SAFETY: the successful `getrusage` call above initialized `usage`.
    let usage = unsafe { usage.assume_init() };
    let timeval = |value: libc::timeval| -> Result<Duration> {
        let seconds = u64::try_from(value.tv_sec).context("negative CPU seconds")?;
        let micros = u32::try_from(value.tv_usec).context("invalid CPU microseconds")?;
        Ok(Duration::new(seconds, micros * 1_000))
    };
    Ok(timeval(usage.ru_utime)? + timeval(usage.ru_stime)?)
}

fn run_llm_only(
    profile: NetworkProfile,
    input: &Path,
    output_dir: &Path,
    max_latency_ms: u32,
    target_latency_ms: u32,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let pcm = read_pcm(input)?;
    let trace = encode_trace(&pcm, profile)?;
    write_trace(
        &output_dir.join("trace.json"),
        profile,
        &trace,
        max_latency_ms,
        target_latency_ms,
    )?;

    let quality_run = run_local_once(&trace, max_latency_ms, target_latency_ms, true)?;
    write_pcm(&output_dir.join("llm-rtc.pcm"), &quality_run.pcm)?;

    let mut cpu_samples = Vec::new();
    for _ in 0..21 {
        let started = process_cpu_time()?;
        let run = run_local_once(&trace, max_latency_ms, target_latency_ms, false)?;
        std::hint::black_box(run.output_frames);
        let elapsed = process_cpu_time()?
            .checked_sub(started)
            .context("process CPU clock moved backwards")?;
        cpu_samples.push(elapsed.as_secs_f64() * 1_000.0);
    }
    cpu_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_cpu_ms = cpu_samples[cpu_samples.len() / 2];
    let network_packets_lost = trace
        .iter()
        .filter(|packet| packet.frame_index < CONTENT_FRAMES && packet.dropped)
        .count();
    let metrics = LocalMetrics {
        engine: "llm-rtc",
        profile: profile.name.to_string(),
        max_latency_ms,
        target_latency_ms,
        content_frames: CONTENT_FRAMES,
        network_packets_lost,
        continuity_pct: quality_run.output_frames as f64 / CONTENT_FRAMES as f64 * 100.0,
        normal_frames: quality_run.normal_frames,
        fec_frames: quality_run.fec_frames,
        fec_rate_pct: quality_run.fec_frames as f64 / CONTENT_FRAMES as f64 * 100.0,
        plc_frames: quality_run.plc_frames,
        plc_rate_pct: quality_run.plc_frames as f64 / CONTENT_FRAMES as f64 * 100.0,
        concealment_rate_pct: quality_run.plc_frames as f64 / CONTENT_FRAMES as f64 * 100.0,
        late_drops: quality_run.late_drops,
        adaptive_target_latency_ms: quality_run.adaptive_target_latency_ms,
        p50_playout_delay_ms: percentile(&quality_run.delays_ms, 0.50),
        p95_playout_delay_ms: percentile(&quality_run.delays_ms, 0.95),
        p99_playout_delay_ms: percentile(&quality_run.delays_ms, 0.99),
        first_content_playout_ms: quality_run.first_content_playout_ms,
        last_content_playout_end_ms: quality_run.last_content_playout_end_ms,
        content_playout_timeline: quality_run.content_playout_timeline,
        median_cpu_ms_per_audio_second: median_cpu_ms / 10.0,
        peak_rss_mib: peak_rss_mib(),
    };
    fs::write(
        output_dir.join("llm-rtc.json"),
        serde_json::to_vec_pretty(&metrics)?,
    )?;
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values[values.len() / 2]
}

fn run_llm_load(
    profile: NetworkProfile,
    input: &Path,
    output_dir: &Path,
    max_latency_ms: u32,
    target_latency_ms: u32,
    concurrent_calls: usize,
    repetitions: usize,
) -> Result<()> {
    if concurrent_calls == 0 || repetitions == 0 {
        bail!("concurrent-calls and repetitions must be positive");
    }
    fs::create_dir_all(output_dir)?;
    let pcm = read_pcm(input)?;
    let trace = Arc::new(encode_trace(&pcm, profile)?);
    let mut cpu_samples = Vec::with_capacity(repetitions);
    let mut wall_samples = Vec::with_capacity(repetitions);

    for _ in 0..repetitions {
        let cpu_started = process_cpu_time()?;
        let wall_started = Instant::now();
        let mut workers = Vec::with_capacity(concurrent_calls);
        for _ in 0..concurrent_calls {
            let trace = Arc::clone(&trace);
            workers.push(thread::spawn(move || {
                run_local_once(trace.as_slice(), max_latency_ms, target_latency_ms, false)
            }));
        }
        for worker in workers {
            let run = worker
                .join()
                .map_err(|_| anyhow::anyhow!("load worker panicked"))??;
            if run.output_frames != CONTENT_FRAMES {
                bail!("load worker produced an incomplete stream");
            }
        }
        let cpu_elapsed = process_cpu_time()?
            .checked_sub(cpu_started)
            .context("process CPU clock moved backwards")?;
        cpu_samples.push(
            cpu_elapsed.as_secs_f64() * 1_000.0
                / (concurrent_calls as f64 * CONTENT_FRAMES as f64 * FRAME_MS as f64 / 1_000.0),
        );
        wall_samples.push(wall_started.elapsed().as_secs_f64() * 1_000.0);
    }

    let metrics = LoadMetrics {
        engine: "llm-rtc",
        profile: profile.name.to_string(),
        concurrent_calls,
        repetitions,
        max_latency_ms,
        target_latency_ms,
        median_cpu_ms_per_audio_second_per_call: median(&mut cpu_samples),
        median_batch_wall_ms: median(&mut wall_samples),
        peak_rss_mib: peak_rss_mib(),
    };
    fs::write(
        output_dir.join("llm-rtc-load.json"),
        serde_json::to_vec_pretty(&metrics)?,
    )?;
    Ok(())
}

async fn wait_for_ice_complete(pc: &webrtc::peer_connection::RTCPeerConnection) {
    for _ in 0..200 {
        if pc.ice_gathering_state() == RTCIceGatheringState::Complete {
            return;
        }
        sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_connected(pc: &webrtc::peer_connection::RTCPeerConnection) -> Result<()> {
    timeout(Duration::from_secs(15), async {
        loop {
            match pc.connection_state() {
                RTCPeerConnectionState::Connected => return Ok(()),
                RTCPeerConnectionState::Failed | RTCPeerConnectionState::Closed => {
                    bail!("peer connection failed")
                }
                _ => sleep(Duration::from_millis(25)).await,
            }
        }
    })
    .await
    .context("timed out waiting for libwebrtc receiver")??;
    Ok(())
}

fn protocol_line() -> Result<String> {
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

fn emit_protocol(message: &str) -> Result<()> {
    println!("{message}");
    io::stdout().flush()?;
    Ok(())
}

async fn run_neteq_sender(profile: NetworkProfile, input: &Path, output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let pcm = read_pcm(input)?;
    let trace = encode_trace(&pcm, profile)?;
    write_trace(
        &output_dir.join("trace.json"),
        profile,
        &trace,
        120,
        DEFAULT_TARGET_LATENCY_MS,
    )?;

    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;
    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;
    let api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .build();
    let pc = api.new_peer_connection(RTCConfiguration::default()).await?;
    let track = Arc::new(TrackLocalStaticRTP::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_string(),
            clock_rate: SAMPLE_RATE,
            channels: 2,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".to_string(),
            ..Default::default()
        },
        "audio".to_string(),
        "neteq-comparison".to_string(),
    ));
    let sender = pc.add_track(track.clone()).await?;
    tokio::spawn(async move {
        let mut rtcp = vec![0_u8; 1_500];
        while sender.read(&mut rtcp).await.is_ok() {}
    });

    let offer = pc.create_offer(None).await?;
    pc.set_local_description(offer).await?;
    wait_for_ice_complete(&pc).await;
    let local = pc
        .local_description()
        .await
        .context("missing gathered local description")?;
    fs::write(output_dir.join("offer.sdp"), local.sdp)?;
    emit_protocol("OFFER_READY")?;
    if protocol_line()? != "ANSWER_READY" {
        bail!("receiver did not provide an SDP answer");
    }
    let answer = fs::read_to_string(output_dir.join("answer.sdp"))?;
    pc.set_remote_description(RTCSessionDescription::answer(answer)?)
        .await?;
    wait_for_connected(&pc).await?;
    emit_protocol("READY")?;
    if protocol_line()? != "GO" {
        bail!("receiver did not start the trace");
    }

    let mut arrivals: Vec<_> = trace.iter().filter(|packet| !packet.dropped).collect();
    arrivals.sort_by(|a, b| {
        a.arrival_ms
            .partial_cmp(&b.arrival_ms)
            .unwrap_or(Ordering::Equal)
            .then(a.frame_index.cmp(&b.frame_index))
    });
    let start = tokio::time::Instant::now();
    for packet in arrivals {
        sleep_until(start + Duration::from_secs_f64(packet.arrival_ms / 1_000.0)).await;
        track
            .write_rtp(&Packet {
                header: Header {
                    version: 2,
                    sequence_number: packet.sequence_number,
                    timestamp: packet.timestamp,
                    ssrc: 0x1122_3344,
                    payload_type: 111,
                    ..Default::default()
                },
                payload: Bytes::copy_from_slice(&packet.payload),
            })
            .await?;
    }
    emit_protocol("SENT")?;
    if protocol_line()? != "STOP" {
        bail!("receiver did not finish cleanly");
    }
    pc.close().await?;
    Ok(())
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  neteq_trace_sender llm-only <profile> <input.pcm> <output-dir> [max-latency-ms] [target-latency-ms]\n  neteq_trace_sender neteq-sender <profile> <input.pcm> <output-dir>\n  neteq_trace_sender llm-load <profile> <input.pcm> <output-dir> <max-latency-ms> <target-latency-ms> <concurrent-calls> <repetitions>"
    );
    std::process::exit(2);
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        usage();
    }
    let profile = NetworkProfile::parse(&args[2])?;
    let input = PathBuf::from(&args[3]);
    let output_dir = PathBuf::from(&args[4]);
    match args[1].as_str() {
        "llm-only" if (5..=7).contains(&args.len()) => {
            let max_latency_ms = args
                .get(5)
                .map_or(Ok(120), |value| value.parse::<u32>())
                .context("max-latency-ms must be an integer")?;
            let target_latency_ms = args
                .get(6)
                .map_or(Ok(DEFAULT_TARGET_LATENCY_MS), |value| value.parse::<u32>())
                .context("target-latency-ms must be an integer")?;
            run_llm_only(
                profile,
                &input,
                &output_dir,
                max_latency_ms,
                target_latency_ms,
            )
        }
        "neteq-sender" if args.len() == 5 => run_neteq_sender(profile, &input, &output_dir).await,
        "llm-load" if args.len() == 9 => run_llm_load(
            profile,
            &input,
            &output_dir,
            args[5]
                .parse()
                .context("max-latency-ms must be an integer")?,
            args[6]
                .parse()
                .context("target-latency-ms must be an integer")?,
            args[7]
                .parse()
                .context("concurrent-calls must be an integer")?,
            args[8].parse().context("repetitions must be an integer")?,
        ),
        _ => usage(),
    }
}
