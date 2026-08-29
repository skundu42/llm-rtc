#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "faster-whisper==1.2.1",
#   "numpy==2.2.0",
#   "pesq==0.0.4",
#   "scipy==1.14.1",
#   "webrtcvad-wheels==2.0.14",
# ]
# ///

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from pesq import pesq
from scipy.signal import correlate, correlation_lags, resample_poly

SAMPLE_RATE = 48_000
FRAME_SAMPLES = 480
CONTENT_MS = 10_000.0
TRANSCRIPT = (
    "Real time voice agents need steady timing. "
    "The receiver must absorb network jitter. "
    "Error correction should preserve clear and continuous conversation."
)


def read_pcm(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<i2")


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_words(reference)
    actual = normalize_words(hypothesis)
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_word != actual_word),
            ))
        previous = current
    return previous[-1] / max(1, len(expected)) * 100.0


def align_lag_samples(reference: np.ndarray, degraded: np.ndarray) -> int:
    factor = 8
    reference_small = reference[::factor].astype(np.float64)
    degraded_small = degraded[::factor].astype(np.float64)
    correlation = correlate(degraded_small, reference_small, mode="full", method="fft")
    lags = correlation_lags(len(degraded_small), len(reference_small), mode="full")
    max_lag = int(2 * SAMPLE_RATE / factor)
    valid = np.abs(lags) <= max_lag
    return int(lags[valid][int(np.argmax(correlation[valid]))] * factor)


def aligned_audio(reference: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, int]:
    lag = align_lag_samples(reference, degraded)
    if lag >= 0:
        aligned = degraded[lag:lag + len(reference)]
    else:
        aligned = np.pad(degraded, (-lag, 0))[:len(reference)]
    if len(aligned) < len(reference):
        aligned = np.pad(aligned, (0, len(reference) - len(aligned)))
    return aligned, lag


def pesq_mos(reference: np.ndarray, degraded: np.ndarray) -> tuple[float, float]:
    aligned, lag = aligned_audio(reference, degraded)
    reference_16k = resample_poly(reference.astype(np.float32) / 32768.0, 1, 3)
    degraded_16k = resample_poly(aligned.astype(np.float32) / 32768.0, 1, 3)
    score = pesq(16_000, reference_16k, degraded_16k, "wb")
    return float(score), lag * 1_000.0 / SAMPLE_RATE


def transcribe(model: WhisperModel, pcm_path: Path, repetitions: int = 5) -> dict:
    transcript = ""
    elapsed_samples = []
    audio_16k = resample_poly(
        read_pcm(pcm_path).astype(np.float32) / 32768.0,
        1,
        3,
    ).astype(np.float32)
    for _ in range(repetitions):
        started = time.perf_counter()
        segments, _ = model.transcribe(
            audio_16k,
            beam_size=5,
            condition_on_previous_text=False,
            language="en",
            temperature=0,
            vad_filter=False,
        )
        candidate = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed_samples.append((time.perf_counter() - started) * 1_000.0)
        if not transcript:
            transcript = candidate
    return {
        "transcript": transcript,
        "wer_pct": word_error_rate(TRANSCRIPT, transcript),
        "median_asr_wall_ms": median(elapsed_samples),
        "asr_wall_runs_ms": elapsed_samples,
    }


def reference_barge_events(reference: np.ndarray) -> list[dict]:
    frames = reference[:len(reference) // FRAME_SAMPLES * FRAME_SAMPLES]
    frames = frames.reshape(-1, FRAME_SAMPLES).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1.0)
    dbfs = 20 * np.log10(rms / 32768.0)
    threshold = min(-34.0, float(np.max(dbfs) - 24.0))
    active = dbfs >= threshold
    indices = np.flatnonzero(active)
    groups = []
    if len(indices):
        start = previous = int(indices[0])
        for index in map(int, indices[1:]):
            if index - previous > 20:
                groups.append((start, previous + 1))
                start = index
            previous = index
        groups.append((start, previous + 1))
    events = [
        {"onset_ms": start * 10.0, "end_ms": end * 10.0}
        for start, end in groups
        if end - start >= 5
    ]
    if len(events) != 5:
        raise RuntimeError(f"expected five barge-in phrases, detected {len(events)} in reference")
    return events


def vad_frames(pcm: np.ndarray) -> list[bool]:
    vad = webrtcvad.Vad(2)
    complete = pcm[:len(pcm) // FRAME_SAMPLES * FRAME_SAMPLES]
    return [
        vad.is_speech(frame.astype("<i2", copy=False).tobytes(), SAMPLE_RATE)
        for frame in complete.reshape(-1, FRAME_SAMPLES)
    ]


def barge_metrics(
    reference: np.ndarray,
    degraded: np.ndarray,
    events: list[dict],
    stream_start_ms: float,
    callback_timeline: list[dict] | None = None,
) -> dict:
    lag = align_lag_samples(reference, degraded)
    lag_ms = lag * 1_000.0 / SAMPLE_RATE
    decisions = vad_frames(degraded)
    delays = []
    missed = 0
    event_details = []
    for event in events:
        expected_degraded_ms = event["onset_ms"] + lag_ms
        first_frame = max(0, int((expected_degraded_ms - 50.0) // 10.0))
        last_frame = min(len(decisions), int((expected_degraded_ms + 300.0) // 10.0) + 1)
        detected_frame = next(
            (index for index in range(first_frame, last_frame) if decisions[index]),
            None,
        )
        if detected_frame is None:
            missed += 1
            event_details.append({**event, "detected": False})
            continue
        detection_sample = detected_frame * FRAME_SAMPLES
        detection_ms = timeline_callback_ms(callback_timeline, detection_sample)
        if detection_ms is None:
            detection_ms = stream_start_ms + detected_frame * 10.0
        delay_ms = detection_ms - event["onset_ms"]
        delays.append(delay_ms)
        event_details.append({
            **event,
            "detected": True,
            "source_to_detection_ms": delay_ms,
        })
    return {
        "events": len(events),
        "missed": missed,
        "missed_rate_pct": missed / len(events) * 100.0,
        "median_source_to_detection_ms": median(delays) if delays else None,
        "p95_source_to_detection_ms": float(np.percentile(delays, 95)) if delays else None,
        "alignment_lag_ms": lag_ms,
        "details": event_details,
    }


def timeline_callback_ms(timeline: list[dict] | None, sample_offset: int) -> float | None:
    if not timeline:
        return None
    for point in timeline:
        start = point["sample_offset"]
        if start <= sample_offset < start + point["sample_count"]:
            return float(point["elapsed_ms"])
    return None


def median_run_metrics(rows: list[dict], fields: list[str]) -> dict:
    result = {field: median(row[field] for row in rows) for field in fields}
    result["runs"] = rows
    return result


def write_load_csv(results_dir: Path, rows: list[dict]) -> None:
    fields = [
        "engine",
        "concurrent_calls",
        "median_cpu_ms_per_audio_second_per_call",
        "peak_rss_mib",
        "median_batch_wall_ms",
    ]
    with (results_dir / "application-load.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def evaluate_tuning(results_dir: Path) -> None:
    reference = read_pcm(results_dir / "speech-reference.pcm")
    barge_reference = read_pcm(results_dir / "barge-reference.pcm")
    events = reference_barge_events(barge_reference)
    model_name = os.environ.get("LLM_RTC_ASR_MODEL", "Systran/faster-whisper-base.en")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    transcribe(model, results_dir / "speech-reference.pcm", repetitions=1)
    rows = []
    for target_dir in sorted(
        (results_dir / "tuning").glob("target-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    ):
        speech_dir = target_dir / "speech"
        barge_dir = target_dir / "barge"
        metrics = read_json(speech_dir / "llm-rtc.json")
        barge_transport = read_json(barge_dir / "llm-rtc.json")
        asr = transcribe(model, speech_dir / "llm-rtc.pcm", repetitions=3)
        mos, _ = pesq_mos(reference, read_pcm(speech_dir / "llm-rtc.pcm"))
        barge = barge_metrics(
            barge_reference,
            read_pcm(barge_dir / "llm-rtc.pcm"),
            events,
            barge_transport["first_content_playout_ms"],
            barge_transport["content_playout_timeline"],
        )
        rows.append({
            "target_latency_ms": metrics["target_latency_ms"],
            "wer_pct": asr["wer_pct"],
            "pesq_mos_lqo": mos,
            "barge_p95_ms": barge["p95_source_to_detection_ms"],
            "ingress_tail_ms": metrics["last_content_playout_end_ms"] - CONTENT_MS,
            "normal_frames": metrics["normal_frames"],
            "fec_frames": metrics["fec_frames"],
            "plc_frames": metrics["plc_frames"],
            "late_drops": metrics["late_drops"],
            "p95_playout_delay_ms": metrics["p95_playout_delay_ms"],
            "p99_playout_delay_ms": metrics["p99_playout_delay_ms"],
            "cpu_ms_per_audio_second": metrics["median_cpu_ms_per_audio_second"],
        })
    (results_dir / "tuning" / "tuning-summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(
        "target_ms,wer_pct,pesq_mos_lqo,barge_p95_ms,ingress_tail_ms,"
        "normal,fec,plc,late,p95_ms,p99_ms,cpu_ms_per_audio_s"
    )
    for row in rows:
        print(",".join([
            str(row["target_latency_ms"]),
            f'{row["wer_pct"]:.2f}',
            f'{row["pesq_mos_lqo"]:.3f}',
            f'{row["barge_p95_ms"]:.2f}',
            f'{row["ingress_tail_ms"]:.2f}',
            str(row["normal_frames"]),
            str(row["fec_frames"]),
            str(row["plc_frames"]),
            str(row["late_drops"]),
            f'{row["p95_playout_delay_ms"]:.2f}',
            f'{row["p99_playout_delay_ms"]:.2f}',
            f'{row["cpu_ms_per_audio_second"]:.3f}',
        ]))


def main(results_dir: Path) -> None:
    speech_dir = results_dir / "speech"
    barge_dir = results_dir / "barge"
    speech_reference = read_pcm(results_dir / "speech-reference.pcm")
    barge_reference = read_pcm(results_dir / "barge-reference.pcm")
    llm_speech_metrics = read_json(speech_dir / "llm-rtc.json")
    neteq_speech_metrics = read_json(speech_dir / "neteq.json")
    llm_barge_metrics = read_json(barge_dir / "llm-rtc.json")
    neteq_barge_metrics = read_json(barge_dir / "neteq.json")

    model_name = os.environ.get("LLM_RTC_ASR_MODEL", "Systran/faster-whisper-base.en")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    # Warm model kernels and file decoding before timed comparisons.
    transcribe(model, results_dir / "speech-reference.pcm", repetitions=1)

    reference_asr = transcribe(model, results_dir / "speech-reference.pcm")
    llm_asr = transcribe(model, speech_dir / "llm-rtc.pcm")
    neteq_pcm_paths = sorted(speech_dir.glob("neteq-run-*/neteq.pcm"))
    neteq_asr_runs = [transcribe(model, path) for path in neteq_pcm_paths]
    neteq_asr = median_run_metrics(
        neteq_asr_runs,
        ["wer_pct", "median_asr_wall_ms"],
    )

    llm_mos, llm_lag_ms = pesq_mos(speech_reference, read_pcm(speech_dir / "llm-rtc.pcm"))
    neteq_mos_runs = []
    for path in neteq_pcm_paths:
        score, lag_ms = pesq_mos(speech_reference, read_pcm(path))
        neteq_mos_runs.append({"pesq_mos_lqo": score, "alignment_lag_ms": lag_ms})
    neteq_mos = median_run_metrics(neteq_mos_runs, ["pesq_mos_lqo", "alignment_lag_ms"])

    events = reference_barge_events(barge_reference)
    llm_barge = barge_metrics(
        barge_reference,
        read_pcm(barge_dir / "llm-rtc.pcm"),
        events,
        llm_barge_metrics["first_content_playout_ms"],
        llm_barge_metrics.get("content_playout_timeline"),
    )
    neteq_barge_pcm_paths = sorted(barge_dir.glob("neteq-run-*/neteq.pcm"))
    neteq_barge_runs = []
    for index, path in enumerate(neteq_barge_pcm_paths):
        stream_start_ms = neteq_barge_metrics["runs"][index]["first_callback_elapsed_ms"]
        neteq_barge_runs.append(barge_metrics(
            barge_reference,
            read_pcm(path),
            events,
            stream_start_ms,
            neteq_barge_metrics["runs"][index].get("callback_timeline"),
        ))
    neteq_barge = median_run_metrics(
        neteq_barge_runs,
        [
            "missed",
            "missed_rate_pct",
            "median_source_to_detection_ms",
            "p95_source_to_detection_ms",
        ],
    )

    llm_ingress_tail_ms = llm_speech_metrics["last_content_playout_end_ms"] - CONTENT_MS
    llm_turn_ms = llm_ingress_tail_ms + llm_asr["median_asr_wall_ms"]
    neteq_turn_runs = []
    for index, asr in enumerate(neteq_asr_runs):
        _, lag_ms = pesq_mos(speech_reference, read_pcm(neteq_pcm_paths[index]))
        reference_end_sample = round((lag_ms / 1_000.0 + CONTENT_MS / 1_000.0) * SAMPLE_RATE) - 1
        end_callback_ms = timeline_callback_ms(
            neteq_speech_metrics["runs"][index].get("callback_timeline"),
            reference_end_sample,
        )
        if end_callback_ms is None:
            first_callback_ms = neteq_speech_metrics["runs"][index]["first_callback_elapsed_ms"]
            end_callback_ms = first_callback_ms + lag_ms + CONTENT_MS
        ingress_tail_ms = end_callback_ms - CONTENT_MS
        neteq_turn_runs.append({
            "ingress_tail_ms": ingress_tail_ms,
            "speech_end_to_transcript_ms": ingress_tail_ms + asr["median_asr_wall_ms"],
        })
    neteq_turn = median_run_metrics(
        neteq_turn_runs,
        ["ingress_tail_ms", "speech_end_to_transcript_ms"],
    )

    load_rows = []
    for calls in [1, 10, 25, 50]:
        llm_load = read_json(results_dir / "load" / f"llm-{calls}" / "llm-rtc-load.json")
        neteq_load = read_json(results_dir / "load" / f"neteq-{calls}" / "neteq-load.json")
        load_rows.extend([llm_load, neteq_load])
    write_load_csv(results_dir, load_rows)

    summary = {
        "scenario": {
            "profile": "severe",
            "frame_ms": read_json(speech_dir / "trace.json")["frame_ms"],
            "base_delay_ms": 50,
            "jitter_ms": 40,
            "loss_rate_pct": 10,
            "llm_rtc_max_latency_ms": 115,
            "llm_rtc_target_latency_ms": llm_speech_metrics["target_latency_ms"],
            "asr_model": model_name,
            "vad": "WebRTC VAD mode 2",
            "mos": "PESQ wideband MOS-LQO",
            "speech_source": (
                (results_dir / "speech-source.txt").read_text().strip()
                if (results_dir / "speech-source.txt").exists()
                else "unknown"
            ),
        },
        "reference_asr": reference_asr,
        "llm_rtc": {
            "asr": llm_asr,
            "pesq_mos_lqo": llm_mos,
            "alignment_lag_ms": llm_lag_ms,
            "transport": {
                key: llm_speech_metrics[key]
                for key in [
                    "continuity_pct",
                    "normal_frames",
                    "fec_frames",
                    "plc_frames",
                    "late_drops",
                    "p50_playout_delay_ms",
                    "p95_playout_delay_ms",
                    "p99_playout_delay_ms",
                ]
            },
            "barge_in": llm_barge,
            "turn_ingress": {
                "ingress_tail_ms": llm_ingress_tail_ms,
                "speech_end_to_transcript_ms": llm_turn_ms,
            },
        },
        "neteq": {
            "asr": neteq_asr,
            "pesq": neteq_mos,
            "transport": {
                key: neteq_speech_metrics[key]
                for key in [
                    "continuity_pct",
                    "concealment_rate_pct",
                    "p50_playout_delay_ms",
                    "p95_playout_delay_ms",
                    "p99_playout_delay_ms",
                ]
            },
            "barge_in": neteq_barge,
            "turn_ingress": neteq_turn,
        },
        "load": load_rows,
        "limitations": [
            "Speech is synthesized rather than recorded from human speakers.",
            "PESQ MOS-LQO is an objective estimator, not a subjective listening-panel MOS.",
            "Turn latency ends at the completed batch-ASR transcript; LLM generation and TTS are not present in this repository and are not modeled.",
            "NetEq load includes Node, libwebrtc peer connections, ICE/DTLS/SRTP, and audio sinks; llm-rtc load covers its jitter and Opus decode path.",
        ],
    }
    (results_dir / "application-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("engine,wer_pct,missed_barge_pct,barge_p95_ms,speech_end_to_transcript_ms,pesq_mos_lqo")
    print(",".join([
        "llm-rtc",
        f"{llm_asr['wer_pct']:.2f}",
        f"{llm_barge['missed_rate_pct']:.2f}",
        f"{llm_barge['p95_source_to_detection_ms']:.2f}",
        f"{llm_turn_ms:.2f}",
        f"{llm_mos:.3f}",
    ]))
    print(",".join([
        "Chromium NetEq",
        f"{neteq_asr['wer_pct']:.2f}",
        f"{neteq_barge['missed_rate_pct']:.2f}",
        f"{neteq_barge['p95_source_to_detection_ms']:.2f}",
        f"{neteq_turn['speech_end_to_transcript_ms']:.2f}",
        f"{neteq_mos['pesq_mos_lqo']:.3f}",
    ]))
    print("\nengine,calls,cpu_ms_per_audio_s_per_call,peak_rss_mib,batch_wall_ms")
    for row in load_rows:
        print(",".join([
            row["engine"],
            str(row["concurrent_calls"]),
            f"{row['median_cpu_ms_per_audio_second_per_call']:.3f}",
            f"{row['peak_rss_mib']:.2f}",
            f"{row['median_batch_wall_ms']:.2f}",
        ]))


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "usage: application_benchmark.py <application-results-dir> [--tuning]"
        )
    if len(sys.argv) == 3 and sys.argv[2] == "--tuning":
        evaluate_tuning(Path(sys.argv[1]))
    elif len(sys.argv) == 2:
        main(Path(sys.argv[1]))
    else:
        raise SystemExit("unknown option")
