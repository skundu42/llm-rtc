#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
from pystoi import stoi
from scipy.signal import correlate, correlation_lags

SAMPLE_RATE = 48_000


def read_pcm(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<i2").astype(np.float64) / 32768.0


def align(reference: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, float]:
    factor = 8
    ref_small = reference[::factor]
    degraded_small = degraded[::factor]
    correlation = correlate(degraded_small, ref_small, mode="full", method="fft")
    lags = correlation_lags(len(degraded_small), len(ref_small), mode="full")
    max_lag = int(2 * SAMPLE_RATE / factor)
    valid = np.abs(lags) <= max_lag
    lag_small = lags[valid][int(np.argmax(correlation[valid]))]
    lag = int(lag_small * factor)
    if lag >= 0:
        aligned = degraded[lag:lag + len(reference)]
    else:
        aligned = np.pad(degraded, (-lag, 0))[:len(reference)]
    if len(aligned) < len(reference):
        aligned = np.pad(aligned, (0, len(reference) - len(aligned)))
    return aligned, lag * 1000.0 / SAMPLE_RATE


def si_sdr(reference: np.ndarray, degraded: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    degraded = degraded - np.mean(degraded)
    scale = np.dot(degraded, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    noise = degraded - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def segmental_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    frame = 960
    values = []
    for offset in range(0, len(reference) - frame + 1, frame):
        clean = reference[offset:offset + frame]
        test = degraded[offset:offset + frame]
        signal_energy = np.sum(clean * clean)
        if signal_energy < 1e-6:
            continue
        noise_energy = np.sum((clean - test) ** 2)
        values.append(np.clip(10 * np.log10((signal_energy + 1e-12) / (noise_energy + 1e-12)), -10, 35))
    return float(np.mean(values)) if values else 0.0


def quality(reference: np.ndarray, path: Path) -> dict:
    degraded = read_pcm(path)
    aligned, lag_ms = align(reference, degraded)
    return {
        "stoi": float(stoi(reference, aligned, SAMPLE_RATE, extended=False)),
        "si_sdr_db": si_sdr(reference, aligned),
        "segmental_snr_db": segmental_snr(reference, aligned),
        "alignment_lag_ms": lag_ms,
        "captured_samples": int(len(degraded)),
    }


def median_quality(rows: list[dict]) -> dict:
    result = {}
    for key in rows[0]:
        result[key] = float(np.median([row[key] for row in rows]))
    result["runs"] = rows
    return result


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def main(results_dir: Path) -> None:
    reference = read_pcm(results_dir / "reference.pcm")
    profiles = ["clean", "moderate", "severe"]
    summary = {"sample_rate": SAMPLE_RATE, "content_seconds": 10, "profiles": {}}
    for profile in profiles:
        profile_dir = results_dir / profile
        with (profile_dir / "llm-rtc.json").open() as handle:
            llm = json.load(handle)
        with (profile_dir / "neteq.json").open() as handle:
            neteq = json.load(handle)
        llm["audio_quality"] = quality(reference, profile_dir / "llm-rtc.pcm")
        neteq_quality_runs = [
            quality(reference, run_path)
            for run_path in sorted(profile_dir.glob("neteq-run-*/neteq.pcm"))
        ]
        if not neteq_quality_runs:
            neteq_quality_runs = [quality(reference, profile_dir / "neteq.pcm")]
        neteq["audio_quality"] = median_quality(neteq_quality_runs)
        (profile_dir / "llm-rtc.json").write_text(json.dumps(llm, indent=2) + "\n")
        (profile_dir / "neteq.json").write_text(json.dumps(neteq, indent=2) + "\n")
        summary["profiles"][profile] = {"llm_rtc": llm, "neteq": neteq}

    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("engine,profile,continuity_pct,concealment_pct,stoi,si_sdr_db,p95_ms,p99_ms,cpu_ms_per_audio_s,peak_rss_mib")
    for profile in profiles:
        pair = summary["profiles"][profile]
        for key in ["llm_rtc", "neteq"]:
            row = pair[key]
            quality_row = row["audio_quality"]
            print(",".join([
                row["engine"], profile,
                fmt(row["continuity_pct"]), fmt(row["concealment_rate_pct"]),
                fmt(quality_row["stoi"], 3), fmt(quality_row["si_sdr_db"]),
                fmt(row["p95_playout_delay_ms"]), fmt(row["p99_playout_delay_ms"]),
                fmt(row["median_cpu_ms_per_audio_second"]), fmt(row["peak_rss_mib"]),
            ]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze.py <results-dir>")
    main(Path(sys.argv[1]))
