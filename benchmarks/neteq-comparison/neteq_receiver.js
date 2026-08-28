#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const wrtc = require('@roamhq/wrtc');
const { RTCAudioSink } = wrtc.nonstandard;

const [profile, inputPath, outputDir, repetitionsText = '3'] = process.argv.slice(2);
if (!profile || !inputPath || !outputDir) {
  console.error('usage: neteq_receiver.js <profile> <input.pcm> <output-dir> [repetitions]');
  process.exit(2);
}
const repetitions = Number(repetitionsText);
const senderPath = '/work/target/release/examples/neteq_trace_sender';

function waitForIceGathering(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('ICE gathering timed out')), 10000);
    const handler = () => {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timer);
        pc.removeEventListener('icegatheringstatechange', handler);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', handler);
  });
}

function weightedPercentile(samples, quantile) {
  if (samples.length === 0) return 0;
  const sorted = [...samples].sort((a, b) => a.value - b.value);
  const total = sorted.reduce((sum, sample) => sum + sample.weight, 0);
  const target = total * quantile;
  let cumulative = 0;
  for (const sample of sorted) {
    cumulative += sample.weight;
    if (cumulative >= target) return sample.value;
  }
  return sorted[sorted.length - 1].value;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function inboundAudioReport(reportSet) {
  for (const report of reportSet.values()) {
    if (report.type === 'inbound-rtp' && (report.kind === 'audio' || report.mediaType === 'audio')) {
      return report;
    }
  }
  return null;
}

async function runOnce(runIndex) {
  const runDir = path.join(outputDir, `neteq-run-${runIndex + 1}`);
  fs.mkdirSync(runDir, { recursive: true });
  const child = spawn(senderPath, ['neteq-sender', profile, inputPath, runDir], {
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const pc = new wrtc.RTCPeerConnection({ iceServers: [] });
  let sink = null;
  const pcmChunks = [];
  let sinkSamples = 0;
  let sinkCallbacks = 0;
  let callbackGaps = 0;
  let lastCallbackNs = null;
  let measurementStarted = false;
  let cpuStart = null;
  let peakRss = process.memoryUsage().rss;
  let memoryTimer = null;
  let statsTimer = null;
  let previousStats = null;
  let mediaBaseline = null;
  let mediaEndStats = null;
  const delaySamples = [];
  let finalInbound = null;

  pc.ontrack = ({ track }) => {
    sink = new RTCAudioSink(track);
    sink.ondata = ({ samples }) => {
      if (!measurementStarted) return;
      const now = process.hrtime.bigint();
      if (lastCallbackNs !== null && Number(now - lastCallbackNs) / 1e6 > 16) callbackGaps += 1;
      lastCallbackNs = now;
      const copy = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      pcmChunks.push(Buffer.from(copy));
      sinkSamples += samples.length;
      sinkCallbacks += 1;
    };
  };

  async function sampleStats() {
    const inbound = inboundAudioReport(await pc.getStats());
    if (!inbound) return;
    finalInbound = Object.fromEntries(Object.entries(inbound));
    if (!mediaBaseline && Number(inbound.packetsReceived || 0) > 0) {
      mediaBaseline = {
        totalSamplesReceived: Number(inbound.totalSamplesReceived || 0),
        concealedSamples: Number(inbound.concealedSamples || 0),
        silentConcealedSamples: Number(inbound.silentConcealedSamples || 0),
        fecPacketsReceived: Number(inbound.fecPacketsReceived || 0),
      };
    }
    const delay = Number(inbound.jitterBufferDelay || 0);
    const emitted = Number(inbound.jitterBufferEmittedCount || 0);
    if (previousStats && emitted > previousStats.emitted && delay >= previousStats.delay) {
      delaySamples.push({
        value: ((delay - previousStats.delay) / (emitted - previousStats.emitted)) * 1000,
        weight: emitted - previousStats.emitted,
      });
    }
    previousStats = { delay, emitted };
  }

  const finished = new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code) => {
      if (code !== 0) reject(new Error(`sender exited with ${code}`));
      else resolve();
    });
  });

  let protocolBuffer = '';
  const lines = [];
  let wakeLine = null;
  child.stdout.on('data', (chunk) => {
    protocolBuffer += chunk.toString('utf8');
    while (protocolBuffer.includes('\n')) {
      const newline = protocolBuffer.indexOf('\n');
      const line = protocolBuffer.slice(0, newline).trim();
      protocolBuffer = protocolBuffer.slice(newline + 1);
      if (line) lines.push(line);
    }
    if (wakeLine) {
      const wake = wakeLine;
      wakeLine = null;
      wake();
    }
  });
  async function nextLine() {
    while (lines.length === 0) {
      await new Promise((resolve) => { wakeLine = resolve; });
    }
    return lines.shift();
  }

  if (await nextLine() !== 'OFFER_READY') throw new Error('sender did not produce an offer');
  const offer = fs.readFileSync(path.join(runDir, 'offer.sdp'), 'utf8');
  await pc.setRemoteDescription({ type: 'offer', sdp: offer });
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  await waitForIceGathering(pc);
  fs.writeFileSync(path.join(runDir, 'answer.sdp'), pc.localDescription.sdp);
  child.stdin.write('ANSWER_READY\n');
  if (await nextLine() !== 'READY') throw new Error('sender did not connect');

  measurementStarted = true;
  cpuStart = process.cpuUsage();
  memoryTimer = setInterval(() => {
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
  }, 10);
  statsTimer = setInterval(() => { sampleStats().catch(() => {}); }, 10);
  child.stdin.write('GO\n');
  if (await nextLine() !== 'SENT') throw new Error('sender did not finish the trace');
  await sampleStats();
  mediaEndStats = { ...finalInbound };
  // Five guard packets already provide 100 ms for final FEC/plc decisions.
  // A short extra drain lets the last packet play without counting a long
  // post-stream run of NetEq-generated silence as media concealment.
  await new Promise((resolve) => setTimeout(resolve, 300));
  await sampleStats();
  const cpu = process.cpuUsage(cpuStart);
  clearInterval(memoryTimer);
  clearInterval(statsTimer);
  measurementStarted = false;
  child.stdin.write('STOP\n');
  await finished;
  if (sink) sink.stop();
  await pc.close();

  const capturedPcm = Buffer.concat(pcmChunks);
  fs.writeFileSync(path.join(runDir, 'neteq.pcm'), capturedPcm);
  if (runIndex === 0) fs.writeFileSync(path.join(outputDir, 'neteq.pcm'), capturedPcm);
  const totalSamples = Math.max(0,
    Number(mediaEndStats?.totalSamplesReceived || sinkSamples) -
    Number(mediaBaseline?.totalSamplesReceived || 0));
  const concealedSamples = Math.max(0,
    Number(mediaEndStats?.concealedSamples || 0) -
    Number(mediaBaseline?.concealedSamples || 0));
  const silentConcealedSamples = Math.max(0,
    Number(mediaEndStats?.silentConcealedSamples || 0) -
    Number(mediaBaseline?.silentConcealedSamples || 0));
  const audibleConcealedSamples = Math.max(0, concealedSamples - silentConcealedSamples);
  const elapsedAudioSeconds = 10;
  return {
    engine: 'Chromium NetEq',
    implementation: '@roamhq/wrtc 0.10.0 (libwebrtc)',
    profile,
    sink_samples: sinkSamples,
    sink_callbacks: sinkCallbacks,
    callback_gaps_over_16ms: callbackGaps,
    packets_received: Number(finalInbound?.packetsReceived || 0),
    packets_lost: Number(finalInbound?.packetsLost || 0),
    total_samples_received: totalSamples,
    concealed_samples: concealedSamples,
    silent_concealed_samples: silentConcealedSamples,
    audible_concealed_samples: audibleConcealedSamples,
    concealment_events: Number(finalInbound?.concealmentEvents || 0),
    fec_packets_received: Math.max(0,
      Number(mediaEndStats?.fecPacketsReceived || 0) -
      Number(mediaBaseline?.fecPacketsReceived || 0)),
    concealment_rate_pct: totalSamples > 0 ? audibleConcealedSamples / totalSamples * 100 : 0,
    p50_playout_delay_ms: weightedPercentile(delaySamples, 0.50),
    p95_playout_delay_ms: weightedPercentile(delaySamples, 0.95),
    p99_playout_delay_ms: weightedPercentile(delaySamples, 0.99),
    cpu_ms_per_audio_second: (cpu.user + cpu.system) / 1000 / elapsedAudioSeconds,
    peak_rss_mib: peakRss / 1024 / 1024,
    raw_inbound_stats: finalInbound,
  };
}

(async () => {
  fs.mkdirSync(outputDir, { recursive: true });
  const runs = [];
  for (let i = 0; i < repetitions; i += 1) runs.push(await runOnce(i));
  const metric = (key) => median(runs.map((run) => run[key]));
  const summary = {
    engine: 'Chromium NetEq',
    implementation: '@roamhq/wrtc 0.10.0 (libwebrtc)',
    profile,
    repetitions,
    continuity_pct: Math.max(0, 100 - metric('callback_gaps_over_16ms') / 1000 * 100),
    concealment_rate_pct: metric('concealment_rate_pct'),
    fec_packets_received: metric('fec_packets_received'),
    p50_playout_delay_ms: metric('p50_playout_delay_ms'),
    p95_playout_delay_ms: metric('p95_playout_delay_ms'),
    p99_playout_delay_ms: metric('p99_playout_delay_ms'),
    median_cpu_ms_per_audio_second: metric('cpu_ms_per_audio_second'),
    peak_rss_mib: metric('peak_rss_mib'),
    runs,
  };
  fs.writeFileSync(path.join(outputDir, 'neteq.json'), JSON.stringify(summary, null, 2));
  // Some node-webrtc builds fault while global native objects are torn down.
  // All peers/sinks are explicitly closed above; exiting here avoids running
  // those redundant process-global destructors.
  process.exit(0);
})().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
