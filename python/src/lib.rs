use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use llm_rtc_core::audio::codec as core_codec;
use llm_rtc_core::audio::jitter as core_jitter;
use llm_rtc_core::audio::pipeline as core_pipeline;
use llm_rtc_core::audio::processor as core_processor;

fn config_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn runtime_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct PyCodecConfig {
    inner: core_codec::CodecConfig,
}

#[pymethods]
impl PyCodecConfig {
    #[new]
    #[pyo3(signature = (
        sample_rate = 48_000,
        channels = 1,
        bitrate = 24_000,
        frame_size_ms = 10.0,
        use_dtx = true,
        use_fec = true,
        complexity = 0,
    ))]
    fn new(
        sample_rate: u32,
        channels: u8,
        bitrate: u32,
        frame_size_ms: f32,
        use_dtx: bool,
        use_fec: bool,
        complexity: u8,
    ) -> Self {
        Self {
            inner: core_codec::CodecConfig {
                sample_rate,
                channels,
                bitrate,
                frame_size_ms,
                use_dtx,
                use_fec,
                complexity,
            },
        }
    }
}

#[pyclass]
struct PyOpusEncoder {
    inner: core_codec::OpusEncoder,
}

#[pymethods]
impl PyOpusEncoder {
    #[new]
    fn new(config: PyCodecConfig) -> PyResult<Self> {
        Ok(Self {
            inner: core_codec::OpusEncoder::new(config.inner).map_err(config_err)?,
        })
    }

    fn encode(&self, pcm: Vec<i16>) -> PyResult<Vec<u8>> {
        self.inner.encode(&pcm).map_err(runtime_err)
    }

    fn encode_frames(&mut self, pcm: Vec<i16>) -> PyResult<Vec<Vec<u8>>> {
        self.inner.encode_frames(&pcm).map_err(runtime_err)
    }

    fn samples_per_frame(&self) -> usize {
        self.inner.samples_per_frame()
    }
}

#[pyclass(unsendable)]
struct PyOpusDecoder {
    inner: core_codec::OpusDecoder,
}

#[pymethods]
impl PyOpusDecoder {
    #[new]
    fn new(config: PyCodecConfig) -> PyResult<Self> {
        Ok(Self {
            inner: core_codec::OpusDecoder::new(config.inner).map_err(config_err)?,
        })
    }

    fn decode(&mut self, packet: &[u8]) -> PyResult<Vec<i16>> {
        self.inner.decode(packet).map_err(runtime_err)
    }

    fn decode_fec(&mut self, packet: &[u8]) -> PyResult<Vec<i16>> {
        self.inner.decode_fec(packet).map_err(runtime_err)
    }
}

// ---------------------------------------------------------------------------
// Jitter buffer
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct PyJitterBufferConfig {
    inner: core_jitter::JitterBufferConfig,
}

#[pymethods]
impl PyJitterBufferConfig {
    #[new]
    #[pyo3(signature = (
        max_latency_ms = 120,
        target_latency_ms = 5,
        max_packets = 100,
        sample_rate = 48_000,
        frame_size_ms = 10,
    ))]
    fn new(
        max_latency_ms: u32,
        target_latency_ms: u32,
        max_packets: usize,
        sample_rate: u32,
        frame_size_ms: u32,
    ) -> Self {
        Self {
            inner: core_jitter::JitterBufferConfig {
                max_latency_ms,
                target_latency_ms,
                max_packets,
                sample_rate,
                frame_size_ms,
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyAudioPacket {
    inner: core_jitter::AudioPacket,
}

#[pymethods]
impl PyAudioPacket {
    #[new]
    fn new(sequence_number: u16, timestamp: u32, payload: Vec<u8>) -> Self {
        Self {
            inner: core_jitter::AudioPacket {
                sequence_number,
                timestamp,
                payload,
            },
        }
    }

    #[getter]
    fn sequence_number(&self) -> u16 {
        self.inner.sequence_number
    }

    #[getter]
    fn timestamp(&self) -> u32 {
        self.inner.timestamp
    }

    #[getter]
    fn payload(&self) -> Vec<u8> {
        self.inner.payload.clone()
    }
}

#[pyclass]
#[derive(Clone)]
struct PyJitterStats {
    inner: core_jitter::JitterStats,
}

#[pymethods]
impl PyJitterStats {
    #[getter]
    fn packets_in(&self) -> u64 {
        self.inner.packets_in
    }

    #[getter]
    fn packets_out(&self) -> u64 {
        self.inner.packets_out
    }

    #[getter]
    fn packets_dropped(&self) -> u64 {
        self.inner.packets_dropped
    }

    #[getter]
    fn packets_late(&self) -> u64 {
        self.inner.packets_late
    }

    #[getter]
    fn current_jitter_ms(&self) -> f32 {
        self.inner.current_jitter_ms
    }

    #[getter]
    fn current_target_latency_ms(&self) -> f32 {
        self.inner.current_target_latency_ms
    }
}

#[pyclass]
struct PyJitterBuffer {
    inner: core_jitter::JitterBuffer,
}

#[pymethods]
impl PyJitterBuffer {
    #[new]
    fn new(config: PyJitterBufferConfig) -> Self {
        Self {
            inner: core_jitter::JitterBuffer::new(config.inner),
        }
    }

    fn push(&mut self, packet: PyAudioPacket) {
        self.inner.push(packet.inner);
    }

    fn pop(&mut self) -> Option<PyAudioPacket> {
        self.inner
            .pop()
            .map(|packet| PyAudioPacket { inner: packet })
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn stats(&self) -> PyJitterStats {
        PyJitterStats {
            inner: self.inner.stats(),
        }
    }
}

// ---------------------------------------------------------------------------
// Audio processor
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct PyProcessorConfig {
    inner: core_processor::ProcessorConfig,
}

#[pymethods]
impl PyProcessorConfig {
    #[new]
    #[pyo3(signature = (
        enable_aec = true,
        enable_ns = true,
        enable_agc = true,
        enable_vad = true,
        agc_target_level_dbfs = 3,
        agc_compression_gain_db = 9,
    ))]
    fn new(
        enable_aec: bool,
        enable_ns: bool,
        enable_agc: bool,
        enable_vad: bool,
        agc_target_level_dbfs: i32,
        agc_compression_gain_db: i32,
    ) -> Self {
        Self {
            inner: core_processor::ProcessorConfig {
                enable_aec,
                enable_ns,
                enable_agc,
                enable_vad,
                agc_target_level_dbfs,
                agc_compression_gain_db,
                ..core_processor::ProcessorConfig::default()
            },
        }
    }
}

#[pyclass]
struct PyAudioProcessor {
    inner: core_processor::AudioProcessor,
}

#[pymethods]
impl PyAudioProcessor {
    #[new]
    fn new(config: PyProcessorConfig) -> PyResult<Self> {
        Ok(Self {
            inner: core_processor::AudioProcessor::new(config.inner).map_err(config_err)?,
        })
    }

    fn process(&mut self, near_end: Vec<i16>) -> PyResult<Vec<i16>> {
        let mut frame = near_end;
        self.inner.process(&mut frame).map_err(runtime_err)?;
        Ok(frame)
    }

    fn process_with_reference(
        &mut self,
        near_end: Vec<i16>,
        far_end: Vec<i16>,
    ) -> PyResult<Vec<i16>> {
        let mut frame = near_end;
        self.inner
            .process_with_reference(&mut frame, &far_end)
            .map_err(runtime_err)?;
        Ok(frame)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

#[pyclass(unsendable)]
struct PyAudioPipeline {
    inner: core_pipeline::AudioPipeline,
}

#[pymethods]
impl PyAudioPipeline {
    #[new]
    fn new(
        codec_config: PyCodecConfig,
        jitter_config: PyJitterBufferConfig,
        processor_config: PyProcessorConfig,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core_pipeline::AudioPipeline::new(core_pipeline::AudioPipelineConfig {
                codec: codec_config.inner,
                jitter: jitter_config.inner,
                processor: processor_config.inner,
            })
            .map_err(config_err)?,
        })
    }

    fn process_outgoing(&mut self, mic_pcm: Vec<i16>) -> PyResult<Vec<Vec<u8>>> {
        let mut pcm = mic_pcm;
        self.inner.process_outgoing(&mut pcm).map_err(runtime_err)
    }

    fn push_incoming(&mut self, packet: PyAudioPacket) -> bool {
        self.inner.push_incoming(packet.inner)
    }

    fn pop_decoded(&mut self) -> PyResult<Option<Vec<i16>>> {
        self.inner.pop_decoded().map_err(runtime_err)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _llm_rtc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCodecConfig>()?;
    m.add_class::<PyOpusEncoder>()?;
    m.add_class::<PyOpusDecoder>()?;
    m.add_class::<PyJitterBufferConfig>()?;
    m.add_class::<PyAudioPacket>()?;
    m.add_class::<PyJitterBuffer>()?;
    m.add_class::<PyJitterStats>()?;
    m.add_class::<PyProcessorConfig>()?;
    m.add_class::<PyAudioProcessor>()?;
    m.add_class::<PyAudioPipeline>()?;
    Ok(())
}
