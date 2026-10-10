# mobile_lm: on-device model architecture

PyTorch definition of the ~1.5B decoder-only model described in the design doc.

- GQA (16 query / 4 KV heads), RoPE, RMSNorm, SwiGLU, tied 32k vocabulary.
- Incremental KV-cache decoding, verified against full-sequence forward.
- `MobileLMConfig.param_count()`, `kv_cache_bytes()` and `weight_bytes_quantized()`
  check changes against the RAM budget without loading weights.

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

The test suite runs on CPU in a few seconds. Full-size instantiation is done on the
`meta` device, because a 1.5B fp32 model needs about 6 GB.

Not yet included: data pipeline, distillation loop, QAT, DPO, and GGUF export.
Training from scratch needs GPU compute, which this repo does not provide.
