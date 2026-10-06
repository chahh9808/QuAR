# Jetson Orin Nano latency measurement

Per-image forward latency of every Table 1 method on the deployment device (Appendix tables
`tab:ondevice` and `tab:chunkmem`). The quantized graph is simulated in PyTorch (no integer kernels run), so the
paper reports ratios to No Adapt on the same device, not integer-kernel milliseconds.

```bash
bash jetson/setup_env.sh        # Python environment on the device (JetPack CUDA)
bash jetson/link_cuda_libs.sh   # expose the JetPack CUDA libraries to that environment
bash jetson/preflight.sh        # one-batch check of every method before the long run
bash jetson/measure_latency.sh  # the measurement; QUAR_CHUNK=2 for the in-place chunked variant
```

`measure_latency.sh` reads the W6A6 ViT-B checkpoint and source statistics from `checkpoints/`
and `stats/`, as the other launchers do.
