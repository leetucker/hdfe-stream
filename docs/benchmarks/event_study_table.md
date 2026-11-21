Measured on an EC2 c8id.4xlarge (16 logical CPUs, 30.8 GB), 2026-11-21; R 4.5.3, HonestDiD 0.2.8, fixest 0.14.2, diff-diff 4.0.0, hdfe-stream 0.12.0.

| | Event study (s) | Relative magnitudes, C-LF (s) | Smoothness, FLCI (s) | Peak memory (GB) |
|---|---:|---:|---:|---:|
| hdfe_stream | 8.24 | 0.132 | 0.0765 | 2.8 |
| R: fixest, HonestDiD | 6.68 | 102 | 13.1 | 6.2 |
| R: fixest, HonestDiD (parallel) | 6.97 | 17.1 | 5.49 | 10.4 |
| diff-diff | 61.3 | – | 2.86 | 15.8 |
