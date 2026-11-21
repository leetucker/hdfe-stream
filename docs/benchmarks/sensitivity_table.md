Measured on an EC2 c8id.4xlarge (16 logical CPUs, 30.8 GB), 2026-11-21; R 4.5.3, HonestDiD 0.2.8, fixest 0.14.2, diff-diff 4.0.0, hdfe-stream 0.12.0.

**One value of M**, seconds:

| Pre-treatment periods | Restriction, interval | R HonestDiD | hdfe-stream, 1 thread | hdfe-stream, 16 threads | diff-diff |
|---:|---|---:|---:|---:|---:|
| 4 | relative magnitudes, C-LF | 7.56 | 0.0257 | 0.0121 | – |
| 4 | smoothness and relative magnitudes, C-LF | 7.55 | 0.0235 | 0.0111 | – |
| 4 | smoothness, FLCI | 2.81 | 0.00323 | 0.0033 | 0.31 |
| 4 | smoothness, C-F | 7.07 | 0.0116 | 0.00956 | – |
| 8 | relative magnitudes, C-LF | 11 | 0.0427 | 0.0168 | – |
| 8 | smoothness and relative magnitudes, C-LF | 11.1 | 0.0392 | 0.0162 | – |
| 8 | smoothness, FLCI | 2.85 | 0.00889 | 0.00879 | 0.323 |
| 8 | smoothness, C-F | 6.95 | 0.0138 | 0.0158 | – |
| 16 | relative magnitudes, C-LF | 27.6 | 0.0906 | 0.0347 | – |
| 16 | smoothness and relative magnitudes, C-LF | 29.9 | 0.0994 | 0.0386 | – |
| 16 | smoothness, FLCI | 3 | 0.0351 | 0.035 | 0.376 |
| 16 | smoothness, C-F | 4.66 | 0.0444 | 0.0439 | – |
| 32 | relative magnitudes, C-LF | 51.7 | 0.186 | 0.0613 | – |
| 32 | smoothness and relative magnitudes, C-LF | 60.9 | 0.195 | 0.063 | – |
| 32 | smoothness, FLCI | 3.34 | 0.0821 | 0.0805 | 0.802 |
| 32 | smoothness, C-F | 7.42 | 0.0925 | 0.0893 | – |
| 64 | relative magnitudes, C-LF | 115 | 0.393 | 0.122 | – |
| 64 | smoothness and relative magnitudes, C-LF | 156 | 0.538 | 0.142 | – |
| 64 | smoothness, FLCI | 7.6 | 0.178 | 0.177 | 4.25 |
| 64 | smoothness, C-F | 9.19 | 0.197 | 0.197 | – |

**R's default grid of 10 values of M**, seconds:

| Pre-treatment periods | Restriction, interval | R HonestDiD | R HonestDiD, parallel | hdfe-stream, 1 thread | hdfe-stream, 16 threads | diff-diff |
|---:|---|---:|---:|---:|---:|---:|
| 4 | relative magnitudes, C-LF | 68.6 | 12.9 | 0.226 | 0.0712 | – |
| 4 | smoothness and relative magnitudes, C-LF | 66.8 | 12.3 | 0.207 | 0.0615 | – |
| 4 | smoothness, FLCI | 11.1 | 5.74 | 0.0261 | 0.0262 | 2.74 |
| 4 | smoothness, C-F | 33.5 | 8.9 | 0.0822 | 0.0444 | – |
| 8 | relative magnitudes, C-LF | 107 | 19.2 | 0.386 | 0.115 | – |
| 8 | smoothness and relative magnitudes, C-LF | 108 | 19.9 | 0.359 | 0.11 | – |
| 8 | smoothness, FLCI | 18.6 | 7.45 | 0.0744 | 0.0746 | 3.03 |
| 8 | smoothness, C-F | 35.3 | 8.45 | 0.124 | 0.0932 | – |
| 16 | relative magnitudes, C-LF | 278 | 47.6 | 0.872 | 0.251 | – |
| 16 | smoothness and relative magnitudes, C-LF | 302 | 51.6 | 4.74 | 0.3 | – |
| 16 | smoothness, FLCI | 12.3 | 6.01 | 0.341 | 0.346 | 3.4 |
| 16 | smoothness, C-F | 34.9 | 9.84 | 0.369 | 0.335 | – |
| 32 | relative magnitudes, C-LF | 501 | 89.3 | 1.74 | 0.492 | – |
| 32 | smoothness and relative magnitudes, C-LF | 587 | 102 | 1.99 | 0.534 | – |
| 32 | smoothness, FLCI | 15.4 | 6.87 | 0.744 | 0.746 | 8 |
| 32 | smoothness, C-F | 36.4 | 9.48 | 0.823 | 0.779 | – |
| 64 | relative magnitudes, C-LF | 1,056 | 182 | 3.85 | 1.11 | – |
| 64 | smoothness and relative magnitudes, C-LF | 1,522 | 244 | 5.78 | 1.39 | – |
| 64 | smoothness, FLCI | 43.8 | 40.1 | 1.64 | 1.65 | 40.7 |
| 64 | smoothness, C-F | 59.1 | 42.3 | 1.76 | 1.71 | – |

Largest increase in peak memory over the process's value after loading the packages:

- R HonestDiD: 706 MB
- R HonestDiD, parallel: 6,111 MB
- hdfe-stream, 1 thread: 488 MB
- hdfe-stream, 16 threads: 491 MB
- diff-diff: 8 MB
