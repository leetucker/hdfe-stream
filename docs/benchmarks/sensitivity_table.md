Measured on one Intel(R) Core(TM) Ultra 7 258V (8 logical CPUs, 26.4 GB), 2026-11-21; R 4.5.3, HonestDiD 0.2.8, fixest 0.14.2, diff-diff 4.0.0, hdfe-stream 0.12.0.

**One value of M**, seconds:

| Pre-treatment periods | Restriction, interval | R HonestDiD | hdfe-stream, 1 thread | hdfe-stream, 8 threads | diff-diff |
|---:|---|---:|---:|---:|---:|
| 4 | relative magnitudes, C-LF | 6.63 | 0.0229 | 0.0121 | – |
| 4 | smoothness and relative magnitudes, C-LF | 6.63 | 0.0239 | 0.0109 | – |
| 4 | smoothness, FLCI | 4.41 | 0.00296 | 0.00269 | 0.274 |
| 4 | smoothness, C-F | 8.23 | 0.00955 | 0.00967 | – |
| 8 | relative magnitudes, C-LF | 11.1 | 0.0382 | 0.0199 | – |
| 8 | smoothness and relative magnitudes, C-LF | 10.9 | 0.0363 | 0.0184 | – |
| 8 | smoothness, FLCI | 4.26 | 0.00746 | 0.00722 | 0.306 |
| 8 | smoothness, C-F | 8.26 | 0.0121 | 0.0116 | – |
| 16 | relative magnitudes, C-LF | 24.4 | 0.0828 | 0.0416 | – |
| 16 | smoothness and relative magnitudes, C-LF | 29.4 | 0.0902 | 0.0413 | – |
| 16 | smoothness, FLCI | 4.36 | 0.0334 | 0.0281 | 0.364 |
| 16 | smoothness, C-F | 5.76 | 0.0377 | 0.0362 | – |
| 32 | relative magnitudes, C-LF | 51.3 | 0.177 | 0.0786 | – |
| 32 | smoothness and relative magnitudes, C-LF | 55.9 | 0.192 | 0.0741 | – |
| 32 | smoothness, FLCI | 4.31 | 0.073 | 0.0705 | 8.3 |
| 32 | smoothness, C-F | 9.09 | 0.0821 | 0.0788 | – |

Largest increase in peak memory over the process's value after loading the packages:

- R HonestDiD: 1,019 MB
- hdfe-stream, 1 thread: 10 MB
- hdfe-stream, 8 threads: 12 MB
- diff-diff: 0 MB
