# gaussian-approx

ASCII terminal visualizer for a Kalman-based Gaussian approximation filter.

## What it does

This is a visualization tool that exercises a sigma-point style Kalman filter
library. It simulates tracking problems and renders the filter state as ASCII
art in your terminal — think 2D grids with moving markers, Gaussian bell curves,
and covariance ellipses.

Built as a hobby project to understand how unscented/sigma-point filters work
by actually watching them run step by step.

## Build

```bash
make
./vizga
```

No external dependencies beyond gcc and standard libc/libm.

## Usage

```
./vizga [options]
  -m <mode>   demo mode: 2d, 1d, multi, rot, test, grid, compare (default: 2d)
  -t <traj>   trajectory: circle, line, fig8, random (default: circle)
  -n <steps>  number of steps                 (default: 100)
  -d <dt>     time step                       (default: 0.1)
  -L <level>  approximation level (3,5,7)     (default: 7)
  -s <seed>   random seed                     (default: time-based)
  -k <num>    number of targets (1-4)         (default: 2, multi mode)
  -o <file>   export data to CSV file
  -q          quiet mode (final result only)
  -i          interactive mode (step with keyboard)
  -v          verbose mode (show sigma points)
  --speed <ms> animation delay in ms           (default: 80)
  --no-color  disable ANSI colors
  --loop      auto-restart with new seed when done
  --metrics   show convergence metrics (NIS, sparkline)
  -h          show this help
```

## Controls (interactive mode)

- space/enter — step forward
- p — pause/resume
- r — restart
- n — next trajectory type (2d mode)
- q — quit

## Demo Modes

### 2D Tracking (`-m 2d`)

The default mode. Tracks a target moving along a trajectory (circle, line,
figure-8, or random walk) using noisy position measurements.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Step 42/100  t=4.2s                       │
│                                                                  │
│     y                                                            │
│     ^                                                            │
│   8+        ..............                                       │
│    |      ..              ..                                     │
│    |     .                  .   ~~~~                             │
│    |    .      o            .~      ~           . = true path    │
│   4+   .     o    +         .                  o = measurement  │
│    |  .                     .                  + = estimate     │
│    | .                      .                  ~ = covariance   │
│    |.                       .                                   │
│   0+-------------------------.----------->x                      │
│     0        4        8       12       16                        │
│                                                                  │
│  RMSE: 1.23  [####....] 42%  [PAUSED]                           │
└──────────────────────────────────────────────────────────────────┘
```

### 1D Tracking (`-m 1d`)

Simpler demo showing a 1D position filter with Gaussian PDF visualization.

### Multi-Target (`-m multi`)

Track up to 4 independent targets simultaneously. Each has its own filter
instance, trajectory, and color-coded markers (A/B/C/D).

### Rotation (`-m rot`)

12-state filter tracking both position and 3D rotation. Shows projected
rotated axes on the 2D grid using Rodrigues' rotation formula.

### Compare (`-m compare`)

Run L=3, L=5, and L=7 sigma point approximations side by side to compare
accuracy vs. computational cost.

## Example

Quick demo with circular trajectory:

```bash
make demo
```

Or manually:

```bash
./vizga -t circle -n 80 --speed 80
```

## Background

Kalman filters estimate the state of a system from noisy measurements. The
"sigma point" or "unscented" approach approximates the Gaussian distribution
by propagating carefully chosen sample points through nonlinear motion
and observation models, then reconstructing the mean and covariance.

This implementation uses a pre-computed optimal sample placement for three
precision levels (L=3, 5, or 7 sigma points). The filter runs in pure C with
no external dependencies.

## Export

Use `-o file.csv` to export tracking data for offline analysis. A gnuplot
script (`plot.gp`) is included for quick visualization:

```bash
./vizga -n 200 -o track.csv -q
gnuplot -p plot.gp
```

## Files

```
matrix.c/h       - custom matrix library (heap-allocated)
eig.c/h          - eigenvalue decomposition (Householder + QL)
gaussianApprox.c/h - optimal sigma point positions
gaussianEstimator.c/h - core filter predict/update
viz.c/h          - ASCII rendering (bars, grids, ellipses, colors)
sim.c/h          - trajectory generation and scenarios
export.c/h       - CSV data export
main.c           - CLI and demo dispatch
```

## TODO

- add measurement gating (maybe someday)
- handle non-square grids
- add process noise adaptation
