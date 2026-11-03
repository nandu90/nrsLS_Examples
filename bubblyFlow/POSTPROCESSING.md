# Bubbly-channel post-processing workflow

The workflow separates interpolation from statistical analysis so that the
expensive Nek-to-structured-grid operation is performed only once.

## 1. Select statistically stationary checkpoints

Use the `CHANNEL` lines in the run log to identify an interval over which
`Ub`, `utau`, `ReTau`, and `kMonitor` fluctuate without drift.  Use the same
sampling interval and approximately the same number of snapshots for the
single- and two-phase cases.

The snapshots should be separated enough in time to avoid treating strongly
correlated fields as independent samples.

## 2. Interpolate the NekRS checkpoints

Run from the directory containing the selected checkpoint files:

```bash
mpiexec -n 8 python ../interpolateRestart.py \
  --pattern 'channel0.f00*' \
  --nx 256 --ny 129 --nz 128 \
  --output interpolatedData
```

The default physical domain is the one defined in `channel.udf`:

- `x` in `[-pi, pi)`, uniform and periodic;
- `y` in `[-1, 1]`, including both walls and tanh-stretched with `beta=1.5`;
- `z` in `[-pi/2, pi/2)`, uniform and periodic.

The output contains `u`, `v`, `w`, `p`, `T`, and `cls`. A checkpoint header such
as `XUPTS02` stores temperature in Nek's dedicated `T` field and has two passive
scalars: TLS=`s0` and CLS=`s1`. The interpolation script therefore reads
temperature from `T` and CLS from `s1` explicitly; TLS is not exported.

The default grid stores about 0.19 GiB of raw field values per snapshot. A
`512 x 257 x 256` production grid stores about 1.51 GiB per snapshot. Choose
`nx` and `nz` based on the maximum wavenumber needed, not merely disk space.

## 3. Extract spectra

```bash
mpiexec -n 8 python ../extractSpectra.py \
  --input 'interpolatedData/*.npz' \
  --output spectraData \
  --re-tau 280 \
  --target-yplus 12 280
```

The script performs one-dimensional FFTs only in the homogeneous periodic
directions.  It writes:

- `spectraData/spectra.npz`: `Ex(y,kx,component)`,
  `Ez(y,kz,component)`, velocity and temperature spectra, variances, and TKE;
- premultiplied, variance-normalized spectra at the requested wall distances;
- premultiplied temperature spectra when `T` is present;
- a Parseval error report for normalization verification.

The zero Fourier mode is removed by subtracting each instantaneous x-z plane
mean.  The saved nonzero spectra therefore describe spatial turbulent
fluctuations and integrate to the saved plane variance.

## 4. Extract statistics and anisotropy

```bash
mpiexec -n 8 python ../extractStatistics.py \
  --input 'interpolatedData/*.npz' \
  --output statisticsData \
  --re 5600 --re-tau 280
```

The script uses intrinsic phase averaging:

```text
<f>_liquid = <cls f>/<cls>
<f>_gas    = <(1-cls) f>/<1-cls>
```

It writes:

- phase mean velocities and all six Reynolds stresses;
- phase TKE and Reynolds-stress anisotropy tensors;
- anisotropy invariants, Lumley flatness, eigenvalues, and barycentric
  coordinates;
- gas-volume-fraction profiles;
- mean and RMS temperature, velocity--temperature covariance, and
  phase-conditioned temperature statistics when `T` is present;
- `temperatureProfile.dat` and `temperatureStatistics.png` when `T` is
  present;
- `anisotropyProfile.dat` and summary figures for both modes;
- conventional channel TKE-budget terms, `tkeBudget.dat`, and
  `tkeBudget.png` for a single-phase data set only.

The conventional budget includes production, dissipation, turbulent
transport, pressure transport, and viscous diffusion. It is generated only
for the constant-property single-phase channel. If gas is detected, the
second budget pass is skipped: no two-phase budget or residual is written.

With checkpoints restricted to `u`, `v`, `w`, `p`, the dedicated temperature
field `T`, and the two passive scalars TLS and CLS, the two-phase-only terms
are accumulated in situ by `channel.udf`. They are
first x-z averaged on the device and then time averaged as one-dimensional
wall-normal profiles. At each normal checkpoint event an interval block is
appended to `budgetProfiles.dat` and the interval accumulator is reset. No
additional full-mesh field file is written, and the restart layout is
unchanged.

## 5. Plot and combine the in-situ two-phase profiles

Each block in `budgetProfiles.dat` is already x-z planar averaged and time
averaged over the duration recorded in its interval header. Combine
chronological restart files for one case with `+`, and use `--t-min` to select
the statistically stationary interval:

```bash
python ../plotBudgetProfiles.py \
  --input 'run1/budgetProfiles.dat+run2/budgetProfiles.dat' \
  --labels 'N=7' \
  --t-min 20 \
  --output budgetProfilePlots
```

Later files replace overlapping intervals from earlier files. Several meshes
can be compared without averaging them together:

```bash
python ../plotBudgetProfiles.py \
  --input 'hbase/p7/budgetProfiles.dat' \
          'hbase/p9/budgetProfiles.dat' \
          'hrefine/p7/budgetProfiles.dat' \
  --labels 'N=7' 'N=9' 'h/2, N=7' \
  --t-min 20 \
  --output budgetProfilePlots
```

Newly generated files also contain mean and RMS temperature, conductivity,
volumetric heat capacity, velocity--temperature covariance, and convective
enthalpy-flux profiles. Old files remain readable but cannot supply thermal
profiles retroactively.

## 6. Reduce the in-situ two-phase fields

Run this in the two-phase case directory after the simulation:

```bash
python ../extractInSituBudget.py \
  --input 'budgetProfiles*.dat' \
  --output inSituBudgetData \
  --re-tau 280
```

The reducer weights each already planar-averaged interval by its actual
averaging duration and merges duplicate GLL points at element boundaries. It writes
`inSituBudgetProfiles.npz`, a text table, and a diagnostic plot containing:

- gas fraction, mean velocity, density, and Favre mean velocity;
- all three surface-tension-force components;
- total, mean, and fluctuating surface-tension work;
- instantaneous viscous dissipation and all six viscous stresses;
- pressure-gradient and gravity power.

The column ordering in `budgetProfiles.dat` is fixed by `channel.udf` and
checked by the reducer. Old mechanical-only and new thermal layouts are both
recognized, but do not combine the two layouts in one reduction.

Set `collectAvg` near the top of `postProcessing.inc` to `true` only when the
desired statistical averaging interval begins. Check its value before every
production run rather than assuming a particular default.

Before restarting with `collectAvg = true`, rename the existing file, for
example to `budgetProfiles_segment01.dat`; setup opens a new
`budgetProfiles.dat`. The default reducer glob includes both files.

These profiles provide the terms that cannot be reconstructed from the
restricted restarts. The restart workflow still supplies mean velocity,
Reynolds stresses, anisotropy, spectra, and the velocity-pressure quantities.
The plotted in-situ power terms are not, by themselves, a closed TKE budget;
closure requires combining them with the restart-derived transport and
production terms on the same statistical interval.

## 7. Plot monitor and runtime histories

Use `plotBubbly.py` to compare the time histories in `bubbly.dat`:

```bash
python ../plotBubbly.py \
  --input hbase/p7/bubbly.dat hbase/p9/bubbly.dat hrefine/p7/bubbly.dat \
  --labels 'N=7' 'N=9' 'h/2, N=7' \
  --output bubblyPlots
```

Use `plotLog.py` for CHANNEL, THERMAL, y+, timestep, CFL, and per-step runtime
diagnostics. Chronological restart logs for one case use the same `+` syntax:

```bash
python ../plotLog.py \
  --input 'log1+log2+log3' \
  --labels 'h/2, N=7' \
  --t-min 20 \
  --output logPlots
```

## 8. Compare single- and two-phase cases

After processing each case separately:

```bash
python ../compareCases.py \
  --single-statistics singlePhase/statisticsData/statistics.npz \
  --two-statistics twoPhase/statisticsData/statistics.npz \
  --single-spectra singlePhase/spectraData/spectra.npz \
  --two-spectra twoPhase/spectraData/spectra.npz \
  --output comparisonPlots
```

The comparison includes liquid TKE, `b11`, Lumley flatness, barycentric-map
trajectories, and premultiplied streamwise-velocity spectra.

## Interpretation checks

Before drawing physical conclusions:

1. Confirm small Parseval errors in both spectra calculations.
2. Confirm that every barycentric point lies inside the realizability
   triangle, allowing only small floating-point excursions.
3. Confirm statistical symmetry between the two channel halves before folding
   profiles.
4. Repeat the calculation with more snapshots to establish convergence.
5. Compare matched `ReTau` values; otherwise Reynolds-number effects can be
   mistaken for bubble-induced changes.
