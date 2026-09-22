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

The output contains `u`, `v`, `w`, `p`, and `cls`.  In this case the scalar
ordering is temperature=`s0`, TLS=`s1`, and CLS=`s2`; the interpolation script
therefore reads `s2` explicitly.

The default grid stores about 0.16 GiB of raw field values per snapshot.  A
`512 x 257 x 256` production grid stores about 1.26 GiB per snapshot.  Choose
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
  `Ez(y,kz,component)`, variances, and TKE;
- premultiplied, variance-normalized spectra at the requested wall distances;
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
- `anisotropyProfile.dat` and summary figures for both modes;
- conventional channel TKE-budget terms, `tkeBudget.dat`, and
  `tkeBudget.png` for a single-phase data set only.

The conventional budget includes production, dissipation, turbulent
transport, pressure transport, and viscous diffusion. It is generated only
for the constant-property single-phase channel. If gas is detected, the
second budget pass is skipped: no two-phase budget or residual is written.

With checkpoints restricted to `u`, `v`, `w`, `p`, `cls`, and `tls`, the
two-phase-only terms are accumulated in situ by `channel.udf`. They are
first x-z averaged on the device and then time averaged as one-dimensional
wall-normal profiles. At each normal checkpoint event an interval block is
appended to `budgetProfiles.dat` and the interval accumulator is reset. No
additional full-mesh field file is written, and the restart layout is
unchanged.

## 5. Reduce the in-situ two-phase fields

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
checked by the reducer. Do not mix files produced by different UDF versions.

Set `collectAvg = true` near the top of `channel.udf` only when the desired
statistical averaging interval begins. It defaults to `false`, so precursor
and transient runs do not allocate or write the extra averaging fields.

Before restarting with `collectAvg = true`, rename the existing file, for
example to `budgetProfiles_segment01.dat`; setup opens a new
`budgetProfiles.dat`. The default reducer glob includes both files.

These profiles provide the terms that cannot be reconstructed from the
restricted restarts. The restart workflow still supplies mean velocity,
Reynolds stresses, anisotropy, spectra, and the velocity-pressure quantities.
The plotted in-situ power terms are not, by themselves, a closed TKE budget;
closure requires combining them with the restart-derived transport and
production terms on the same statistical interval.

## 6. Compare single- and two-phase cases

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
