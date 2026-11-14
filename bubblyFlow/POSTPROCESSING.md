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

The default grid stores about 0.19 GiB of raw field values per snapshot. The
`512 x 129 x 320` grid used here stores about 0.94 GiB per snapshot. Choose
`nx` and `nz` based on the maximum wavenumber needed, not merely disk space.

## 3. Extract spectra

```bash
mpiexec -n 8 python ../extractSpectra.py \
  --input 'interpolatedData/*.npz' \
  --output spectraData \
  --re-tau 280 \
  --target-yplus 12 280 \
  --phase-filter-count 48 \
  --phase-tolerance 1e-3 \
  --relative-spectrum-floor 1e-5
```

To average snapshots across chronological restart directories, pass one quoted
glob per run in chronological order:

```bash
mpiexec -n 8 python extractSpectra.py \
  --input 'run7/interpolatedData/*.npz' \
          'run8/interpolatedData/*.npz' \
          'run9/interpolatedData/*.npz' \
  --output spectraData
```

A directory may be supplied instead of its `*.npz` glob. Later input segments
replace snapshots from earlier segments at overlapping restart times, and an
exact duplicate time is counted only once. Keep the separate input arguments
rather than using one `run*/...` glob when restart intervals may overlap.

The script performs one-dimensional FFTs only in the homogeneous periodic
directions.  It writes:

- `spectraData/spectra.npz`: `Ex(y,kx,component)`,
  `Ez(y,kz,component)`, velocity and temperature spectra, variances, and TKE;
- premultiplied, variance-normalized spectra at the requested wall distances;
- separate non-premultiplied total-TKE spectra in `x` and `z`;
- premultiplied temperature spectra when `T` is present;
- HIT-style filtered liquid and gas spectra in each homogeneous direction at
  the requested wall distances;
- a Parseval error report for normalization verification.

`Ex` and `Ez` store discrete FFT modal energies, whose sum recovers the saved
plane variance. Premultiplied plots convert modal energy to spectral density
using `E=P/delta_k`; this matters because the streamwise and spanwise domains
have different fundamental wavenumber spacings. Their horizontal axes are the
outer-scaled wavenumbers `kx*delta` and `kz*delta`, where `delta` is the channel
half-gap; both spectral axes are logarithmic. The instantaneous x-z plane
mean is subtracted before each transform, removing the joint horizontal mean.
Only plotted values below `--relative-spectrum-floor` times the peak are
omitted, suppressing the negligible post-decay interpolation/noise floor; the
complete untrimmed modal spectra remain in `spectra.npz`. The total-TKE
spectral densities are also saved as `ExTke` and `EzTke`.

After changing only plotting options, regenerate figures without rereading the
interpolated snapshots:

```bash
python ../extractSpectra.py --plot-only \
  --output spectraData \
  --target-yplus 12 280 \
  --relative-spectrum-floor 1e-5
```

The phase spectra use CLS for the liquid fraction and `1-CLS` for the gas
fraction. They are obtained by differentiating Gaussian-filtered phase kinetic
energy, rather than Fourier transforming a discontinuously masked velocity.
This avoids injecting interface-mask content into the high-wavenumber range.
At every requested wall distance, spectral energies and variances from the
matching lower and upper channel planes are averaged before normalization; the
centerline plane is included only once. The filtered phase spectra likewise
treat the two half-channel planes as separate conditional samples.
They are saved as `ExLiquidFiltered`, `ExGasFiltered`, `EzLiquidFiltered`, and
`EzGasFiltered`, with their filter wavenumbers, phase TKE normalizers, mean
phase fractions, and valid sample counts. Computing them is more expensive
than the ordinary FFT spectra; reduce `--phase-filter-count` if necessary.

## 4. Extract statistics and anisotropy

```bash
mpiexec -n 8 python ../extractStatistics.py \
  --input 'interpolatedData/*.npz' \
  --output statisticsData \
  --re 5600 --re-tau 280
```

The same ordered multi-run input syntax extends the statistics window:

```bash
mpiexec -n 8 python extractStatistics.py \
  --input 'run7/interpolatedData/*.npz' \
          'run8/interpolatedData/*.npz' \
          'run9/interpolatedData/*.npz' \
  --output statisticsData \
  --re 5600 --re-tau 280
```

Restart overlap and duplicate times are handled identically to
`extractSpectra.py`.

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
- `anisotropyProfile.dat`, `anisotropyProfiles.png`, the barycentric map, and
  the `(-II,III)` turbulence-invariant triangle;
- unconditioned, liquid-conditioned, and gas-conditioned TKE profiles;
- the streamwise mean-velocity profile;
- separate diagonal and off-diagonal Reynolds-stress figures, with the two
  channel halves folded using the appropriate component symmetry;
- conventional channel TKE-budget terms, `tkeBudget.dat`, and
  `tkeBudget.png` for a single-phase data set only.

The conventional budget includes production, dissipation, turbulent
transport, pressure transport, and viscous diffusion. It is generated only
for the constant-property single-phase channel. If gas is detected, the
second budget pass is skipped: no two-phase budget or residual is written.
An otherwise single-phase run containing a frozen nonuniform CLS field is
also detected as two phase. Its unconditioned velocity statistics remain
usable, but its CLS-conditioned "gas" statistics are only mask-conditioned
diagnostics and a conventional single-phase budget is not generated.

All figures can be regenerated from the saved arrays without rereading the
interpolated snapshots:

```bash
python ../extractStatistics.py --plot-only \
  --statistics statisticsData/statistics.npz \
  --output statisticsData
```

Multiple saved statistics files can be compared directly:

```bash
python extractStatistics.py --plot-only \
  --statistics singlePhase/p9/statisticsData/statistics.npz \
               hbase/p9/statisticsData/statistics.npz \
  --labels 'single phase' 'two phase' \
  --output statisticsComparison
```

The comparison includes unconditioned TKE, streamwise mean velocity versus
`y/delta`, diagonal and off-diagonal Reynolds stresses, anisotropy profiles,
Lumley flatness, barycentric and turbulence-invariant triangles, and
temperature statistics when every input contains temperature. Triangle color
represents `y+`; marker shape identifies the case. The triangle trajectories
use one channel half to avoid plotting duplicate upper- and lower-wall paths.
No MPI launcher is needed in `--plot-only` mode.

With checkpoints restricted to `u`, `v`, `w`, `p`, the dedicated temperature
field `T`, and the two passive scalars TLS and CLS, the two-phase-only terms
are accumulated in situ by `channel.udf`. They are
first x-z averaged on the device and then time averaged as one-dimensional
wall-normal profiles. At each normal checkpoint event an interval block is
appended to `budgetProfiles.dat` and the interval accumulator is reset. No
additional full-mesh field file is written, and the restart layout is
unchanged.

## 5. Plot and combine the in-situ two-phase profiles

Run the plotting commands in this section from the `bubblyFlow` directory.

Each block in `budgetProfiles.dat` is already x-z planar averaged and time
averaged over the duration recorded in its interval header. An input directory
automatically expands to every `budgetProfiles*.dat` file in natural numeric
order. Use `--t-min` to select the statistically stationary interval:

```bash
python plotBudgetProfiles.py \
  --input run1 \
  --labels 'N=7' \
  --t-min 20 \
  --output budgetProfilePlots
```

Explicit files and `+`-separated files or directories remain supported, for
example `run1/budgetProfiles2.dat+run2`. Later files replace overlapping
intervals from earlier files. Several meshes can be compared without averaging
them together:

```bash
python plotBudgetProfiles.py \
  --input hbase/p7 hbase/p9 hrefine/p7 \
  --labels 'N=7' 'N=9' 'h/2, N=7' \
  --t-min 20 \
  --output budgetProfilePlots
```

Newly generated files also contain mean and RMS temperature, conductivity,
volumetric heat capacity, velocity--temperature covariance, and convective
enthalpy-flux profiles. Legacy and current files may be combined in one `+`
group even when their headers differ. Fields are aligned by name: shared
mechanical quantities use every selected interval, while thermal quantities
use only intervals from files that contain them. The script prints any shorter
field-specific averaging durations. Old files cannot supply thermal profiles
retroactively.

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
- total surface-tension work;
- time-averaged viscous dissipation and all six viscous stresses;
- pressure-gradient and gravity power.

The column ordering in `budgetProfiles.dat` is fixed by `channel.udf` and
checked by the reducer. Legacy layouts containing capillary-force components
and velocity--capillary correlations remain readable, but those redundant
columns are omitted from newly generated files. Do not combine different
layouts in one reduction.

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

Run these comparison commands from the `bubblyFlow` directory.

Use `plotBubbly.py` to compare the time histories in `bubbly.dat`:

```bash
python plotBubbly.py \
  --input hbase/p7/bubbly.dat hbase/p9/bubbly.dat hrefine/p7/bubbly.dat \
  --labels 'N=7' 'N=9' 'h/2, N=7' \
  --output bubblyPlots
```

Use `plotLog.py` for CHANNEL, THERMAL, y+, timestep, CFL, and per-step runtime
diagnostics. A directory input automatically reads every `log*` file in
natural numeric order (`log9` before `log10`):

```bash
python plotLog.py \
  --input hrefine/p7 \
  --labels 'h/2, N=7' \
  --t-min 20 \
  --output logPlots
```

Explicit logs and `+`-separated files or directories remain supported.

## 8. Compare single- and two-phase cases

After processing each case separately:

```bash
python ../compareCases.py \
  --input singlePhase/spectraData twoPhase/spectraData \
  --labels 'single phase' 'bubbly' \
  --target-yplus 12 280 \
  --output comparisonPlots
```

Each input can be either a spectra directory or its `spectra.npz` file. The
spectral comparison averages matching lower- and upper-half channel planes.
It writes one premultiplied component comparison containing `u`, `v`, and `w`
in consistent colors with a different line style for each input case. It also
writes a non-premultiplied total-TKE spectrum comparison in both homogeneous
directions at every requested wall distance. Both use the same relative
high-wavenumber noise cutoff as `extractSpectra.py`.
Add `--single-statistics` and `--two-statistics` to also compare liquid TKE,
`b11`, Lumley flatness, and barycentric-map trajectories. The original
`--single-spectra` and `--two-spectra` interface remains supported.

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
