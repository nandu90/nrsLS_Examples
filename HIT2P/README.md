# HIT2P: forced two-phase homogeneous isotropic turbulence

This case is based on the `damBreak` NekRS-LS setup and represents case D2
of Jain and Elnahhas (2025): a light phase initially placed as a sphere in a
triply periodic turbulent carrier.

Set `singlePhase = true` near the top of `hit2p.udf` to generate the
single-phase HIT precursor. In this mode `cls=1`, surface tension and level-set
evolution are disabled, and Eq. (25) reduces to the single-phase forcing in
Eq. (21). After the precursor becomes stationary, checkpoint it, set
`singlePhase = false`, and restart. A uniform precursor `cls` field is then
replaced by the centered bubble while the restarted velocity is retained.

## Physical and numerical parameters

- Domain: genbox mesh `[-1,1]^3`, rescaled in `UDF_Setup` to
  `[-pi,pi]^3`; periodic in all directions.
- Initial sphere: center `(0,0,0)`, radius `pi/2`, volume fraction `pi/48`.
- Density: `rho_inside/rho_carrier = 0.1`.
- Dynamic viscosity: `mu_inside/mu_carrier = 1`.
- Target carrier-phase TKE: `k_c = 1`.
- `Re = 518`, estimated for the stationary target `Re_lambda = 87` using
  the linear-forcing integral scale `0.19 L` reported by Bassenne et al.
- NekRS conventional `We = 29.25`, equivalent to the paper's `We_L = 19.5`
  because its definition contains a factor `2/3`.
- Gravity is disabled.

The linear forcing is smoothly masked by `cls` and therefore acts only in
the outer carrier phase. It is applied to the carrier fluctuation relative
to the intrinsic carrier-volume mean. Its coefficient is evaluated from
Eq. (25) using the instantaneous carrier-phase TKE, dissipation, pressure
transport, and viscous transport, together with the exponential-relaxation
controller. The controller uses `G = 67` and
`t_l = 0.19 L/sqrt(2 k_c/3)` following Bassenne et al.

## Build and run

Generate the mesh using the Nek5000 `genbox` utility:

```bash
genbox <<EOF
hit2p.box
EOF
```

Then build or run with the NekRS-LS installation used by the other examples:

```bash
nrsmpi hit2p 1 --build-only
nrsmpi hit2p 1
```

For a production calculation, use enough MPI ranks for the 32^3-element,
order-7 mesh. Runtime output reports carrier TKE, dissipation, forcing power,
Taylor-scale Reynolds number, measured integral length, and the three
carrier-mean velocity components.

The reproducible startup field approximates the random-phase, solenoidal model
spectrum of Bassenne et al. using 100 Fourier modes over wavenumber shells
`1 <= kappa <= 25`, with the spectrum peaking at `kappa_0 = 12.5`. It is
numerically normalized to `k = 1`. For closest agreement with the paper, first
generate a statistically stationary single-phase HIT field for about 30
turnover times, then use that velocity as the restart state while initializing
the same centered sphere.
