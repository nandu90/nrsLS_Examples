"""Extract wall-normal-resolved channel spectra from interpolated snapshots.

One-dimensional FFTs are applied only in the homogeneous periodic directions.
The saved component spectra integrate to the corresponding plane variance;
the TKE spectrum is one half the sum of the three component spectra.
"""

import argparse
import glob
import os
import time

from mpi4py import MPI
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="interpolatedData/*.npz")
    parser.add_argument("--output", default="spectraData")
    parser.add_argument("--re-tau", type=float, default=280.0)
    parser.add_argument("--target-yplus", type=float, nargs="*",
                        default=[12.0, 280.0])
    parser.add_argument("--phase-filter-count", type=int, default=48,
                        help="log-spaced filter cutoffs for phase spectra")
    parser.add_argument("--phase-tolerance", type=float, default=1.0e-3,
                        help="minimum plane phase fraction")
    return parser.parse_args()


def one_sided_power(field, axis):
    """Return one-sided FFT power with Parseval-consistent normalization."""
    n = field.shape[axis]
    transformed = np.fft.rfft(field, axis=axis)/n
    power = np.abs(transformed)**2
    index = [slice(None)]*power.ndim
    if n % 2 == 0:
        index[axis] = slice(1, -1)
    else:
        index[axis] = slice(1, None)
    power[tuple(index)] *= 2.0
    return power


def spectra_for_snapshot(data):
    velocity = np.stack((data["u"], data["v"], data["w"]), axis=-1)
    fluctuations = velocity - np.mean(
        velocity, axis=(0, 2), keepdims=True)

    # E_x(y,kx,component): average the x-transform over z.
    ex = np.mean(one_sided_power(fluctuations, axis=0), axis=2)
    ex = np.transpose(ex, (1, 0, 2))

    # E_z(y,kz,component): average the z-transform over x.
    ez = np.mean(one_sided_power(fluctuations, axis=2), axis=0)
    ez = np.transpose(ez, (0, 1, 2))

    variance = np.mean(fluctuations**2, axis=(0, 2))
    return ex, ez, variance


def scalar_spectra_for_snapshot(field):
    fluctuations = field - np.mean(field, axis=(0, 2), keepdims=True)
    ex = np.mean(one_sided_power(fluctuations, axis=0), axis=2).T
    ez = np.mean(one_sided_power(fluctuations, axis=2), axis=0)
    variance = np.mean(fluctuations**2, axis=(0, 2))
    return ex, ez, variance


def wavenumber_spacing(wave):
    """Return the constant spacing of a uniformly sampled FFT grid."""
    if wave.size < 2:
        raise ValueError("At least two wavenumbers are required")
    spacing = float(wave[1] - wave[0])
    if spacing <= 0.0 or not np.allclose(np.diff(wave), spacing):
        raise ValueError("Expected uniformly spaced increasing wavenumbers")
    return spacing


def matching_half_indices(y, yplus, target):
    """Return matching lower/upper-half indices for a target wall distance."""
    selected = []
    for mask in (y <= 0.0, y >= 0.0):
        candidates = np.flatnonzero(mask)
        selected.append(candidates[np.argmin(np.abs(yplus[candidates]
                                               - target))])
    return np.unique(selected)


def filtered_phase_spectrum(velocity, fraction, wave, cutoffs, axis,
                            tolerance):
    """HIT-style phase spectrum from differentiated filtered energy.

    The input is one x-z plane: velocity has shape (nx,nz,3), fraction has
    shape (nx,nz), and axis is 0 for x or 1 for z.  Constant phase density
    cancels because the returned energy is specific to that phase.
    """
    alpha = float(np.mean(fraction))
    if alpha < tolerance:
        return np.zeros_like(cutoffs), 0.0, alpha

    weight = np.sum(fraction)
    phase_mean = np.sum(
        fraction[..., None]*velocity, axis=(0, 1))/weight
    fluctuation = velocity - phase_mean
    phase_tke = 0.5*np.sum(
        np.mean(fraction[..., None]*fluctuation*fluctuation,
                axis=(0, 1)))/alpha

    weighted_fields = np.concatenate(
        (fraction[..., None], fraction[..., None]*fluctuation), axis=-1)
    transformed = np.fft.fft(weighted_fields, axis=axis)
    signed_wave = 2.0*np.pi*np.fft.fftfreq(
        fraction.shape[axis],
        d=2.0*np.pi/(fraction.shape[axis]*wavenumber_spacing(wave)))
    transfer_shape = [1]*weighted_fields.ndim
    transfer_shape[axis] = signed_wave.size

    cumulative = np.zeros_like(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        width = 2.0*np.pi/cutoff
        sigma = width/(2.0*np.sqrt(2.0*np.log(2.0)))
        transfer = np.exp(-0.5*sigma*sigma*signed_wave*signed_wave)
        filtered = np.fft.ifft(
            transformed*transfer.reshape(transfer_shape), axis=axis).real
        filtered_fraction = filtered[..., 0]
        # The user tolerance decides whether the phase has enough plane
        # occupancy to report a spectrum.  Once accepted, retain all positive
        # filtered mass here so that the spectrum is not biased by clipping
        # low-concentration regions at that much larger threshold.
        valid = filtered_fraction > 1.0e-14
        momentum_squared = np.sum(filtered[..., 1:]**2, axis=-1)
        resolved = np.divide(
            momentum_squared,
            filtered_fraction,
            out=np.zeros_like(filtered_fraction),
            where=valid)
        cumulative[i] = 0.5*np.mean(resolved)/alpha

    spectrum = np.gradient(cumulative, cutoffs)
    # Small negative values can arise from finite sampling and differentiation.
    spectrum = np.maximum(spectrum, 0.0)
    return spectrum, phase_tke, alpha


def plot_phase_spectra(output_dir, target_yplus, actual_yplus,
                       kx_filter, kz_filter, ex_liquid, ex_gas,
                       ez_liquid, ez_gas, tke_liquid, tke_gas):
    colors = ("#0072B2", "#D55E00")
    for target_index, target in enumerate(target_yplus):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, wave, liquid, gas, direction in (
                (axes[0], kx_filter, ex_liquid[target_index],
                 ex_gas[target_index], "x"),
                (axes[1], kz_filter, ez_liquid[target_index],
                 ez_gas[target_index], "z")):
            for spectrum, normalizer, label, color, style in (
                    (liquid, tke_liquid[target_index], "liquid",
                     colors[0], "-"),
                    (gas, tke_gas[target_index], "gas",
                     colors[1], "--")):
                if normalizer > 1.0e-30:
                    premultiplied = wave*spectrum/normalizer
                    axis.semilogx(
                        wave, premultiplied,
                        color=color, linestyle=style, label=label)
            axis.set_xlabel(r"$k_%s\delta$" % direction)
            axis.set_ylabel(r"$k_%s E_{m}/k_m$" % direction)
            axis.grid(True, which="both", alpha=0.35)
        axes[0].legend()
        fig.suptitle(r"filtered phase spectra, $y^+=%.2f$" %
                     actual_yplus[target_index])
        fig.tight_layout()
        tag = ("%.2f" % target).replace(".", "p")
        fig.savefig(os.path.join(
            output_dir, "premultiplied_phase_yplus_%s.png" % tag),
            dpi=200)
        plt.close(fig)


def plot_selected_planes(output_dir, y, yplus, kx, kz, ex, ez,
                         variance, targets):
    component_names = (r"$u'$", r"$v'$", r"$w'$")
    colors = ("#0072B2", "#D55E00", "#009E73")

    for target in targets:
        indices = matching_half_indices(y, yplus, target)
        actual = float(np.mean(yplus[indices]))
        ex_folded = np.mean(ex[indices], axis=0)
        ez_folded = np.mean(ez[indices], axis=0)
        variance_folded = np.mean(variance[indices], axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for spectrum, wave, axis, direction in (
                (ex_folded, kx, axes[0], "x"),
                (ez_folded, kz, axes[1], "z")):
            positive = wave > 0.0
            delta_wave = wavenumber_spacing(wave)

            for component in range(3):
                normalizer = max(variance_folded[component], 1.0e-30)
                premultiplied = (
                    wave[positive]*spectrum[positive, component]
                    /(delta_wave*normalizer))
                axis.semilogx(
                    wave[positive],
                    premultiplied,
                    color=colors[component],
                    label=component_names[component])

            axis.set_xlabel(r"$k_%s\delta$" % direction)
            axis.set_ylabel(r"$k_%s E_{ii}/\langle u_i'^2\rangle$" % direction)
            axis.grid(True, which="both", alpha=0.35)

        axes[0].legend()
        fig.suptitle(r"$y^+=%.2f$" % actual)
        fig.tight_layout()
        tag = ("%.2f" % target).replace(".", "p")
        fig.savefig(os.path.join(output_dir,
                                 "premultiplied_yplus_%s.png" % tag),
                    dpi=200)
        plt.close(fig)


def plot_selected_temperature_planes(output_dir, y, yplus, kx, kz, ex, ez,
                                     variance, targets):
    for target in targets:
        indices = matching_half_indices(y, yplus, target)
        actual = float(np.mean(yplus[indices]))
        ex_folded = np.mean(ex[indices], axis=0)
        ez_folded = np.mean(ez[indices], axis=0)
        variance_folded = float(np.mean(variance[indices]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for spectrum, wave, axis, direction in (
                (ex_folded, kx, axes[0], "x"),
                (ez_folded, kz, axes[1], "z")):
            positive = wave > 0.0
            delta_wave = wavenumber_spacing(wave)
            normalizer = max(variance_folded, 1.0e-30)
            premultiplied = (wave[positive]*spectrum[positive]
                             /(delta_wave*normalizer))
            axis.semilogx(wave[positive], premultiplied,
                          color="#CC0000")
            axis.set_xlabel(r"$k_%s\delta$" % direction)
            axis.set_ylabel(r"$k_%s E_{TT}/\langle T'^2\rangle$" %
                            direction)
            axis.grid(True, which="both", alpha=0.35)
        fig.suptitle(r"temperature, $y^+=%.2f$" % actual)
        fig.tight_layout()
        tag = ("%.2f" % target).replace(".", "p")
        fig.savefig(os.path.join(
            output_dir, "premultiplied_temperature_yplus_%s.png" % tag),
            dpi=200)
        plt.close(fig)


def main():
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    input_files = sorted(glob.glob(args.input))
    if not input_files:
        raise RuntimeError("No files match %r" % args.input)

    with np.load(input_files[0]) as first:
        x = first["x"]
        y = first["y"]
        z = first["z"]
        nx, ny, nz = first["u"].shape
        temperature_available = "T" in first.files

    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    kx = 2.0*np.pi*np.fft.rfftfreq(nx, d=dx)
    kz = 2.0*np.pi*np.fft.rfftfreq(nz, d=dz)
    yplus = args.re_tau*(1.0 - np.abs(y))
    target_plane_indices = [
        matching_half_indices(y, yplus, target)
        for target in args.target_yplus]
    target_actual_yplus = np.array([
        np.mean(yplus[indices]) for indices in target_plane_indices])
    if args.phase_filter_count < 3:
        raise ValueError("phase-filter-count must be at least three")
    kx_filter = np.logspace(
        np.log10(0.5*wavenumber_spacing(kx)),
        np.log10(kx[-1]), args.phase_filter_count)
    kz_filter = np.logspace(
        np.log10(0.5*wavenumber_spacing(kz)),
        np.log10(kz[-1]), args.phase_filter_count)

    local_ex = np.zeros((ny, kx.size, 3))
    local_ez = np.zeros((ny, kz.size, 3))
    local_variance = np.zeros((ny, 3))
    if temperature_available:
        local_ex_temperature = np.zeros((ny, kx.size))
        local_ez_temperature = np.zeros((ny, kz.size))
        local_temperature_variance = np.zeros(ny)
    number_targets = len(target_plane_indices)
    phase_shape = (number_targets, args.phase_filter_count)
    local_ex_liquid = np.zeros(phase_shape)
    local_ex_gas = np.zeros(phase_shape)
    local_ez_liquid = np.zeros(phase_shape)
    local_ez_gas = np.zeros(phase_shape)
    local_tke_liquid = np.zeros(number_targets)
    local_tke_gas = np.zeros(number_targets)
    local_alpha_liquid = np.zeros(number_targets)
    local_alpha_gas = np.zeros(number_targets)
    local_liquid_count = np.zeros(number_targets)
    local_gas_count = np.zeros(number_targets)
    local_plane_count = np.zeros(number_targets)
    local_times = []

    for filename in input_files[rank::size]:
        with np.load(filename) as data:
            if data["u"].shape != (nx, ny, nz):
                raise RuntimeError("Grid mismatch in %s" % filename)
            ex, ez, variance = spectra_for_snapshot(data)
            local_ex += ex
            local_ez += ez
            local_variance += variance
            cls = np.clip(data["cls"], 0.0, 1.0)
            velocity = np.stack(
                (data["u"], data["v"], data["w"]), axis=-1)
            for target_index, plane_indices in enumerate(
                    target_plane_indices):
                for iy in plane_indices:
                    plane_velocity = velocity[:, iy, :, :]
                    liquid_fraction = cls[:, iy, :]
                    gas_fraction = 1.0 - liquid_fraction
                    local_plane_count[target_index] += 1.0
                    phase_data = (
                        (liquid_fraction, local_ex_liquid, local_ez_liquid,
                         local_tke_liquid, local_alpha_liquid,
                         local_liquid_count),
                        (gas_fraction, local_ex_gas, local_ez_gas,
                         local_tke_gas, local_alpha_gas,
                         local_gas_count))
                    for (fraction, ex_phase, ez_phase, tke_phase,
                         alpha_phase, phase_count) in phase_data:
                        spectrum_x, phase_tke, alpha = \
                            filtered_phase_spectrum(
                                plane_velocity, fraction, kx, kx_filter, 0,
                                args.phase_tolerance)
                        spectrum_z, _, _ = filtered_phase_spectrum(
                            plane_velocity, fraction, kz, kz_filter, 1,
                            args.phase_tolerance)
                        ex_phase[target_index] += spectrum_x
                        ez_phase[target_index] += spectrum_z
                        tke_phase[target_index] += phase_tke
                        alpha_phase[target_index] += alpha
                        if alpha >= args.phase_tolerance:
                            phase_count[target_index] += 1.0
            if temperature_available:
                if "T" not in data.files:
                    raise RuntimeError("Temperature is missing from %s" %
                                       filename)
                ex_t, ez_t, variance_t = scalar_spectra_for_snapshot(data["T"])
                local_ex_temperature += ex_t
                local_ez_temperature += ez_t
                local_temperature_variance += variance_t
            local_times.append(float(data["time"]))

    count = np.array(len(local_times), dtype=np.int64)
    total_count = np.array(0, dtype=np.int64)
    comm.Reduce(count, total_count, op=MPI.SUM, root=0)

    if rank == 0:
        sum_ex = np.empty_like(local_ex)
        sum_ez = np.empty_like(local_ez)
        sum_variance = np.empty_like(local_variance)
    else:
        sum_ex = sum_ez = sum_variance = None

    comm.Reduce(local_ex, sum_ex, op=MPI.SUM, root=0)
    comm.Reduce(local_ez, sum_ez, op=MPI.SUM, root=0)
    comm.Reduce(local_variance, sum_variance, op=MPI.SUM, root=0)
    phase_local_arrays = (
        local_ex_liquid, local_ex_gas, local_ez_liquid, local_ez_gas,
        local_tke_liquid, local_tke_gas,
        local_alpha_liquid, local_alpha_gas,
        local_liquid_count, local_gas_count, local_plane_count)
    if rank == 0:
        phase_sum_arrays = tuple(np.empty_like(value)
                                 for value in phase_local_arrays)
    else:
        phase_sum_arrays = (None,)*len(phase_local_arrays)
    for local_value, sum_value in zip(phase_local_arrays, phase_sum_arrays):
        comm.Reduce(local_value, sum_value, op=MPI.SUM, root=0)
    if temperature_available:
        if rank == 0:
            sum_ex_temperature = np.empty_like(local_ex_temperature)
            sum_ez_temperature = np.empty_like(local_ez_temperature)
            sum_temperature_variance = np.empty_like(
                local_temperature_variance)
        else:
            sum_ex_temperature = None
            sum_ez_temperature = None
            sum_temperature_variance = None
        comm.Reduce(local_ex_temperature, sum_ex_temperature,
                    op=MPI.SUM, root=0)
        comm.Reduce(local_ez_temperature, sum_ez_temperature,
                    op=MPI.SUM, root=0)
        comm.Reduce(local_temperature_variance, sum_temperature_variance,
                    op=MPI.SUM, root=0)
    gathered_times = comm.gather(local_times, root=0)

    if rank != 0:
        return

    if total_count == 0:
        raise RuntimeError("No snapshots were processed")
    ex = sum_ex/int(total_count)
    ez = sum_ez/int(total_count)
    variance = sum_variance/int(total_count)
    times = np.sort(np.concatenate([np.asarray(t) for t in gathered_times]))
    (sum_ex_liquid, sum_ex_gas, sum_ez_liquid, sum_ez_gas,
     sum_tke_liquid, sum_tke_gas,
     sum_alpha_liquid, sum_alpha_gas,
     sum_liquid_count, sum_gas_count,
     sum_plane_count) = phase_sum_arrays
    liquid_count = np.maximum(sum_liquid_count, 1.0)
    gas_count = np.maximum(sum_gas_count, 1.0)
    ex_liquid = sum_ex_liquid/liquid_count[:, None]
    ex_gas = sum_ex_gas/gas_count[:, None]
    ez_liquid = sum_ez_liquid/liquid_count[:, None]
    ez_gas = sum_ez_gas/gas_count[:, None]
    tke_liquid = sum_tke_liquid/liquid_count
    tke_gas = sum_tke_gas/gas_count
    plane_count = np.maximum(sum_plane_count, 1.0)
    alpha_liquid = sum_alpha_liquid/plane_count
    alpha_gas = sum_alpha_gas/plane_count

    os.makedirs(args.output, exist_ok=True)
    output = dict(
        time=times,
        x=x, y=y, z=z, yplus=yplus,
        kx=kx, kz=kz,
        Ex=ex, Ez=ez,
        variance=variance,
        tke=0.5*np.sum(variance, axis=1),
        reTau=np.array(args.re_tau),
        phaseTargetYplus=np.asarray(args.target_yplus),
        phaseActualYplus=target_actual_yplus,
        phaseLowerYIndex=np.array([indices[0]
                                  for indices in target_plane_indices]),
        phaseUpperYIndex=np.array([indices[-1]
                                  for indices in target_plane_indices]),
        kxPhaseFilter=kx_filter,
        kzPhaseFilter=kz_filter,
        ExLiquidFiltered=ex_liquid,
        ExGasFiltered=ex_gas,
        EzLiquidFiltered=ez_liquid,
        EzGasFiltered=ez_gas,
        liquidPhaseTke=tke_liquid,
        gasPhaseTke=tke_gas,
        alphaLiquidAtTargets=alpha_liquid,
        alphaGasAtTargets=alpha_gas,
        liquidPhaseSampleCount=sum_liquid_count,
        gasPhaseSampleCount=sum_gas_count,
        phasePlaneSampleCount=sum_plane_count,
    )
    if temperature_available:
        ex_temperature = sum_ex_temperature/int(total_count)
        ez_temperature = sum_ez_temperature/int(total_count)
        temperature_variance = sum_temperature_variance/int(total_count)
        output.update(
            ExTemperature=ex_temperature,
            EzTemperature=ez_temperature,
            temperatureVariance=temperature_variance,
        )
    np.savez(os.path.join(args.output, "spectra.npz"), **output)

    # Parseval check, excluding the deliberately removed plane-mean mode.
    error_x = np.max(np.abs(np.sum(ex, axis=1) - variance))
    error_z = np.max(np.abs(np.sum(ez, axis=1) - variance))
    print("Maximum Parseval error: x=%g z=%g" % (error_x, error_z))

    plot_selected_planes(args.output, y, yplus, kx, kz,
                         ex, ez, variance, args.target_yplus)
    plot_phase_spectra(
        args.output, args.target_yplus, target_actual_yplus,
        kx_filter, kz_filter, ex_liquid, ex_gas, ez_liquid, ez_gas,
        tke_liquid, tke_gas)
    if temperature_available:
        error_tx = np.max(
            np.abs(np.sum(ex_temperature, axis=1) - temperature_variance))
        error_tz = np.max(
            np.abs(np.sum(ez_temperature, axis=1) - temperature_variance))
        print("Temperature Parseval error: x=%g z=%g" %
              (error_tx, error_tz))
        plot_selected_temperature_planes(
            args.output, y, yplus, kx, kz, ex_temperature, ez_temperature,
            temperature_variance, args.target_yplus)
    print("Wrote spectra to %s" % args.output)


if __name__ == "__main__":
    start = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- spectra completed in %.2f s ---" % (time.time() - start))
