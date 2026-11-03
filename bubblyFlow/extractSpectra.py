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


def plot_selected_planes(output_dir, y, yplus, kx, kz, ex, ez,
                         variance, targets, re_tau):
    component_names = (r"$u'$", r"$v'$", r"$w'$")
    colors = ("#0072B2", "#D55E00", "#009E73")

    for target in targets:
        iy = int(np.argmin(np.abs(yplus - target)))
        actual = yplus[iy]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for spectrum, wave, axis, direction in (
                (ex[iy], kx, axes[0], "x"),
                (ez[iy], kz, axes[1], "z")):
            positive = wave > 0.0
            wavelength_plus = 2.0*np.pi*re_tau/wave[positive]
            order = np.argsort(wavelength_plus)

            for component in range(3):
                normalizer = max(variance[iy, component], 1.0e-30)
                premultiplied = (
                    wave[positive]*spectrum[positive, component]/normalizer)
                axis.semilogx(
                    wavelength_plus[order],
                    premultiplied[order],
                    color=colors[component],
                    label=component_names[component])

            axis.set_xlabel(r"$\lambda_%s^+$" % direction)
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


def plot_selected_temperature_planes(output_dir, yplus, kx, kz, ex, ez,
                                     variance, targets, re_tau):
    for target in targets:
        iy = int(np.argmin(np.abs(yplus - target)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for spectrum, wave, axis, direction in (
                (ex[iy], kx, axes[0], "x"),
                (ez[iy], kz, axes[1], "z")):
            positive = wave > 0.0
            wavelength_plus = 2.0*np.pi*re_tau/wave[positive]
            order = np.argsort(wavelength_plus)
            normalizer = max(variance[iy], 1.0e-30)
            premultiplied = wave[positive]*spectrum[positive]/normalizer
            axis.semilogx(wavelength_plus[order], premultiplied[order],
                          color="#CC0000")
            axis.set_xlabel(r"$\lambda_%s^+$" % direction)
            axis.set_ylabel(r"$k_%s E_{TT}/\langle T'^2\rangle$" %
                            direction)
            axis.grid(True, which="both", alpha=0.35)
        fig.suptitle(r"temperature, $y^+=%.2f$" % yplus[iy])
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

    local_ex = np.zeros((ny, kx.size, 3))
    local_ez = np.zeros((ny, kz.size, 3))
    local_variance = np.zeros((ny, 3))
    if temperature_available:
        local_ex_temperature = np.zeros((ny, kx.size))
        local_ez_temperature = np.zeros((ny, kz.size))
        local_temperature_variance = np.zeros(ny)
    local_times = []

    for filename in input_files[rank::size]:
        with np.load(filename) as data:
            if data["u"].shape != (nx, ny, nz):
                raise RuntimeError("Grid mismatch in %s" % filename)
            ex, ez, variance = spectra_for_snapshot(data)
            local_ex += ex
            local_ez += ez
            local_variance += variance
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
    yplus = args.re_tau*(1.0 - np.abs(y))

    os.makedirs(args.output, exist_ok=True)
    output = dict(
        time=times,
        x=x, y=y, z=z, yplus=yplus,
        kx=kx, kz=kz,
        Ex=ex, Ez=ez,
        variance=variance,
        tke=0.5*np.sum(variance, axis=1),
        reTau=np.array(args.re_tau),
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
                         ex, ez, variance, args.target_yplus, args.re_tau)
    if temperature_available:
        error_tx = np.max(
            np.abs(np.sum(ex_temperature, axis=1) - temperature_variance))
        error_tz = np.max(
            np.abs(np.sum(ez_temperature, axis=1) - temperature_variance))
        print("Temperature Parseval error: x=%g z=%g" %
              (error_tx, error_tz))
        plot_selected_temperature_planes(
            args.output, yplus, kx, kz, ex_temperature, ez_temperature,
            temperature_variance, args.target_yplus, args.re_tau)
    print("Wrote spectra to %s" % args.output)


if __name__ == "__main__":
    start = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- spectra completed in %.2f s ---" % (time.time() - start))
