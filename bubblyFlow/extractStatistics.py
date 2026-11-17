"""Extract channel statistics, phase anisotropy, and SP TKE budgets.

Phase statistics use intrinsic CLS-weighted averages.  A conventional TKE
budget is generated only for a constant-property single-phase channel.  With
the available checkpoint fields, a complete two-phase budget cannot be
reconstructed and is therefore deliberately not produced.
"""

import argparse
import glob
import os
import re
import time

from mpi4py import MPI
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", default=["interpolatedData/*.npz"],
        help=("One or more chronological snapshot globs, files, or "
              "directories. Later inputs replace overlapping restart times"))
    parser.add_argument("--output", default="statisticsData")
    parser.add_argument("--t-min", type=float, default=None,
                        help="Ignore snapshots with time below this value")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Ignore snapshots with time above this value")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate plots from an existing statistics.npz file")
    parser.add_argument(
        "--statistics", nargs="+", default=None,
        help=("One or more statistics.npz files used with --plot-only; "
              "defaults to OUTPUT/statistics.npz"))
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Case labels for multiple --statistics files")
    parser.add_argument("--re", type=float, default=5600.0,
                        help="Re based on full channel height")
    parser.add_argument("--re-tau", type=float, default=280.0)
    return parser.parse_args()


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path)]


def snapshot_time(filename):
    with np.load(filename) as data:
        return float(data["time"])


def resolve_snapshot_inputs(input_specifications, t_min=None, t_max=None):
    """Expand and stitch chronological restart segments by snapshot time."""
    stitched = []
    for specification in input_specifications:
        if os.path.isdir(specification):
            matches = glob.glob(os.path.join(specification, "*.npz"))
        else:
            matches = glob.glob(specification)
        matches = sorted((name for name in matches if os.path.isfile(name)),
                         key=natural_sort_key)
        if not matches:
            raise RuntimeError("No files match %r" % specification)
        segment = sorted(((snapshot_time(name), name) for name in matches),
                         key=lambda item: (item[0], natural_sort_key(item[1])))
        restart_time = segment[0][0]
        tolerance = 1.0e-12*max(1.0, abs(restart_time))
        stitched = [item for item in stitched
                    if item[0] < restart_time - tolerance]
        stitched.extend(segment)

    by_time = {sample_time: filename for sample_time, filename in stitched}
    selected_times = [sample_time for sample_time in sorted(by_time)
                      if (t_min is None or sample_time >= t_min) and
                      (t_max is None or sample_time <= t_max)]
    if not selected_times:
        raise RuntimeError("No snapshots remain in the requested time window")
    return [by_time[sample_time] for sample_time in selected_times]


def periodic_derivative(field, spacing, axis):
    n = field.shape[axis]
    wave = 2.0*np.pi*np.fft.fftfreq(n, d=spacing)
    shape = [1]*field.ndim
    shape[axis] = n
    transformed = np.fft.fft(field, axis=axis)
    return np.fft.ifft(1j*wave.reshape(shape)*transformed,
                       axis=axis).real


def strain_rate(fluctuations, y, dx, dz):
    """Return S'_ij with shape (nx,ny,nz,3,3)."""
    gradient = np.empty(fluctuations.shape + (3,))
    for component in range(3):
        value = fluctuations[..., component]
        gradient[..., component, 0] = periodic_derivative(value, dx, 0)
        gradient[..., component, 1] = np.gradient(
            value, y, axis=1, edge_order=2)
        gradient[..., component, 2] = periodic_derivative(value, dz, 2)
    return 0.5*(gradient + np.swapaxes(gradient, -1, -2))


def phase_results(weight, first, second):
    safe = np.maximum(weight, 1.0e-30)
    mean = first/safe[:, None]
    raw_second = second/safe[:, None, None]
    reynolds = raw_second - mean[:, :, None]*mean[:, None, :]
    reynolds = 0.5*(reynolds + np.swapaxes(reynolds, 1, 2))
    tke = 0.5*np.trace(reynolds, axis1=1, axis2=2)

    anisotropy = np.full_like(reynolds, np.nan)
    ii = np.full(weight.size, np.nan)
    iii = np.full(weight.size, np.nan)
    flatness = np.full(weight.size, np.nan)
    eigenvalues = np.full((weight.size, 3), np.nan)
    barycentric = np.full((weight.size, 3), np.nan)
    bary_x = np.full(weight.size, np.nan)
    bary_y = np.full(weight.size, np.nan)

    for iy in range(weight.size):
        if weight[iy] <= 1.0e-14 or tke[iy] <= 1.0e-14:
            continue
        tensor = reynolds[iy]/(2.0*tke[iy]) - np.eye(3)/3.0
        tensor = 0.5*(tensor + tensor.T)
        anisotropy[iy] = tensor
        ii[iy] = -0.5*np.sum(tensor*tensor.T)
        iii[iy] = np.linalg.det(tensor)
        flatness[iy] = 1.0 + 9.0*ii[iy] + 27.0*iii[iy]

        lam = np.linalg.eigvalsh(tensor)[::-1]
        eigenvalues[iy] = lam
        c1 = lam[0] - lam[1]
        c2 = 2.0*(lam[1] - lam[2])
        c3 = 3.0*lam[2] + 1.0
        barycentric[iy] = (c1, c2, c3)
        bary_x[iy] = c1 + 0.5*c3
        bary_y[iy] = 0.5*np.sqrt(3.0)*c3

    return {
        "mean": mean,
        "reynolds": reynolds,
        "tke": tke,
        "anisotropy": anisotropy,
        "II": ii,
        "III": iii,
        "flatness": flatness,
        "eigenvalues": eigenvalues,
        "barycentric": barycentric,
        "barycentricX": bary_x,
        "barycentricY": bary_y,
    }


def reduce_sum(comm, value):
    result = np.empty_like(value)
    comm.Allreduce(value, result, op=MPI.SUM)
    return result


def phase_temperature_results(weight, first, second,
                              velocity_temperature, velocity_mean):
    safe = np.maximum(weight, 1.0e-30)
    mean = first/safe
    variance = np.maximum(second/safe - mean*mean, 0.0)
    covariance = (velocity_temperature/safe[:, None]
                  - velocity_mean*mean[:, None])
    return {"mean": mean, "variance": variance,
            "rms": np.sqrt(variance), "velocityCovariance": covariance}


def plot_results(output, y, yplus, all_statistics, liquid, gas, budget=None,
                 temperature=None):
    upper = y >= -1.0e-12
    order = np.argsort(yplus[upper])
    yp = yplus[upper][order]

    def symmetric(profile):
        folded = 0.5*(profile + profile[::-1])
        return folded[upper][order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for component, label in enumerate((r"$b_{11}$", r"$b_{22}$", r"$b_{33}$")):
        axes[0].semilogx(
            yp, symmetric(liquid["anisotropy"][:, component, component]),
            label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"$y^+$")
    axes[0].set_ylabel(r"$b_{ii}$, liquid phase")
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].legend()

    axes[1].semilogx(yp, symmetric(liquid["flatness"]), label="liquid")
    if np.any(np.isfinite(gas["flatness"])):
        axes[1].semilogx(yp, symmetric(gas["flatness"]), label="gas")
    axes[1].set_xlabel(r"$y^+$")
    axes[1].set_ylabel(r"Lumley flatness $F$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "anisotropyProfiles.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(yp, symmetric(all_statistics["tke"]),
                label="unconditioned")
    ax.semilogx(yp, symmetric(liquid["tke"]), label="liquid")
    if np.any(np.isfinite(gas["tke"])):
        ax.semilogx(yp, symmetric(gas["tke"]), label="gas")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$k$")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "tkeProfiles.png"), dpi=200)
    plt.close(fig)

    # Freshly accumulated phase statistics use the generic key ``mean``.
    # Statistics reloaded from statistics.npz use the descriptive alias
    # ``meanVelocity``.  Accept both representations so the direct and
    # --plot-only paths share this plotting routine.
    mean_velocity = all_statistics.get("meanVelocity")
    if mean_velocity is None:
        mean_velocity = all_statistics["mean"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_velocity[:, 0], y,
            label=r"$\langle u\rangle$")
    ax.set_xlabel(r"$\langle u\rangle$")
    ax.set_ylabel(r"$y/\delta$")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "meanVelocityProfile.png"), dpi=200)
    plt.close(fig)

    reynolds = all_statistics["reynolds"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for first, second, label in (
            (0, 0, r"$\langle u'u'\rangle$"),
            (1, 1, r"$\langle v'v'\rangle$"),
            (2, 2, r"$\langle w'w'\rangle$")):
        ax.semilogx(yp, symmetric(reynolds[:, first, second]), label=label)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel("Reynolds normal stress")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "reynoldsStressDiagonal.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for first, second, parity, label in (
            (0, 1, -1.0, r"$\langle u'v'\rangle$"),
            (0, 2, 1.0, r"$\langle u'w'\rangle$"),
            (1, 2, -1.0, r"$\langle v'w'\rangle$")):
        folded = 0.5*(reynolds[:, first, second] +
                      parity*reynolds[::-1, first, second])
        ax.semilogx(yp, folded[upper][order], label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel("Reynolds shear stress")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "reynoldsStressOffDiagonal.png"),
                dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0)/2.0, 0.0]
    ax.plot(triangle_x, triangle_y, "k-", linewidth=1.3)
    points = ax.scatter(liquid["barycentricX"], liquid["barycentricY"],
                        c=yplus, cmap="viridis", s=18)
    ax.text(1.02, 0.0, "1C", va="center")
    ax.text(-0.08, 0.0, "2C", va="center")
    ax.text(0.5, np.sqrt(3.0)/2.0 + 0.035, "3C / isotropic",
            ha="center")
    fig.colorbar(points, ax=ax, label=r"$y^+$")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 0.95)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "barycentricLiquid.png"), dpi=200)
    plt.close(fig)

    # Turbulence-invariant triangle in the (-II, III) representation used by
    # Saini and Bolotnov, Nuclear Technology 208 (2022), Fig. 16.  Use one
    # channel half so that the locus corresponds to a single wall-normal path.
    minus_ii = -liquid["II"][upper][order]
    iii = liquid["III"][upper][order]
    valid = np.isfinite(minus_ii) & np.isfinite(iii)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))

    axisymmetric_positive = np.linspace(0.0, 1.0/3.0, 300)
    axisymmetric_negative = np.linspace(0.0, 1.0/12.0, 160)
    coefficient = 2.0/(3.0*np.sqrt(3.0))
    ax.plot(coefficient*axisymmetric_positive**1.5,
            axisymmetric_positive, "k--", linewidth=1.1)
    ax.plot(-coefficient*axisymmetric_negative**1.5,
            axisymmetric_negative, "k--", linewidth=1.1)
    two_component = np.linspace(1.0/12.0, 1.0/3.0, 220)
    ax.plot(two_component/3.0 - 1.0/27.0, two_component,
            "k--", linewidth=1.1)

    points = ax.scatter(iii[valid], minus_ii[valid], c=yp[valid],
                        cmap="viridis", s=24, zorder=3)
    fig.colorbar(points, ax=ax, label=r"$y^+$")
    ax.text(-0.0065, 0.012, "isotropic", fontsize=9)
    ax.text(2.0/27.0, 1.0/3.0, "1C", ha="left", va="bottom",
            fontsize=9)
    ax.text(-1.0/108.0 + 0.001, 1.0/12.0, "2C", ha="left", va="bottom",
            fontsize=9)
    ax.set_xlabel(r"$III$")
    ax.set_ylabel(r"$-II$")
    ax.set_xlim(-0.014, 0.080)
    ax.set_ylim(-0.005, 0.35)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output, "invariantTriangleLiquid.png"),
                dpi=200)
    plt.close(fig)

    if budget is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, label in (("production", "production"),
                            ("dissipation", "dissipation"),
                            ("turbulentTransport", "turbulent transport"),
                            ("pressureTransport", "pressure transport"),
                            ("viscousDiffusion", "viscous diffusion"),
                            ("residual", "resolved residual")):
            ax.plot(yp, symmetric(budget[name]), label=label)
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel("TKE budget term")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output, "tkeBudget.png"), dpi=200)
        plt.close(fig)

    if temperature is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(temperature["all"]["mean"], y,
                     label=r"$\langle T\rangle$")
        axes[0].plot(temperature["all"]["rms"], y,
                     label=r"$T_{rms}$")
        axes[0].set_xlabel("temperature statistic")
        axes[0].set_ylabel(r"$y/\delta$")
        for component, label in enumerate(
                (r"$\langle u'T'\rangle$", r"$\langle v'T'\rangle$",
                 r"$\langle w'T'\rangle$")):
            axes[1].plot(
                temperature["all"]["velocityCovariance"][:, component],
                y, label=label)
        axes[1].set_xlabel("velocity--temperature covariance")
        for axis in axes:
            axis.grid(True, alpha=0.35)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output, "temperatureStatistics.png"),
                    dpi=200)
        plt.close(fig)


def load_saved_statistics(filename):
    """Load the statistics needed to regenerate plots."""
    with np.load(filename) as data:
        y = data["y"]
        yplus = data["yplus"]
        all_statistics = {
            "tke": data["tke"],
            "reynolds": data["reynoldsStress"],
            "meanVelocity": data["meanVelocity"],
        }
        liquid = {
            "tke": data["liquidTke"],
            "anisotropy": data["liquidAnisotropy"],
            "flatness": data["liquidFlatness"],
            "barycentricX": data["liquidBarycentricX"],
            "barycentricY": data["liquidBarycentricY"],
            "II": data["liquidII"],
            "III": data["liquidIII"],
        }
        gas = {
            "tke": data["gasTke"],
            "flatness": data["gasFlatness"],
        }

        budget_names = (
            "production", "dissipation", "turbulentTransport",
            "pressureTransport", "viscousDiffusion", "residual")
        budget = None
        if all("budget" + name[0].upper() + name[1:] in data.files
               for name in budget_names):
            budget = {
                name: data["budget" + name[0].upper() + name[1:]]
                for name in budget_names
            }

        temperature = None
        temperature_keys = (
            "meanTemperature", "temperatureRms", "turbulentHeatFlux")
        if all(name in data.files for name in temperature_keys):
            temperature = {
                "all": {
                    "mean": data["meanTemperature"],
                    "rms": data["temperatureRms"],
                    "velocityCovariance": data["turbulentHeatFlux"],
                }
            }

    return (y, yplus, all_statistics, liquid, gas, budget, temperature)


def plot_saved_statistics(filename, output):
    """Regenerate all available plots without rereading field snapshots."""
    values = load_saved_statistics(filename)
    os.makedirs(output, exist_ok=True)
    plot_results(output, *values)


def plot_saved_comparison(filenames, labels, output):
    """Compare invariant trajectories and TKE from saved statistics."""
    datasets = [load_saved_statistics(filename) for filename in filenames]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ("-", "--", ":", "-.")
    markers = ("o", "s", "^", "D")
    maximum_yplus = max(float(np.nanmax(values[1])) for values in datasets)
    yplus_norm = mpl.colors.Normalize(vmin=0.0, vmax=maximum_yplus)
    yplus_cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    positive = np.linspace(0.0, 1.0/3.0, 300)
    negative = np.linspace(0.0, 1.0/12.0, 160)
    coefficient = 2.0/(3.0*np.sqrt(3.0))
    ax.plot(coefficient*positive**1.5, positive, "k--", linewidth=1.0)
    ax.plot(-coefficient*negative**1.5, negative, "k--", linewidth=1.0)
    two_component = np.linspace(1.0/12.0, 1.0/3.0, 220)
    ax.plot(two_component/3.0 - 1.0/27.0, two_component,
            "k--", linewidth=1.0)
    for index, (values, label) in enumerate(zip(datasets, labels)):
        y, yplus, _, liquid, _, _, _ = values
        upper = y >= -1.0e-12
        order = np.argsort(yplus[upper])
        iii = liquid["III"][upper][order]
        minus_ii = -liquid["II"][upper][order]
        valid = np.isfinite(iii) & np.isfinite(minus_ii)
        ax.scatter(iii[valid], minus_ii[valid],
                   c=yplus[upper][order][valid], cmap=yplus_cmap,
                   norm=yplus_norm, marker=markers[index % len(markers)],
                   s=27, linewidths=0.55, edgecolors="black", label=label,
                   zorder=3)
    ax.text(-0.0065, 0.012, "isotropic", fontsize=9)
    ax.text(2.0/27.0, 1.0/3.0, "1C", ha="left", va="bottom",
            fontsize=9)
    ax.text(-1.0/108.0 + 0.001, 1.0/12.0, "2C", ha="left",
            va="bottom", fontsize=9)
    ax.set_xlabel(r"$III$")
    ax.set_ylabel(r"$-II$")
    ax.set_xlim(-0.014, 0.080)
    ax.set_ylim(-0.005, 0.35)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.colorbar(mpl.cm.ScalarMappable(norm=yplus_norm, cmap=yplus_cmap),
                 ax=ax, label=r"$y^+$")
    fig.tight_layout()
    fig.savefig(os.path.join(output, "invariantTriangleComparison.png"),
                dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for index, (values, label) in enumerate(zip(datasets, labels)):
        y, yplus, all_statistics, _, _, _, _ = values
        upper = y >= -1.0e-12
        order = np.argsort(yplus[upper])
        folded_tke = 0.5*(all_statistics["tke"] +
                          all_statistics["tke"][::-1])
        ax.semilogx(
            yplus[upper][order], folded_tke[upper][order], label=label,
            color=colors[index % len(colors)],
            linestyle=linestyles[index % len(linestyles)], linewidth=2.0)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"unconditioned $k$")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "tkeComparison.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for case_index, (values, label) in enumerate(zip(datasets, labels)):
        y, _, all_statistics, _, _, _, _ = values
        ax.plot(
            all_statistics["meanVelocity"][:, 0], y, label=label,
            color=colors[case_index % len(colors)],
            linestyle=linestyles[case_index % len(linestyles)],
            linewidth=2.0)
    ax.set_xlabel(r"$\langle u\rangle$")
    ax.set_ylabel(r"$y/\delta$")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "meanVelocityComparison.png"), dpi=200)
    plt.close(fig)

    stress_components = (
        (0, 0, r"$\langle u'u'\rangle$"),
        (1, 1, r"$\langle v'v'\rangle$"),
        (2, 2, r"$\langle w'w'\rangle$"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for case_index, (values, case_label) in enumerate(zip(datasets, labels)):
        y, yplus, all_statistics, _, _, _, _ = values
        upper = y >= -1.0e-12
        order = np.argsort(yplus[upper])
        reynolds = all_statistics["reynolds"]
        for component_index, (first, second, component_label) in enumerate(
                stress_components):
            folded = 0.5*(reynolds[:, first, second] +
                          reynolds[::-1, first, second])
            ax.semilogx(
                yplus[upper][order], folded[upper][order],
                color=colors[component_index % len(colors)],
                linestyle=linestyles[case_index % len(linestyles)],
                label="%s, %s" % (component_label, case_label))
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel("Reynolds normal stress")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output, "reynoldsStressDiagonalComparison.png"),
        dpi=200)
    plt.close(fig)

    shear_components = (
        (0, 1, -1.0, r"$\langle u'v'\rangle$"),
        (0, 2, 1.0, r"$\langle u'w'\rangle$"),
        (1, 2, -1.0, r"$\langle v'w'\rangle$"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for case_index, (values, case_label) in enumerate(zip(datasets, labels)):
        y, yplus, all_statistics, _, _, _, _ = values
        upper = y >= -1.0e-12
        order = np.argsort(yplus[upper])
        reynolds = all_statistics["reynolds"]
        for component_index, (first, second, parity,
                              component_label) in enumerate(shear_components):
            folded = 0.5*(reynolds[:, first, second] +
                          parity*reynolds[::-1, first, second])
            ax.semilogx(
                yplus[upper][order], folded[upper][order],
                color=colors[component_index % len(colors)],
                linestyle=linestyles[case_index % len(linestyles)],
                label="%s, %s" % (component_label, case_label))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel("Reynolds shear stress")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output, "reynoldsStressOffDiagonalComparison.png"),
        dpi=200)
    plt.close(fig)

    component_colors = colors[:3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for case_index, (values, label) in enumerate(zip(datasets, labels)):
        y, yplus, _, liquid, gas, _, _ = values
        upper = y >= -1.0e-12
        order = np.argsort(yplus[upper])
        yp = yplus[upper][order]

        def symmetric(profile):
            return (0.5*(profile + profile[::-1]))[upper][order]

        for component, component_label in enumerate(
                (r"$b_{11}$", r"$b_{22}$", r"$b_{33}$")):
            axes[0].semilogx(
                yp, symmetric(liquid["anisotropy"][:, component, component]),
                color=component_colors[component],
                linestyle=linestyles[case_index % len(linestyles)],
                label="%s, %s" % (component_label, label))
        axes[1].semilogx(
            yp, symmetric(liquid["flatness"]),
            color=colors[case_index % len(colors)], linestyle="-",
            label="liquid, %s" % label)
        if np.any(np.isfinite(gas["flatness"])):
            axes[1].semilogx(
                yp, symmetric(gas["flatness"]),
                color=colors[case_index % len(colors)], linestyle=":",
                label="gas, %s" % label)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"$y^+$")
    axes[0].set_ylabel(r"$b_{ii}$")
    axes[1].set_xlabel(r"$y^+$")
    axes[1].set_ylabel(r"Lumley flatness $F$")
    axes[1].set_ylim(-0.05, 1.05)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.35)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output, "anisotropyProfilesComparison.png"),
                dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0)/2.0, 0.0]
    ax.plot(triangle_x, triangle_y, "k-", linewidth=1.3)
    for case_index, (values, label) in enumerate(zip(datasets, labels)):
        y, yplus, _, liquid, _, _, _ = values
        upper = y >= -1.0e-12
        valid = (upper & np.isfinite(liquid["barycentricX"]) &
                 np.isfinite(liquid["barycentricY"]))
        ax.scatter(
            liquid["barycentricX"][valid],
            liquid["barycentricY"][valid], c=yplus[valid],
            cmap=yplus_cmap, norm=yplus_norm,
            marker=markers[case_index % len(markers)], s=27,
            linewidths=0.55, edgecolors="black", label=label)
    ax.text(1.02, 0.0, "1C", va="center")
    ax.text(-0.08, 0.0, "2C", va="center")
    ax.text(0.5, np.sqrt(3.0)/2.0 + 0.035, "3C / isotropic",
            ha="center")
    fig.colorbar(mpl.cm.ScalarMappable(norm=yplus_norm, cmap=yplus_cmap),
                 ax=ax, label=r"$y^+$")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 0.95)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "barycentricComparison.png"), dpi=200)
    plt.close(fig)

    if all(values[6] is not None for values in datasets):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        quantity_colors = colors[:3]
        for case_index, (values, label) in enumerate(zip(datasets, labels)):
            y, _, _, _, _, _, temperature = values
            style = linestyles[case_index % len(linestyles)]
            axes[0].plot(
                temperature["all"]["mean"], y,
                color=quantity_colors[0], linestyle=style,
                label=r"$\langle T\rangle$, %s" % label)
            axes[0].plot(
                temperature["all"]["rms"], y,
                color=quantity_colors[1], linestyle=style,
                label=r"$T_{rms}$, %s" % label)
            for component, component_label in enumerate(
                    (r"$\langle u'T'\rangle$", r"$\langle v'T'\rangle$",
                     r"$\langle w'T'\rangle$")):
                axes[1].plot(
                    temperature["all"]["velocityCovariance"][:, component],
                    y, color=quantity_colors[component], linestyle=style,
                    label="%s, %s" % (component_label, label))
        axes[0].set_xlabel("temperature statistic")
        axes[0].set_ylabel(r"$y/\delta$")
        axes[1].set_xlabel("velocity--temperature covariance")
        for axis in axes:
            axis.grid(True, alpha=0.35)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output, "temperatureStatisticsComparison.png"),
            dpi=200)
        plt.close(fig)


def main():
    args = arguments()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if args.plot_only:
        if rank == 0:
            statistics_files = (args.statistics if args.statistics else
                                [os.path.join(args.output,
                                              "statistics.npz")])
            for statistics_file in statistics_files:
                if not os.path.isfile(statistics_file):
                    raise RuntimeError("Statistics file not found: %s" %
                                       statistics_file)
            if args.labels is not None and \
                    len(args.labels) != len(statistics_files):
                raise RuntimeError(
                    "--labels must match the number of --statistics files")
            labels = (args.labels if args.labels is not None else
                      [os.path.basename(os.path.dirname(filename))
                       for filename in statistics_files])
            os.makedirs(args.output, exist_ok=True)
            if len(statistics_files) == 1:
                plot_saved_statistics(statistics_files[0], args.output)
            else:
                plot_saved_comparison(statistics_files, labels, args.output)
            print("Regenerated plots in %s" % args.output)
        return
    files = (resolve_snapshot_inputs(args.input, args.t_min, args.t_max)
             if rank == 0 else None)
    files = comm.bcast(files, root=0)

    with np.load(files[0]) as first:
        x, y, z = first["x"], first["y"], first["z"]
        shape = first["u"].shape
        temperature_available = "T" in first.files
    nx, ny, nz = shape
    plane_points = nx*nz
    local_count = 0

    # Raw profile moments for unconditioned, liquid, and gas statistics.
    w = {name: np.zeros(ny) for name in ("all", "liquid", "gas")}
    m1 = {name: np.zeros((ny, 3)) for name in w}
    m2 = {name: np.zeros((ny, 3, 3)) for name in w}
    pressure_sum = np.zeros(ny)
    alpha_sum = np.zeros(ny)
    if temperature_available:
        temperature_first = {name: np.zeros(ny) for name in w}
        temperature_second = {name: np.zeros(ny) for name in w}
        velocity_temperature = {
            name: np.zeros((ny, 3)) for name in w}
    local_times = []

    for filename in files[rank::size]:
        with np.load(filename) as data:
            velocity = np.stack((data["u"], data["v"], data["w"]), axis=-1)
            cls = np.clip(data["cls"], 0.0, 1.0)
            weights = {
                "all": np.ones_like(cls),
                "liquid": cls,
                "gas": 1.0 - cls,
            }
            for name, weight in weights.items():
                w[name] += np.sum(weight, axis=(0, 2))
                m1[name] += np.sum(weight[..., None]*velocity, axis=(0, 2))
                m2[name] += np.einsum(
                    "xyza,xyzb,xyz->yab", velocity, velocity, weight)

            pressure = data["p"] - np.mean(data["p"])
            pressure_sum += np.sum(pressure, axis=(0, 2))
            alpha_sum += np.sum(1.0 - cls, axis=(0, 2))
            if temperature_available:
                if "T" not in data.files:
                    raise RuntimeError("Temperature is missing from %s" %
                                       filename)
                temperature = data["T"]
                for name, weight in weights.items():
                    temperature_first[name] += np.sum(
                        weight*temperature, axis=(0, 2))
                    temperature_second[name] += np.sum(
                        weight*temperature*temperature, axis=(0, 2))
                    velocity_temperature[name] += np.sum(
                        weight[..., None]*velocity*temperature[..., None],
                        axis=(0, 2))
            local_count += 1
            local_times.append(float(data["time"]))

    for name in w:
        w[name] = reduce_sum(comm, w[name])
        m1[name] = reduce_sum(comm, m1[name])
        m2[name] = reduce_sum(comm, m2[name])
    pressure_sum = reduce_sum(comm, pressure_sum)
    alpha_sum = reduce_sum(comm, alpha_sum)
    if temperature_available:
        for name in w:
            temperature_first[name] = reduce_sum(
                comm, temperature_first[name])
            temperature_second[name] = reduce_sum(
                comm, temperature_second[name])
            velocity_temperature[name] = reduce_sum(
                comm, velocity_temperature[name])
    count = comm.allreduce(local_count, op=MPI.SUM)
    all_times = comm.gather(local_times, root=0)

    statistics = {name: phase_results(w[name], m1[name], m2[name])
                  for name in w}
    temperature_statistics = None
    if temperature_available:
        temperature_statistics = {
            name: phase_temperature_results(
                w[name], temperature_first[name], temperature_second[name],
                velocity_temperature[name], statistics[name]["mean"])
            for name in w
        }
    mean_velocity = statistics["all"]["mean"]
    mean_pressure = pressure_sum/(count*plane_points)
    alpha_gas = alpha_sum/(count*plane_points)
    reynolds = statistics["all"]["reynolds"]
    tke = statistics["all"]["tke"]
    two_phase = np.max(alpha_gas) > 1.0e-8
    budget = None

    if not two_phase:
        # A second pass supplies the third moments and gradients needed by the
        # conventional constant-property single-phase channel budget.
        turbulent_flux = np.zeros(ny)
        pressure_flux = np.zeros(ny)
        dissipation = np.zeros(ny)
        dx = float(x[1] - x[0])
        dz = float(z[1] - z[0])
        viscosity = 2.0/args.re

        for filename in files[rank::size]:
            with np.load(filename) as data:
                velocity = np.stack(
                    (data["u"], data["v"], data["w"]), axis=-1)
                fluctuation = velocity - mean_velocity[None, :, None, :]
                q2 = np.sum(fluctuation**2, axis=-1)
                turbulent_flux += np.sum(
                    0.5*fluctuation[..., 1]*q2, axis=(0, 2))

                pressure = data["p"] - np.mean(data["p"])
                pressure_prime = pressure - mean_pressure[None, :, None]
                pressure_flux += np.sum(
                    pressure_prime*fluctuation[..., 1], axis=(0, 2))

                strain = strain_rate(fluctuation, y, dx, dz)
                instantaneous_dissipation = 2.0*viscosity*np.sum(
                    strain*strain, axis=(-1, -2))
                dissipation += np.sum(
                    instantaneous_dissipation, axis=(0, 2))

        turbulent_flux = reduce_sum(
            comm, turbulent_flux)/(count*plane_points)
        pressure_flux = reduce_sum(
            comm, pressure_flux)/(count*plane_points)
        dissipation = reduce_sum(
            comm, dissipation)/(count*plane_points)

        mean_gradient = np.gradient(
            mean_velocity, y, axis=0, edge_order=2)
        production = -np.sum(
            reynolds[:, :, 1]*mean_gradient, axis=1)
        turbulent_transport = -np.gradient(
            turbulent_flux, y, edge_order=2)
        pressure_transport = -np.gradient(
            pressure_flux, y, edge_order=2)
        viscous_diffusion = viscosity*np.gradient(
            np.gradient(tke, y, edge_order=2), y, edge_order=2)
        residual = (production - dissipation + turbulent_transport
                    + pressure_transport + viscous_diffusion)
        budget = {
            "production": production,
            "dissipation": -dissipation,
            "turbulentTransport": turbulent_transport,
            "pressureTransport": pressure_transport,
            "viscousDiffusion": viscous_diffusion,
            "residual": residual,
        }

    if rank != 0:
        return

    os.makedirs(args.output, exist_ok=True)
    times = np.sort(np.concatenate([np.asarray(t) for t in all_times]))
    yplus = args.re_tau*(1.0 - np.abs(y))
    output = {
        "time": times, "y": y, "yplus": yplus,
        "alphaGas": alpha_gas,
        "meanVelocity": mean_velocity,
        "meanPressure": mean_pressure,
        "reynoldsStress": reynolds,
        "tke": tke,
        "reTau": np.array(args.re_tau),
        "twoPhase": np.array(two_phase),
    }
    for phase in ("liquid", "gas"):
        for name, value in statistics[phase].items():
            output[phase + name[0].upper() + name[1:]] = value
    if temperature_available:
        output["meanTemperature"] = temperature_statistics["all"]["mean"]
        output["temperatureVariance"] = temperature_statistics["all"][
            "variance"]
        output["temperatureRms"] = temperature_statistics["all"]["rms"]
        output["turbulentHeatFlux"] = temperature_statistics["all"][
            "velocityCovariance"]
        for phase in ("liquid", "gas"):
            prefix = phase + "Temperature"
            output[prefix + "Mean"] = temperature_statistics[phase]["mean"]
            output[prefix + "Variance"] = temperature_statistics[phase][
                "variance"]
            output[prefix + "Rms"] = temperature_statistics[phase]["rms"]
            output[prefix + "VelocityCovariance"] = \
                temperature_statistics[phase]["velocityCovariance"]
    if budget is not None:
        for name, value in budget.items():
            output["budget" + name[0].upper() + name[1:]] = value
    np.savez(os.path.join(args.output, "statistics.npz"), **output)

    profile = np.column_stack((
        y, yplus, alpha_gas,
        statistics["liquid"]["tke"],
        statistics["liquid"]["anisotropy"][:, 0, 0],
        statistics["liquid"]["anisotropy"][:, 1, 1],
        statistics["liquid"]["anisotropy"][:, 2, 2],
        statistics["liquid"]["flatness"],
        statistics["liquid"]["barycentricX"],
        statistics["liquid"]["barycentricY"],
    ))
    np.savetxt(os.path.join(args.output, "anisotropyProfile.dat"), profile,
               header="y yplus alphaGas kLiquid b11 b22 b33 F xB yB")

    if temperature_available:
        temperature_profile = np.column_stack((
            y, yplus,
            temperature_statistics["all"]["mean"],
            temperature_statistics["all"]["rms"],
            temperature_statistics["all"]["velocityCovariance"],
            temperature_statistics["liquid"]["mean"],
            temperature_statistics["liquid"]["rms"],
            temperature_statistics["gas"]["mean"],
            temperature_statistics["gas"]["rms"],
        ))
        np.savetxt(
            os.path.join(args.output, "temperatureProfile.dat"),
            temperature_profile,
            header=("y yplus meanT Trms uPrimeTPrime vPrimeTPrime "
                    "wPrimeTPrime liquidMeanT liquidTrms gasMeanT gasTrms"))

    if budget is not None:
        budget_table = np.column_stack((
            y, yplus, production, -dissipation, turbulent_transport,
            pressure_transport, viscous_diffusion, residual))
        np.savetxt(os.path.join(args.output, "tkeBudget.dat"), budget_table,
                   header=("y yplus production minusDissipation "
                           "turbulentTransport pressureTransport "
                           "viscousDiffusion resolvedResidual"))
    plot_results(args.output, y, yplus, statistics["all"],
                 statistics["liquid"], statistics["gas"], budget,
                 temperature_statistics)
    print("Wrote statistics to %s" % args.output)
    if two_phase:
        print("Two-phase case detected: no TKE budget was generated because "
              "the available checkpoint fields cannot close it.")


if __name__ == "__main__":
    start = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- statistics completed in %.2f s ---" %
              (time.time() - start))
